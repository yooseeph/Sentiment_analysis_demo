#!/usr/bin/env python3
"""
Single Audio File Parallel Chunk Processing

Process a single audio file by splitting it into chunks and processing
multiple chunks simultaneously on GPU for maximum speed.
"""

import os
import torchaudio
import torch
from tqdm import tqdm
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
import argparse
import subprocess
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
import json

from utils.logging_config import get_logger
logger = get_logger(__name__)

def remove_special_characters(text):
    import re
    if text is None:
        return ""
    chars_to_remove_regex = r'[\,\?\.\!\-\;:\"%\'\»\«\؟\(\)،\.]'
    return re.sub(chars_to_remove_regex, '', text.lower())

def load_audio_any_format(audio_path):
    """Load audio from .wav or .ogg format."""
    if str(audio_path).lower().endswith('.ogg'):
        command = [
            "ffmpeg", "-i", str(audio_path), "-f", "wav", "-acodec", "pcm_s16le", "-"
        ]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        wav_bytes, _ = proc.communicate()
        waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
        return waveform, sample_rate
    else:
        return torchaudio.load(audio_path)

def split_audio_into_chunks(waveform, sample_rate, chunk_duration_sec=30, overlap_sec=2):
    """
    Split audio into overlapping chunks for parallel processing.
    
    Args:
        waveform: Audio tensor [channels, samples]
        sample_rate: Sample rate
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between chunks in seconds
    
    Returns:
        List of (chunk_waveform, start_time, end_time) tuples
    """
    chunk_samples = int(chunk_duration_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    total_samples = waveform.shape[1]
    chunks = []
    
    start = 0
    chunk_idx = 0
    
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        
        # Extract chunk
        chunk_waveform = waveform[:, start:end]
        
        # Calculate time positions
        start_time = start / sample_rate
        end_time = end / sample_rate
        
        chunks.append({
            'waveform': chunk_waveform,
            'start_time': start_time,
            'end_time': end_time,
            'chunk_idx': chunk_idx,
            'start_sample': start,
            'end_sample': end
        })
        
        chunk_idx += 1
        start += step_samples
        
        # Break if we've covered the entire audio
        if end >= total_samples:
            break
    
    return chunks

def preprocess_chunk_batch(chunks, target_sample_rate=16000):
    """
    Preprocess a batch of audio chunks in parallel.
    """
    def preprocess_single_chunk(chunk_info):
        try:
            waveform = chunk_info['waveform']
            
            # Resample if needed
            if chunk_info.get('original_sample_rate', target_sample_rate) != target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, chunk_info['original_sample_rate'], target_sample_rate
                )
            
            # Ensure audio is normalized to [-1.0, 1.0] float
            if waveform.dtype != torch.float32:
                waveform = waveform.to(torch.float32) / 32768.0
            waveform = torch.clamp(waveform, -1.0, 1.0)
            
            # Convert stereo to mono if needed (for main transcription)
            if waveform.shape[0] > 1:
                mono_waveform = waveform.mean(dim=0, keepdim=True)
            else:
                mono_waveform = waveform
            
            chunk_info['mono_waveform'] = mono_waveform
            chunk_info['stereo_waveform'] = waveform
            return chunk_info
            
        except Exception as e:
            chunk_info['error'] = str(e)
            return chunk_info
    
    # Process chunks in parallel using threading
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
        processed_chunks = list(executor.map(preprocess_single_chunk, chunks))
    
    return processed_chunks

def pad_chunk_waveforms(waveforms):
    """Pad chunk waveforms to the same length for batch processing."""
    if not waveforms:
        return torch.empty(0)
    
    max_length = max(wf.shape[1] for wf in waveforms)
    padded_waveforms = []
    
    for waveform in waveforms:
        if waveform.shape[1] < max_length:
            padding = max_length - waveform.shape[1]
            padded = torch.nn.functional.pad(waveform, (0, padding))
        else:
            padded = waveform
        padded_waveforms.append(padded)
    
    return torch.stack(padded_waveforms)

def transcribe_chunk_batch_gpu(model, processor, chunks, device, batch_size=8):
    """
    Transcribe a batch of audio chunks on GPU.
    
    Args:
        model: Transcription model
        processor: Audio processor
        chunks: List of preprocessed chunk dictionaries
        device: CUDA device
        batch_size: GPU batch size
    
    Returns:
        List of chunks with transcription results
    """
    # Filter out chunks with errors
    valid_chunks = [chunk for chunk in chunks if 'error' not in chunk]
    error_chunks = [chunk for chunk in chunks if 'error' in chunk]
    
    if not valid_chunks:
        return chunks
    
    # Extract mono waveforms for transcription
    mono_waveforms = [chunk['mono_waveform'] for chunk in valid_chunks]
    
    # Process in GPU batches
    results = []
    
    for i in range(0, len(mono_waveforms), batch_size):
        batch_waveforms = mono_waveforms[i:i+batch_size]
        batch_chunks = valid_chunks[i:i+batch_size]
        
        try:
            # Pad waveforms to same length
            padded_waveforms = pad_chunk_waveforms(batch_waveforms)
            
            # Convert to numpy arrays for processor
            batch_arrays = [wf.squeeze().numpy() for wf in padded_waveforms]
            
            # Process batch on GPU
            inputs = processor(batch_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            input_features = inputs["input_features"].to(device)
            
            with torch.no_grad():
                logits = model(input_features=input_features).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_texts = processor.batch_decode(predicted_ids)
            
            # Clean texts and add to chunks
            for j, (chunk, text) in enumerate(zip(batch_chunks, predicted_texts)):
                chunk['transcription'] = remove_special_characters(text)
            
            results.extend(batch_chunks)
            
            # Cleanup GPU memory
            del input_features, logits, inputs, padded_waveforms
            torch.cuda.empty_cache()
            
        except Exception as e:
            # Mark batch chunks as failed
            for chunk in batch_chunks:
                chunk['transcription'] = None
                chunk['error'] = f"GPU processing failed: {str(e)}"
            results.extend(batch_chunks)
    
    # Add back error chunks
    results.extend(error_chunks)
    
    # Sort by chunk index to maintain order
    results.sort(key=lambda x: x['chunk_idx'])
    
    return results

def transcribe_stereo_chunks_gpu(model, processor, chunks, device, batch_size=8):
    """
    Transcribe stereo channels (agent/client) from chunks on GPU.
    """
    # Filter chunks that have stereo audio
    stereo_chunks = []
    stereo_chunk_indices = []  # Keep track of original indices
    
    for chunk in chunks:
        if 'stereo_waveform' in chunk:
            try:
                # Split stereo channels
                if chunk['stereo_waveform'].shape[0] >= 2:     
                    agent_waveform = chunk['stereo_waveform'][0].unsqueeze(0)
                    client_waveform = chunk['stereo_waveform'][1].unsqueeze(0)
                else:
                    raise Exception("Stereo waveform has less than 2 channels")
                
                
                stereo_chunks.append({
                    **chunk,
                    'agent_waveform': agent_waveform,
                    'client_waveform': client_waveform
                })
                stereo_chunk_indices.append(chunk['chunk_idx'])
            except Exception:
                continue
    
    if not stereo_chunks:
        return chunks
    
    # Process agent channel
    agent_waveforms = [chunk['agent_waveform'] for chunk in stereo_chunks]
    agent_results = transcribe_waveform_batch(model, processor, agent_waveforms, device, batch_size)
    
    # Process client channel  
    client_waveforms = [chunk['client_waveform'] for chunk in stereo_chunks]
    client_results = transcribe_waveform_batch(model, processor, client_waveforms, device, batch_size)
    
    # Create mapping from chunk_idx to result index
    chunk_idx_to_result_idx = {chunk_idx: i for i, chunk_idx in enumerate(stereo_chunk_indices)}
    
    # Add results back to chunks
    for chunk in chunks:
        chunk_idx = chunk['chunk_idx']
        if chunk_idx in chunk_idx_to_result_idx:
            result_idx = chunk_idx_to_result_idx[chunk_idx]
            chunk['agent_transcription'] = agent_results[result_idx] if result_idx < len(agent_results) else None
            chunk['client_transcription'] = client_results[result_idx] if result_idx < len(client_results) else None
        else:
            chunk['agent_transcription'] = None
            chunk['client_transcription'] = None
    
    return chunks

def transcribe_waveform_batch(model, processor, waveforms, device, batch_size):
    """Helper function to transcribe a batch of waveforms."""
    results = []
    
    for i in range(0, len(waveforms), batch_size):
        batch_waveforms = waveforms[i:i+batch_size]
        
        try:
            padded_waveforms = pad_chunk_waveforms(batch_waveforms)
            batch_arrays = [wf.squeeze().numpy() for wf in padded_waveforms]
            
            inputs = processor(batch_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            input_features = inputs["input_features"].to(device)
            
            with torch.no_grad():
                logits = model(input_features=input_features).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_texts = processor.batch_decode(predicted_ids)
            
            cleaned_texts = [remove_special_characters(text) for text in predicted_texts]
            results.extend(cleaned_texts)
            
            # Cleanup
            del input_features, logits, inputs, padded_waveforms
            torch.cuda.empty_cache()
            
        except Exception as e:
            # Add None for failed batch
            results.extend([None] * len(batch_waveforms))
    
    return results

def merge_chunk_transcriptions(chunks, overlap_sec=1):
    """
    Merge transcriptions from overlapping chunks, handling overlap intelligently.
    """
    if not chunks:
        return ""
    
    # Sort chunks by start time
    sorted_chunks = sorted(chunks, key=lambda x: x['start_time'])
    
    merged_text = ""
    last_end_time = 0
    
    for chunk in sorted_chunks:
        transcription = chunk.get('transcription', '')
        if not transcription:
            continue
        
        # For overlapping chunks, we need to handle the overlap
        if chunk['start_time'] < last_end_time and merged_text:
            # This chunk overlaps with previous one
            # We could implement smart merging here, but for now just add with space
            if not merged_text.endswith(' '):
                merged_text += " "
            merged_text += transcription
        else:
            # No overlap or first chunk
            if merged_text and not merged_text.endswith(' '):
                merged_text += " "
            merged_text += transcription
        
        last_end_time = chunk['end_time']
    
    return merged_text.strip()

def transcribe_single_audio_parallel(audio_path, model_path=None, chunk_duration_sec=25, 
                                   overlap_sec=1, batch_size=8, diarize=True):
    """
    Main function to transcribe a single audio file using parallel chunk processing.
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the transcription model
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between chunks in seconds  
        batch_size: GPU batch size for processing chunks
        diarize: Whether to perform diarization (stereo channels)
    
    Returns:
        Tuple of (transcribed_chunks, merged_transcription)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    import os
    if model_path is None:
        model_path = os.getenv("TRANSCRIPTION_MODEL_PATH", "./models/transcription/w2v-bert-darija-finetuned-clean/")
    processor = Wav2Vec2BertProcessor.from_pretrained(model_path)
    model = Wav2Vec2BertForCTC.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        attn_implementation="eager"
    ).to(device)
    model.eval()
    
    # Load audio
    logger.info(f"Loading audio: {audio_path}")
    try:
        waveform, sample_rate = load_audio_any_format(audio_path)
        if waveform.shape[0] < 2:
            raise Exception("Audio has less than 2 channels")
        audio_duration = waveform.shape[1] / sample_rate
        logger.info(f"Audio duration: {audio_duration:.1f} seconds")
    except Exception as e:
        return {"error": f"Failed to load audio: {str(e)}"}
    
    # Split into chunks
    logger.info(f"Splitting into chunks (duration={chunk_duration_sec}s, overlap={overlap_sec}s)...")
    chunks = split_audio_into_chunks(waveform, sample_rate, chunk_duration_sec, overlap_sec)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Preprocess chunks in parallel
    logger.info("Preprocessing chunks...")
    for chunk in chunks:
        chunk['original_sample_rate'] = sample_rate
    processed_chunks = preprocess_chunk_batch(chunks)
    
    # Transcribe chunks in GPU batches
    logger.info(f"Transcribing chunks (batch_size={batch_size})...")
    with tqdm(total=len(processed_chunks), desc="Processing chunks") as pbar:
        
        # Process chunks in batches for progress tracking
        batch_size_progress = max(batch_size * 2, 8)  # Larger batches for progress
        transcribed_chunks = []
        
        for i in range(0, len(processed_chunks), batch_size_progress):
            batch_chunks = processed_chunks[i:i+batch_size_progress]
            
            # Transcribe this batch
            batch_results = transcribe_chunk_batch_gpu(
                model, processor, batch_chunks, device, batch_size
            )
            transcribed_chunks.extend(batch_results)
            
            pbar.update(len(batch_chunks))
    
    # Transcribe stereo channels if diarization is enabled
    if diarize:
        logger.info("Processing stereo channels for diarization...")
        transcribed_chunks = transcribe_stereo_chunks_gpu(
            model, processor, transcribed_chunks, device, batch_size
        )
    
    # Merge transcriptions
    logger.info("Merging chunk transcriptions...")
    merged_transcription = merge_chunk_transcriptions(transcribed_chunks, overlap_sec)
    
    
    return transcribed_chunks, merged_transcription
