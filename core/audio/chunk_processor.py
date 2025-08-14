"""
Chunk processing module for Sentiment Analysis Dashboard
Handles audio chunking and parallel transcription
"""
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import torch
import torchaudio

from config import config
from utils.logging_config import get_logger, log_performance
from core.audio import audio_processor
from core.sentiment import sentiment_analyzer
from core.state_manager import get_session_state, ChunkMetadata
from core.utils import darija_converter, append_rows_to_csv, analyze_topic
import gradio as gr

logger = get_logger(__name__)


class ChunkProcessor:
    """Handles audio chunk processing operations"""
    
    def __init__(self):
        self.audio_proc = audio_processor
        self.sentiment = sentiment_analyzer
        self.converter = darija_converter
        
    @log_performance
    def extract_chunk_from_original(
        self,
        chunk_id: str,
        request: Optional[gr.Request] = None
    ) -> Dict[str, Any]:
        """
        Extract a specific chunk from the original audio file with enhanced caching
        
        Args:
            chunk_id: ID of the chunk to extract
            request: Gradio request object
            
        Returns:
            Dictionary containing chunk data
        """
        state = get_session_state(request)
        
        # Check if chunk is already in memory
        if chunk_id in state.chunks_in_memory:
            logger.info(f"Using cached chunk: {chunk_id}")
            return state.chunks_in_memory[chunk_id]
        
        # Find chunk metadata
        chunk_info = None
        for chunk in state.chunks_metadata:
            if chunk.id == chunk_id:
                chunk_info = chunk
                break
        
        if not chunk_info or not state.original_audio_path:
            raise ValueError(f"Chunk {chunk_id} not found or original audio path not set")
        
        logger.info(f"Extracting chunk {chunk_id} from original audio...")
        
        # Load original audio
        waveform, sample_rate = torchaudio.load(state.original_audio_path)
        
        # Extract the specific chunk using timing information
        start_time = chunk_info.start_time  # in seconds
        end_time = chunk_info.end_time      # in seconds
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract chunk waveform
        chunk_waveform = waveform[:, start_sample:end_sample]
        
        # Create temporary file for this chunk
        chunk_path = self.audio_proc.save_chunk_to_file(chunk_waveform, sample_rate, request)
        
        # Store in memory cache for future use
        chunk_data = {
            'audio_path': chunk_path,
            'waveform': chunk_waveform,
            'sample_rate': sample_rate,
            'agent_text': chunk_info.agent_text,
            'client_text': chunk_info.client_text,
            'sentiments': chunk_info.sentiments
        }
        
        state.cache_chunk(chunk_id, chunk_data)
        return chunk_data
    
    def transcribe_segments(
        self,
        segments: List[Dict[str, Any]],
        orig_sr: int,
        target_sr: int,
        request: Optional[gr.Request] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio segments
        
        Args:
            segments: List of audio segments
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            List of transcribed segments
        """
        results = []
        
        for seg in segments:
            try:
                # Resample if needed
                chunk = seg["chunk"]
                if orig_sr != target_sr:
                    chunk = self.audio_proc.resample_audio(chunk, orig_sr, target_sr)
                
                # Transcribe
                raw_text = self.audio_proc.transcribe_chunk(chunk, target_sr).strip()
                
                if raw_text:
                    # Save chunk to temp file
                    temp_audio_path = self.audio_proc.save_chunk_to_file(chunk, target_sr, request)
                    
                    # Convert Darija to French
                    raw_text = self.converter.convert_text(raw_text)
                    
                    results.append({
                        **seg,
                        "text": raw_text,
                        "audio_path": temp_audio_path
                    })
                    
            except Exception as e:
                logger.error(f"Error processing segment: {e}")
                continue
        
        return results
    
    @log_performance
    def optimized_chunker(
        self,
        audio_path: str,
        topic: bool = True,
        request: Optional[gr.Request] = None
    ) -> Tuple[List[str], List[tuple], str]:
        """
        Optimized chunker that avoids redundant processing
        
        Args:
            audio_path: Path to audio file
            topic: Whether to analyze topic
            request: Gradio request object
            
        Returns:
            Tuple of (chunk_keys, sentiments, topic_result)
        """
        state = get_session_state(request)
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            except:
                pass
                
        csv_rows = []
        
        # Validate audio file format
        # is_valid, validation_message = self.audio_proc.validate_audio_file(audio_path)
        # if not is_valid:
        #     logger.error(f"Audio validation failed: {validation_message}")
        #     return [], [], f"Erreur: {validation_message}"
        
        # Check if we already analyzed this file
        if state.is_same_file(audio_path) and state.full_call_analysis:
            logger.info("Using cached call analysis - no reprocessing needed!")
            cached = state.full_call_analysis
            return cached['chunk_keys'], cached['sentiments'], cached['topic_result']
        
        # Set current file and clear cache if different
        state.set_current_file(audio_path)
        
        # Transcription using chunker module
        try:
            from utils.chunker import transcribe_single_audio_parallel
            chunks_data, call_transcription = transcribe_single_audio_parallel(audio_path)
            logger.info(f"Chunker returned {len(chunks_data)} chunks")
            
            # Check if chunks_data is valid
            if not chunks_data:
                logger.warning("No chunks returned from chunker")
                return [], [], "Aucun chunk retourné par le chunker"
                
        except Exception as e:
            logger.error(f"Error in transcribe_single_audio_parallel: {e}", exc_info=True)
            return [], [], f"Erreur dans le chunker: {str(e)}"
        
        chunk_keys = []
        sentiments = []

        for i, chunk in enumerate(chunks_data):
            chunk_id = f"chunk_{i}"
            
            # Handle None values robustly
            agent_text = chunk.get('agent_transcription', '') or ''
            client_text = chunk.get('client_transcription', '') or ''

            # Get the waveform from the chunk
            waveform = None
            for key in ['stereo_waveform', 'mono_waveform', 'waveform']:
                if key in chunk and chunk[key] is not None:
                    waveform = chunk[key]
                    break
                    
            if waveform is None:
                sentiments.append((None, None, None, None, None, None, {}, {}, None, None))
                chunk_keys.append(chunk_id)
                continue
            
            logger.debug(f"Chunk {i}: agent='{agent_text}', client='{client_text}'")
            
            # Skip chunks without content
            if not agent_text.strip() and not client_text.strip():
                sentiments.append((None, None, None, None, None, None, {}, {}, None, None))
                chunk_keys.append(chunk_id)
                continue

            # Create temporary file for sentiment analysis
            temp_chunk_path = self.audio_proc.save_chunk_to_file(waveform, 16000, request)

            # Perform sentiment analysis
            sentiment = self.sentiment.analyze_sentiment_client_agent(
                agent_text, client_text, temp_chunk_path
            )
            sentiments.append(sentiment)
            
            # Store chunk metadata
            chunk_metadata = ChunkMetadata(
                id=chunk_id,
                start_time=chunk.get('start_time', i * config.audio.chunk_duration_sec),
                end_time=chunk.get('end_time', (i + 1) * config.audio.chunk_duration_sec),
                agent_text=agent_text,
                client_text=client_text,
                sentiments=sentiment
            )
            state.chunks_metadata.append(chunk_metadata)
            chunk_keys.append(chunk_id)
            
            # Prepare CSV row
            csv_rows.append({
                "audio_id": Path(audio_path).stem,
                "chunk_id": chunk_id,
                "transcription_agent": agent_text,
                "transcription_client": client_text,
                "text_label_agent": sentiment[0],
                "prob_text_agent": sentiment[6].get('text', {}).get('prob', []) if sentiment[6] else [],
                "acoustic_label_agent": sentiment[1],
                "prob_acoustic_agent": sentiment[6].get('acoustic', {}).get('prob', []) if sentiment[6] else [],
                "final_label_agent": sentiment[2],
                "fusion_prob_agent": sentiment[8],
                "text_label_client": sentiment[3],
                "prob_text_client": sentiment[7].get('text', {}).get('prob', []) if sentiment[7] else [],
                "acoustic_label_client": sentiment[4],
                "prob_acoustic_client": sentiment[7].get('acoustic', {}).get('prob', []) if sentiment[7] else [],
                "final_label_client": sentiment[5],
                "fusion_prob_client": sentiment[9],
            })
            
            # Clean up temporary file
            try:
                os.remove(temp_chunk_path)
            except:
                pass

        # Analyze global topic
        topic_result = "appel blanc"
        if topic:
            topic_result = analyze_topic(call_transcription)

        # Cache the complete analysis
        state.full_call_analysis = {
            'chunk_keys': chunk_keys,
            'sentiments': sentiments,
            'topic_result': topic_result
        }
        
        logger.info(f"Analysis complete and cached for file: {audio_path}")
        
        # Save to CSV
        # append_rows_to_csv(csv_rows, request)
        
        return chunk_keys, sentiments, topic_result


# Global chunk processor instance
chunk_processor = ChunkProcessor() 