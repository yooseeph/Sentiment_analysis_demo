"""
Audio processing module for Sentiment Analysis Dashboard
Handles audio loading, validation, transcription, and VAD
"""
import os
import tempfile
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path

import torch
import torchaudio
import io
import subprocess
from torchaudio.transforms import Resample
import numpy as np

from config import config
from utils.logging_config import get_logger, log_audio_processing, log_performance
from core.models.model_loader import get_transcription_model, get_vad_pipeline

logger = get_logger(__name__)


class AudioProcessor:
    """Handles audio processing operations"""
    
    def __init__(self):
        self.device = torch.device(config.performance.gpu_device)
        self.resampler_cache = {}
        logger.info(f"AudioProcessor initialized with device: {self.device}")
        
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that the audio file is in a supported format
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not file_path:
            return False, "No file provided"
        
        # Check file extension
        file_extension = os.path.splitext(file_path.lower())[1]
        
        if file_extension not in config.audio.supported_formats:
            return False, (
                f"Unsupported format: {file_extension}. "
                f"Supported formats: {', '.join(config.audio.supported_formats)}"
            )
        
        # Try to load the file to ensure it's valid
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            if waveform.size(0) == 0 or waveform.size(1) == 0:
                return False, "Audio file is empty or corrupted"
            
            if waveform.shape[0] < 2:
                raise Exception("Audio has less than 2 channels please select a stereo audio file")
            
            duration = waveform.shape[1] / sample_rate
            log_audio_processing(file_path, duration, sample_rate)
            
            logger.info(
                f"Audio file validated: {file_extension.upper()}, "
                f"{sample_rate}Hz, {waveform.shape[0]} channels, "
                f"{waveform.shape[1]} samples, {duration:.1f}s"
            )
            
            return True, "Valid audio file"
            
        except Exception as e:
            logger.error(f"Failed to validate audio file: {e}")
            return False, f"Error reading audio file: {str(e)}"
    
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            if str(file_path).lower().endswith('.ogg'):
                # Use subprocess for OGG files
                command = [
                    "ffmpeg", "-i", str(file_path), 
                    "-f", "wav", "-acodec", "pcm_s16le", "-"
                ]
                proc = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                wav_bytes, _ = proc.communicate()
                waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
            else:
                # Use memory mapping for large files
                waveform, sample_rate = torchaudio.load(
                    str(file_path)
                )
            
            return waveform, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def load_audio_channels(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Load audio file and separate channels
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (left_channel, right_channel, sample_rate)
        """
        waveform, sample_rate = self.load_audio(file_path)
            
        return waveform[0].unsqueeze(0), waveform[1].unsqueeze(0), sample_rate
    
    def resample_audio(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate
        
        Args:
            waveform: Audio waveform tensor
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled waveform
        """            
        # Use cached resampler if available
        key = (orig_sr, target_sr)
        if key not in self.resampler_cache:
            self.resampler_cache[key] = Resample(orig_freq=orig_sr, new_freq=target_sr)
            
        return self.resampler_cache[key](waveform)
    
    @log_performance
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio
            waveform, sr = self.load_audio(audio_path)
                           
            # Resample if needed
            if sr != config.audio.target_sample_rate:
                waveform = self.resample_audio(waveform, sr, config.audio.target_sample_rate)
                sr = config.audio.target_sample_rate
                
            # Get model and processor
            model, processor = get_transcription_model()
            
            # Process audio
            inputs = processor(
                waveform.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            
            input_features = inputs["input_features"].to(self.device)
            
            # Transcribe
            with torch.no_grad():
                logits = model(input_features).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            logger.info(f"Transcribed audio: {len(transcription)} characters")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def transcribe_chunk(self, chunk: torch.Tensor, sample_rate: int) -> str:
        """
        Transcribe a single audio chunk
        
        Args:
            chunk: Audio chunk tensor
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        try:
            # Get model and processor
            model, processor = get_transcription_model()
            
            # Process chunk
            inputs = processor(
                chunk.squeeze().cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            input_features = inputs["input_features"].to(self.device)
            
            # Transcribe
            with torch.no_grad():
                logits = model(input_features).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
    
    @log_performance
    def get_speech_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        speaker_label: str
    ) -> List[Dict[str, Any]]:
        """
        Extract speech segments using voice activity detection
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            speaker_label: Speaker label (Agent/Client)
            
        Returns:
            List of speech segments
        """
        path = None
        try:
            # Get VAD pipeline
            vad_pipeline = get_vad_pipeline()
            
            # Save waveform to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
                torchaudio.save(tmp.name, waveform, sample_rate)

            # Run VAD
            vad_result = vad_pipeline(path)
            timeline = list(vad_result.get_timeline())
            
            if not timeline:
                logger.warning(f"No speech segments found for {speaker_label}")
                return []

            # Merge close segments
            merged_segments = self._merge_segments(
                timeline,
                gap_threshold=config.audio.vad_gap_threshold
            )

            # Extract audio chunks with padding
            segments = self._extract_segments(
                waveform,
                sample_rate,
                merged_segments,
                speaker_label,
                padding=config.audio.vad_padding
            )
            
            logger.info(f"Extracted {len(segments)} speech segments for {speaker_label}")
            return segments

        except Exception as e:
            logger.error(f"Error in speech segmentation: {e}")
            return []
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
    
    def _merge_segments(self, timeline: List[Any], gap_threshold: float) -> List[Tuple[float, float]]:
        """Merge segments that are close together"""
        if not timeline:
            return []
            
        merged = []
        current_start = timeline[0].start
        current_end = timeline[0].end
        
        for turn in timeline[1:]:
            if turn.start - current_end <= gap_threshold:
                current_end = turn.end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = turn.start, turn.end
                
        merged.append((current_start, current_end))
        return merged
    
    def _extract_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segments: List[Tuple[float, float]],
        speaker_label: str,
        padding: float
    ) -> List[Dict[str, Any]]:
        """Extract audio segments with padding"""
        total_duration = waveform.shape[1] / sample_rate
        extracted = []
        
        for start, end in segments:
            padded_start = max(0.0, start - padding)
            padded_end = min(total_duration, end + padding)
            start_sample = int(padded_start * sample_rate)
            end_sample = int(padded_end * sample_rate)
            
            extracted.append({
                "chunk": waveform[:, start_sample:end_sample],
                "start": padded_start,
                "end": padded_end,
                "speaker": speaker_label
            })

        return extracted
    
    def save_chunk_to_file(self, chunk: torch.Tensor, sample_rate: int, request: Optional[Any] = None) -> str:
        """
        Save audio chunk to temporary file
        
        Args:
            chunk: Audio chunk tensor
            sample_rate: Sample rate
            
        Returns:
            Path to temporary file
        """
        tmp_dir = os.environ.get("GRADIO_TEMP_DIR") or os.environ.get("TMPDIR")
        kwargs = {"suffix": ".wav", "delete": False}
        if tmp_dir:
            kwargs["dir"] = tmp_dir
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            torchaudio.save(tmp.name, chunk, sample_rate)
            try:
                # Register temp file to session for scoped cleanup
                from core.state_manager import get_session_state
                state = get_session_state(request)
                state.register_temp_file(tmp.name)
            except Exception:
                pass
            return tmp.name


# Global audio processor instance
audio_processor = AudioProcessor() 