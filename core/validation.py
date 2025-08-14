"""
Input validation for Sentiment Analysis Dashboard
"""
import os
from typing import Tuple, List, Optional, Union
from pathlib import Path
import re

from config import config
from core.exceptions import ValidationException
from utils.logging_config import get_logger

logger = get_logger(__name__)


class AudioValidator:
    """Validates audio files and parameters"""
    
    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, str]:
        """
        Validate file path
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not file_path:
            return False, "Aucun fichier fourni"
            
        # Clean path
        file_path = file_path.strip('"\'')
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"Fichier introuvable: {file_path}"
            
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            return False, f"Le chemin n'est pas un fichier: {file_path}"
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Le fichier est vide"
            
        # Max file size: 500MB
        max_size = 500 * 1024 * 1024
        if file_size > max_size:
            return False, f"Fichier trop volumineux: {file_size / 1024 / 1024:.1f}MB (max: 500MB)"
            
        return True, "Chemin valide"
    
    @staticmethod
    def validate_audio_format(file_path: str) -> Tuple[bool, str]:
        """
        Validate audio file format
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check extension
        file_extension = os.path.splitext(file_path.lower())[1]
        
        if file_extension not in config.audio.supported_formats:
            return False, (
                f"Format non supporté: {file_extension}. "
                f"Formats acceptés: {', '.join(config.audio.supported_formats)}"
            )
            
        return True, "Format valide"
    
    @staticmethod
    def validate_duration(duration_seconds: float) -> Tuple[bool, str]:
        """
        Validate audio duration
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Minimum duration: 1 second
        if duration_seconds < 1.0:
            return False, "Audio trop court (minimum: 1 seconde)"
            
        # Maximum duration: 2 hours
        max_duration = 2 * 60 * 60
        if duration_seconds > max_duration:
            return False, f"Audio trop long: {duration_seconds / 60:.1f} minutes (max: 2 heures)"
            
        return True, "Durée valide"


class TextValidator:
    """Validates text inputs"""
    
    @staticmethod
    def validate_transcription(text: str) -> Tuple[bool, str]:
        """
        Validate transcription text
        
        Args:
            text: Transcription text
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not text or not text.strip():
            return False, "Transcription vide"
            
        # Check length
        if len(text) > 100000:  # 100k characters
            return False, "Transcription trop longue (max: 100000 caractères)"
            
        return True, "Transcription valide"
    
    @staticmethod
    def validate_dates(dates: List[str], expected_count: int) -> Tuple[bool, str]:
        """
        Validate date inputs
        
        Args:
            dates: List of date strings
            expected_count: Expected number of dates
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(dates) != expected_count:
            return False, (
                f"Nombre de dates incorrect: {len(dates)} "
                f"(attendu: {expected_count})"
            )
            
        # Validate date format
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        
        for i, date in enumerate(dates):
            if not date_pattern.match(date):
                return False, f"Format de date invalide ligne {i+1}: '{date}' (format attendu: YYYY-MM-DD)"
                
        return True, "Dates valides"


class ConfigValidator:
    """Validates configuration settings"""
    
    @staticmethod
    def validate_model_paths() -> List[str]:
        """
        Validate that required model paths exist
        
        Returns:
            List of missing model paths
        """
        missing = []
        
        # Check transcription model
        if not Path(config.model.transcription_model_path).exists():
            missing.append(f"Transcription model: {config.model.transcription_model_path}")
            
        # Check sentiment models
        model_paths = [
            ("Agent text", config.model.agent_text_model_path),
            ("Agent acoustic", config.model.agent_acoustic_model_path),
            ("Client text", config.model.client_text_model_path),
            ("Client acoustic", config.model.client_acoustic_model_path),
        ]
        
        for name, path in model_paths:
            if not Path(path).exists():
                missing.append(f"{name} model: {path}")
                
        return missing
    
    @staticmethod
    def validate_aws_config() -> Tuple[bool, str]:
        """
        Validate AWS configuration
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not config.aws.is_configured:
            return False, "AWS credentials not configured"
            
        # Check if credentials look valid (basic check)
        if len(config.aws.access_key_id) < 10:
            return False, "AWS Access Key ID appears invalid"
            
        if len(config.aws.secret_access_key) < 20:
            return False, "AWS Secret Access Key appears invalid"
            
        return True, "AWS configuration valid"


def validate_audio_input(
    audio_path: str,
    check_content: bool = True
) -> None:
    """
    Comprehensive audio input validation
    
    Args:
        audio_path: Path to audio file
        check_content: Whether to check file content
        
    Raises:
        ValidationException: If validation fails
    """
    # Validate path
    is_valid, message = AudioValidator.validate_file_path(audio_path)
    if not is_valid:
        raise ValidationException(message)
        
    # Validate format
    is_valid, message = AudioValidator.validate_audio_format(audio_path)
    if not is_valid:
        raise ValidationException(message)
        
    # Validate content if requested
    if check_content:
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Check if empty
            if waveform.size(0) == 0 or waveform.size(1) == 0:
                raise ValidationException("Fichier audio vide ou corrompu")
                
            # Validate duration
            duration = waveform.shape[1] / sample_rate
            is_valid, message = AudioValidator.validate_duration(duration)
            if not is_valid:
                raise ValidationException(message)
                
        except Exception as e:
            if isinstance(e, ValidationException):
                raise
            raise ValidationException(f"Erreur lors de la lecture du fichier audio: {str(e)}")


def validate_chunk_id(chunk_id: str, valid_chunks: List[str]) -> None:
    """
    Validate chunk ID
    
    Args:
        chunk_id: Chunk ID to validate
        valid_chunks: List of valid chunk IDs
        
    Raises:
        ValidationException: If validation fails
    """
    if not chunk_id:
        raise ValidationException("ID de chunk non fourni")
        
    if chunk_id not in valid_chunks:
        raise ValidationException(
            f"ID de chunk invalide: {chunk_id}. "
            f"Chunks valides: {', '.join(valid_chunks[:5])}..."
        )


def validate_session_state(state: Any) -> None:
    """
    Validate session state
    
    Args:
        state: Session state object
        
    Raises:
        ValidationException: If validation fails
    """
    if not state:
        raise ValidationException("État de session non initialisé")
        
    if not hasattr(state, 'original_audio_path'):
        raise ValidationException("État de session invalide")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove special characters
    filename = re.sub(r'[^\w\s.-]', '_', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
        
    return name + ext 