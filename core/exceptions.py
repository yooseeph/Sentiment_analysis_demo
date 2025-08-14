"""
Custom exceptions for Sentiment Analysis Dashboard
"""
from typing import Optional, Any


class DashboardException(Exception):
    """Base exception for all dashboard-related errors"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}


class AudioException(DashboardException):
    """Audio processing related exceptions"""
    pass


class ValidationException(DashboardException):
    """Input validation exceptions"""
    pass


class ModelException(DashboardException):
    """Model loading or inference exceptions"""
    pass


class TranscriptionException(AudioException):
    """Transcription specific exceptions"""
    pass


class SentimentException(ModelException):
    """Sentiment analysis specific exceptions"""
    pass


class ConfigurationException(DashboardException):
    """Configuration related exceptions"""
    pass


class SessionException(DashboardException):
    """Session management exceptions"""
    pass


class AWSException(DashboardException):
    """AWS service related exceptions"""
    pass


def handle_gradio_error(func):
    """Decorator to handle errors in Gradio callbacks"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DashboardException as e:
            # Return user-friendly error message
            return f"❌ {str(e)}"
        except Exception as e:
            # Log unexpected errors and return generic message
            import logging
            logging.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return "❌ Une erreur inattendue s'est produite. Veuillez réessayer."
    return wrapper 