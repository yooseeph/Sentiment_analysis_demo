"""
Logging configuration for Sentiment Analysis Dashboard
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
from datetime import datetime

# Import config
from config import config

# Create a per-run logfile with timestamp
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_LOG_DIR = Path(config.logging.log_file).parent if config.logging.log_file else Path("logs")
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / f"dashboard_{RUN_ID}.log"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for file handlers
        record.levelname = levelname
        
        return result


def setup_logger(
    name: str = "sentiment_dashboard",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console: Enable console logging
        file_logging: Enable file logging
    
    Returns:
        Configured logger
    """
    # Use config defaults if not provided
    if log_level is None:
        log_level = config.logging.log_level
    # Use timestamped file per run by default
    if log_file is None:
        log_file = str(DEFAULT_LOG_FILE)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_logging and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use timed rotating handler
        file_handler = TimedRotatingFileHandler(
            log_file,
            when=config.logging.log_rotation,
            interval=1,
            backupCount=config.logging.log_retention_days,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Standard formatter for file (no colors)
        file_formatter = logging.Formatter(
            config.logging.log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create main logger
logger = setup_logger()

# Expose RUN_ID for other modules
LOG_RUN_ID = RUN_ID


class LoggingContext:
    """Context manager for temporary logging configuration changes"""
    
    def __init__(self, logger_name: str = "sentiment_dashboard", level: str = "INFO"):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper())
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_function_call(func):
    """Decorator to log function calls with arguments and results"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        func_name = func.__name__
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        
        try:
            # Call function
            start_time = datetime.now()
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log success
            logger.debug(f"{func_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            # Log error
            logger.error(f"{func_name} failed with error: {str(e)}", exc_info=True)
            raise
    
    return wrapper


def log_performance(func):
    """Decorator to log function performance metrics"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Import here to avoid circular imports
        import torch
        import psutil
        import os
        
        # Get initial metrics
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            initial_gpu_memory = 0
        
        # Call function
        start_time = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Get final metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_used = final_gpu_memory - initial_gpu_memory
        else:
            gpu_memory_used = 0
        
        # Log metrics
        logger.info(
            f"{func.__name__} performance: "
            f"duration={duration:.2f}s, "
            f"cpu_memory_delta={memory_used:.1f}MB, "
            f"gpu_memory_delta={gpu_memory_used:.1f}MB"
        )
        
        return result
    
    return wrapper


# Convenience functions for common logging patterns
def log_audio_processing(audio_path: str, duration: float, sample_rate: int):
    """Log audio file processing details"""
    logger.info(
        f"Processing audio: {Path(audio_path).name}, "
        f"duration={duration:.1f}s, sample_rate={sample_rate}Hz"
    )


def log_model_loading(model_name: str, model_path: str, success: bool = True):
    """Log model loading status"""
    if success:
        logger.info(f"✓ Loaded {model_name} from {model_path}")
    else:
        logger.error(f"✗ Failed to load {model_name} from {model_path}")


def log_sentiment_result(role: str, text: str, acoustic: str, fusion: str):
    """Log sentiment analysis results"""
    logger.info(
        f"Sentiment {role}: text={text}, acoustic={acoustic}, fusion={fusion}"
    )


def log_error_with_context(error: Exception, context: dict):
    """Log error with additional context information"""
    logger.error(
        f"Error: {str(error)}\n"
        f"Context: {context}",
        exc_info=True
    ) 