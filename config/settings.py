"""
Configuration management for Sentiment Analysis Dashboard
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"
LOGOS_DIR = BASE_DIR / "logos"
TEMP_DIR = BASE_DIR / "GradioTEMP"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, UTILS_DIR, LOGOS_DIR, TEMP_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # Transcription model
    transcription_model_path: str = field(
        default_factory=lambda: os.getenv(
            "TRANSCRIPTION_MODEL_PATH",
            str(MODELS_DIR / "transcription" / "w2v-bert-darija-finetuned-clean")
        )
    )
    
    # VAD model
    vad_model_id: str = "pyannote/voice-activity-detection"
    
    # Sentiment models - Agent
    agent_text_model_path: str = field(
        default_factory=lambda: str(MODELS_DIR / "agent" / "text" / "best_model")
    )
    agent_acoustic_model_path: str = field(
        default_factory=lambda: str(MODELS_DIR / "agent" / "acoustic" / "randomforest_acoustic_model.joblib")
    )
    agent_acoustic_scaler_path: str = field(
        default_factory=lambda: str(MODELS_DIR / "agent" / "acoustic" / "acoustic_scaler.joblib")
    )
    
    # Sentiment models - Client
    client_text_model_path: str = field(
        default_factory=lambda: str(MODELS_DIR / "client" / "text" / "best_model")
    )
    client_acoustic_model_path: str = field(
        default_factory=lambda: str(MODELS_DIR / "client" / "acoustic" / "svm_acoustic_model.joblib")
    )
    client_acoustic_scaler_path: str = field(
        default_factory=lambda: str(MODELS_DIR / "client" / "acoustic" / "acoustic_scaler.joblib")
    )
    
    # Model weights for fusion
    agent_weights: Dict[str, float] = field(
        default_factory=lambda: {"text": 0.54, "acoustic": 0.46}
    )
    client_weights: Dict[str, float] = field(
        default_factory=lambda: {"text": 0.42, "acoustic": 0.58}
    )


@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    original_sample_rate: int = 8000
    target_sample_rate: int = 16000
    
    # Chunking parameters
    chunk_duration_sec: float = 25.0
    chunk_overlap_sec: float = 1.0
    
    # VAD parameters
    vad_gap_threshold: float = 0.8
    vad_padding: float = 0.5
    
    # Batch processing
    batch_size: int = 8
    max_audio_length_sec: int = 30 * 60  # 30 minutes
    
    # Supported formats
    supported_formats: list = field(
        default_factory=lambda: [".wav", ".ogg"]
    )


@dataclass
class GradioConfig:
    """Configuration for Gradio interface"""
    # Server settings
    share: bool = field(
        default_factory=lambda: os.getenv("GRADIO_SHARE", "true").lower() == "true"
    )
    server_name: str = field(
        default_factory=lambda: os.getenv("SERVER_NAME", "0.0.0.0")
    )
    server_port: int = field(
        default_factory=lambda: int(os.getenv("SERVER_PORT", "7861"))
    )
    
    # Authentication
    auth: Optional[tuple] = field(default=None)
    
    def __post_init__(self):
        auth_env = os.getenv("DASHBOARD_AUTH", "").strip()
        if ":" in auth_env:
            self.auth = tuple(auth_env.split(":", 1))
    
    # UI settings
    max_threads: int = 10
    show_error: bool = True
    quiet: bool = False
    show_api: bool = False
    
    # Temp directories
    temp_dir: str = field(default_factory=lambda: str(TEMP_DIR / ".gradio_temp"))
    cache_dir: str = field(default_factory=lambda: str(TEMP_DIR / ".gradio"))


@dataclass
class AWSConfig:
    """Configuration for AWS services"""
    # AWS credentials (from environment)
    access_key_id: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID")
    )
    secret_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    session_token: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_SESSION_TOKEN")
    )
    region: str = field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    )
    
    # Bedrock settings
    bedrock_region: str = "us-west-2"
    summary_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    class_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    max_tokens_summary: int = 300
    max_tokens_class: int = 3
    temperature: float = 0.0
    
    @property
    def is_configured(self) -> bool:
        """Check if AWS is properly configured"""
        return bool(self.access_key_id and self.secret_access_key)


@dataclass
class DataConfig:
    """Configuration for data files"""
    darija_french_dict: str = field(
        default_factory=lambda: str(UTILS_DIR / "darija_french_conversion.xlsx")
    )
    topics_glossary: str = field(
        default_factory=lambda: str(UTILS_DIR / "glossaire B2C.xlsx")
    )
    
    # Logo paths
    inwi_logo: str = field(default_factory=lambda: str(LOGOS_DIR / "logo_inwi.png"))
    clever_logo: str = field(default_factory=lambda: str(LOGOS_DIR / "Cleverlytics-orange.png"))


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_file: str = field(
        default_factory=lambda: str(LOGS_DIR / "dashboard.log")
    )
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "midnight"
    log_retention_days: int = 30


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    # GPU settings
    use_gpu: bool = field(
        default_factory=lambda: torch.cuda.is_available()
    )
    gpu_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory management
    clear_cache_after_chunks: int = 10
    max_cached_chunks: int = 50
    
    # Processing
    enable_parallel_processing: bool = True
    max_workers: int = field(
        default_factory=lambda: min(8, os.cpu_count() or 4)
    )


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    # Sentiment labels
    agent_sentiments: list = field(
        default_factory=lambda: ["aggressive", "courtois", "neutre", "sec"]
    )
    client_sentiments: list = field(
        default_factory=lambda: ["content", "neutre", "mecontent", "tres mecontent"]
    )
    
    # Display mapping
    sentiment_display: Dict[str, str] = field(
        default_factory=lambda: {
            "content": "Content",
            "mecontent": "Mécontent",
            "tres mecontent": "Très Mécontent",
            "neutre": "Neutre",
            "aggressive": "Agressif",
            "sec": "Sec",
            "courtois": "Courtois",
        }
    )


class Config:
    """Main configuration class that combines all config sections"""
    def __init__(self):
        self.model = ModelConfig()
        self.audio = AudioConfig()
        self.gradio = GradioConfig()
        self.aws = AWSConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        self.sentiment = SentimentConfig()
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration and create necessary directories"""
        # Create Gradio temp directories
        os.makedirs(self.gradio.temp_dir, exist_ok=True)
        os.makedirs(self.gradio.cache_dir, exist_ok=True)
        
        # Check if model paths exist
        model_paths = [
            self.model.transcription_model_path,
            self.model.agent_text_model_path,
            self.model.client_text_model_path,
        ]
        
        missing_models = [p for p in model_paths if not Path(p).exists()]
        if missing_models:
            print(f"Warning: Some model paths do not exist: {missing_models}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "audio": self.audio.__dict__,
            "gradio": self.gradio.__dict__,
            "aws": {k: v for k, v in self.aws.__dict__.items() 
                   if k not in ["access_key_id", "secret_access_key", "session_token"]},
            "data": self.data.__dict__,
            "logging": self.logging.__dict__,
            "performance": self.performance.__dict__,
            "sentiment": self.sentiment.__dict__,
        }


# Global config instance
config = Config()

# Import torch after config is created
import torch 