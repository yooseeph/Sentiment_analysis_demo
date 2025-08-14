"""
Model loading and management for Sentiment Analysis Dashboard
"""
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
from pyannote.audio import Pipeline

from config import config
from utils.logging_config import get_logger, log_model_loading

logger = get_logger(__name__)


class ModelManager:
    """Manages loading and caching of AI models"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}
        self.device = torch.device(config.performance.gpu_device)
        logger.info(f"ModelManager initialized with device: {self.device}")
        
    def load_transcription_model(self) -> Tuple[Any, Any]:
        """Load transcription model and processor"""
        model_key = "transcription"
        processor_key = "transcription_processor"
        
        # Check cache
        if model_key in self._models and processor_key in self._processors:
            logger.debug("Using cached transcription model")
            return self._models[model_key], self._processors[processor_key]
            
        try:
            logger.info(f"Loading transcription model from {config.model.transcription_model_path}")
            
            # Load processor
            processor = Wav2Vec2BertProcessor.from_pretrained(
                config.model.transcription_model_path
            )
            
            # Load model
            model = Wav2Vec2BertForCTC.from_pretrained(
                config.model.transcription_model_path,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            ).to(self.device).eval()
            
            # Cache
            self._models[model_key] = model
            self._processors[processor_key] = processor
            
            log_model_loading("Transcription Model", config.model.transcription_model_path, True)
            return model, processor
            
        except Exception as e:
            log_model_loading("Transcription Model", config.model.transcription_model_path, False)
            logger.error(f"Failed to load transcription model: {e}")
            raise
            
    def load_vad_pipeline(self) -> Any:
        """Load Voice Activity Detection pipeline"""
        vad_key = "vad"
        
        # Check cache
        if vad_key in self._models:
            logger.debug("Using cached VAD pipeline")
            return self._models[vad_key]
            
        try:
            logger.info(f"Loading VAD pipeline: {config.model.vad_model_id}")
            
            pipeline = Pipeline.from_pretrained(config.model.vad_model_id)
            
            # Cache
            self._models[vad_key] = pipeline
            
            log_model_loading("VAD Pipeline", config.model.vad_model_id, True)
            return pipeline
            
        except Exception as e:
            log_model_loading("VAD Pipeline", config.model.vad_model_id, False)
            logger.error(f"Failed to load VAD pipeline: {e}")
            raise
            
    def load_sentiment_evaluator(self) -> Any:
        """Load sentiment analysis evaluator"""
        evaluator_key = "sentiment_evaluator"
        
        # Check cache
        if evaluator_key in self._models:
            logger.debug("Using cached sentiment evaluator")
            return self._models[evaluator_key]
            
        try:
            # Import here to avoid circular dependencies
            from utils.six_sentiments import OptimizedDualMultimodalEvaluator
            
            logger.info("Loading sentiment evaluator")
            evaluator = OptimizedDualMultimodalEvaluator()
            
            # Cache
            self._models[evaluator_key] = evaluator
            
            log_model_loading("Sentiment Evaluator", "OptimizedDualMultimodalEvaluator", True)
            return evaluator
            
        except Exception as e:
            log_model_loading("Sentiment Evaluator", "OptimizedDualMultimodalEvaluator", False)
            logger.error(f"Failed to load sentiment evaluator: {e}")
            return None
            
    def clear_cache(self) -> None:
        """Clear model cache to free memory"""
        logger.info("Clearing model cache")
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Note: We don't actually delete models as they're expensive to reload
        # This is just for GPU memory cleanup
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "device": str(self.device),
            "loaded_models": list(self._models.keys()),
            "loaded_processors": list(self._processors.keys()),
        }
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            
        return info


# Global model manager instance
model_manager = ModelManager()


def load_all_models() -> Dict[str, Any]:
    """Load all required models at startup"""
    logger.info("Loading all models...")
    
    results = {
        "transcription": False,
        "vad": False,
        "sentiment": False
    }
    
    # Load transcription model
    try:
        model_manager.load_transcription_model()
        results["transcription"] = True
    except Exception as e:
        logger.error(f"Failed to load transcription model: {e}")
        
    # Load VAD pipeline
    try:
        model_manager.load_vad_pipeline()
        results["vad"] = True
    except Exception as e:
        logger.error(f"Failed to load VAD pipeline: {e}")
        
    # Load sentiment evaluator
    try:
        evaluator = model_manager.load_sentiment_evaluator()
        results["sentiment"] = evaluator is not None
    except Exception as e:
        logger.error(f"Failed to load sentiment evaluator: {e}")
        
    logger.info(f"Model loading results: {results}")
    return results


def get_transcription_model() -> Tuple[Any, Any]:
    """Get transcription model and processor"""
    return model_manager.load_transcription_model()


def get_vad_pipeline() -> Any:
    """Get VAD pipeline"""
    return model_manager.load_vad_pipeline()


def get_sentiment_evaluator() -> Any:
    """Get sentiment evaluator"""
    return model_manager.load_sentiment_evaluator() 