"""
Script to setup a proper local VAD pipeline
"""
import os
import torch
import yaml
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.core.model import CACHE_DIR
from pyannote.audio.tasks import VoiceActivityDetection
from huggingface_hub import snapshot_download, hf_hub_download
from dotenv import load_dotenv

def setup_vad_model():
    """Set up local VAD model with proper configuration"""
    base_dir = Path(__file__).resolve().parent.parent
    vad_dir = base_dir / "models" / "vad" / "pyannote-voice-activity-detection"
    vad_dir.mkdir(parents=True, exist_ok=True)
    
    # Load environment variables to get HF token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN required for initial model download")

    print("Downloading VAD model files...")
    
    try:
        # Download the main model files
        snapshot_download(
            repo_id="pyannote/voice-activity-detection",
            local_dir=vad_dir,
            token=hf_token,
            local_dir_use_symlinks=False
        )
        
        # Also download the segmentation model which is required
        seg_dir = base_dir / "models" / "vad" / "pyannote-segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id="pyannote/segmentation",
            local_dir=seg_dir,
            token=hf_token,
            local_dir_use_symlinks=False
        )
        
        # Create the cache directory structure
        vad_cache_dir = Path(CACHE_DIR) / "models--pyannote--voice-activity-detection" / "snapshots" / "local"
        seg_cache_dir = Path(CACHE_DIR) / "models--pyannote--segmentation" / "snapshots" / "local"
        
        vad_cache_dir.mkdir(parents=True, exist_ok=True)
        seg_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to cache
        for src_dir, cache_dir in [(vad_dir, vad_cache_dir), (seg_dir, seg_cache_dir)]:
            for file in src_dir.glob("*"):
                if file.is_file():
                    with open(cache_dir / file.name, "wb") as f_out:
                        f_out.write(file.read_bytes())
        
        print("Successfully downloaded and setup VAD model files")
        
        # Test the model
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=hf_token
        )
        
        # Save the pipeline
        pipeline.save_pretrained(str(vad_dir))
        print("Successfully loaded and saved the VAD pipeline")
        
    except Exception as e:
        print(f"Error setting up VAD model: {str(e)}")
        raise

if __name__ == "__main__":
    setup_vad_model()
