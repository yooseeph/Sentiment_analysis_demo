"""
Script to download pyannote models for local use
"""
import os
from pathlib import Path
import torch
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download, hf_hub_download
from dotenv import load_dotenv

def download_pyannote_models():
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")

    # Define model IDs and local paths
    models = {
        "pyannote/voice-activity-detection": "models/vad/pyannote-voice-activity-detection",
        "pyannote/segmentation": "models/vad/pyannote-segmentation",  # Required dependency
    }
    
    base_dir = Path(__file__).resolve().parent.parent
    
    for model_id, local_path in models.items():
        print(f"Downloading {model_id}...")
        local_path = base_dir / local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # First download the model files
            local_files = snapshot_download(
                repo_id=model_id,
                token=hf_token,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {model_id} files to {local_path}")
            
            # Also download the model weights specifically
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                token=hf_token,
                cache_dir=local_path / ".cache"
            )
            print(f"Successfully downloaded model weights to {weights_path}")
            
            if model_id == "pyannote/voice-activity-detection":
                # Test the VAD pipeline
                try:
                    pipeline = Pipeline.from_pretrained(
                        local_path,
                        use_auth_token=hf_token
                    )
                    print("Successfully loaded the VAD pipeline")
                except Exception as e:
                    print(f"Warning: Could not test VAD pipeline: {e}")
            
        except Exception as e:
            print(f"Error downloading/testing {model_id}: {str(e)}")
            raise

if __name__ == "__main__":
    download_pyannote_models()
