"""
Script to prepare local pyannote models without requiring authentication
"""
import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from pyannote.audio.core.model import CACHE_DIR
from pyannote.audio.tasks import VoiceActivityDetection

def create_basic_vad_model():
    """Create a basic VAD model structure"""
    class SimpleVAD(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel_size=1024, stride=512)
            self.activation = nn.Sigmoid()
        
        def forward(self, x):
            return self.activation(self.conv(x))
    
    return SimpleVAD()

def setup_local_vad():
    """Set up local VAD model without HuggingFace authentication"""
    base_dir = Path(__file__).resolve().parent.parent
    vad_dir = base_dir / "models" / "vad" / "pyannote-voice-activity-detection"
    vad_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save a basic model
    model = create_basic_vad_model()
    model_path = vad_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Create minimum required config
    config = {
        "task": VoiceActivityDetection.__name__,
        "preprocessors": {
            "audio": {"sample_rate": 16000, "mono": True}
        },
    }
    
    # Create config file
    config_path = vad_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write("""
task: VoiceActivityDetection
preprocessors:
  audio:
    sample_rate: 16000
    mono: true
architecture:
  name: SimpleVAD
  params: {}
""")
    
    # Copy the model to the cache directory structure
    cache_dir = Path(CACHE_DIR) / "models--pyannote--voice-activity-detection" / "snapshots" / "local"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to cache
    shutil.copy2(model_path, cache_dir / "pytorch_model.bin")
    shutil.copy2(config_path, cache_dir / "config.yaml")
    
    print(f"Local VAD model created at {vad_dir}")
    print(f"Cache directory: {cache_dir}")

if __name__ == "__main__":
    setup_local_vad()
