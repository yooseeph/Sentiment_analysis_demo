"""
Simple VAD pipeline implementation
"""
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml

class SimpleVADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(1024, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return self.sigmoid(x)

class SimpleVADPipeline:
    def __init__(self, model_dir: Union[str, Path], device: torch.device = None):
        self.device = device or torch.device('cpu')
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / 'config.yaml') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = SimpleVADModel()
        self.model.load_state_dict(torch.load(model_dir / 'model.pt'))
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, List[Dict[str, float]]]:
        """Run VAD on audio waveform
        
        Args:
            waveform: (channel, samples) audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            Dict with segments containing speech
        """
        # Ensure waveform shape and type
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
            
        # Resample if needed
        if sample_rate != self.config['inference']['params']['target_sample_rate']:
            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config['inference']['params']['target_sample_rate']
            ).to(self.device)
            waveform = transform(waveform)
        
        # Split into chunks and run inference
        chunk_size = int(self.config['inference']['params']['chunk_size_sec'] * 
                        self.config['inference']['params']['target_sample_rate'])
        
        segments = []
        speech_start = None
        
        for i in range(0, waveform.size(1), chunk_size):
            chunk = waveform[:, i:i+chunk_size]
            if chunk.size(1) < chunk_size:
                # Pad last chunk
                pad_size = chunk_size - chunk.size(1) 
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            
            # Get speech probability
            with torch.no_grad():
                probs = self.model(chunk.unsqueeze(0))
                is_speech = probs[0, 1] > self.config['inference']['params']['threshold']
            
            chunk_start = i / self.config['inference']['params']['target_sample_rate']
            chunk_end = (i + chunk_size) / self.config['inference']['params']['target_sample_rate']
            
            if is_speech and speech_start is None:
                speech_start = chunk_start
            elif not is_speech and speech_start is not None:
                segments.append({
                    'start': speech_start,
                    'end': chunk_end
                })
                speech_start = None
                
        # Add final segment if needed
        if speech_start is not None:
            segments.append({
                'start': speech_start,
                'end': waveform.size(1) / self.config['inference']['params']['target_sample_rate']
            })
            
        return {'segments': segments}

    def to(self, device: torch.device) -> 'SimpleVADPipeline':
        """Move pipeline to specified device"""
        self.device = device
        self.model.to(device)
        return self
