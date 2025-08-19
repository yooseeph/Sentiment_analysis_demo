import torch
import os

class SimpleVADModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.linear = torch.nn.Linear(1024, 2)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return self.sigmoid(x)

def create_vad_model():
    model = SimpleVADModel()
    model_path = 'models/vad/simple-vad/model.pt'
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

if __name__ == '__main__':
    create_vad_model()
