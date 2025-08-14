# Deployment Guide for Sentiment Analysis Dashboard

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **RAM**: Minimum 16GB (32GB recommended for production)
- **GPU**: NVIDIA GPU with CUDA 11.7+ (optional but recommended)
- **Storage**: 50GB free space for models and data

### Software Dependencies
- FFmpeg (for audio processing)
- CUDA Toolkit (if using GPU)
- Git

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd sentiment-analysis-dashboard
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sentiment-dashboard python=3.8
conda activate sentiment-dashboard
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (use chocolatey)
choco install ffmpeg
```

### 4. Download Models

Create a `models` directory structure and download required models:

```bash
mkdir -p models/{transcription,agent/{text,acoustic},client/{text,acoustic}}
```

Place your models in the appropriate directories:
- Transcription model → `models/transcription/w2v-bert-darija-finetuned-clean/`
- Agent text model → `models/agent/text/best_model/`
- Agent acoustic model → `models/agent/acoustic/`
- Client text model → `models/client/text/best_model/`
- Client acoustic model → `models/client/acoustic/`

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Model paths (optional if using default locations)
TRANSCRIPTION_MODEL_PATH=/path/to/transcription/model

# AWS Configuration (required for topic analysis)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-west-2

# Server Configuration
GRADIO_SHARE=false
SERVER_NAME=0.0.0.0
SERVER_PORT=7861

# Authentication (optional)
DASHBOARD_AUTH=username:password

# Logging
LOG_LEVEL=INFO
```

### 2. Advanced Configuration

Edit `config/settings.py` for more detailed configuration:

```python
# Example: Adjust batch size for your GPU
@dataclass
class AudioConfig:
    batch_size: int = 8  # Reduce if GPU memory is limited
    
# Example: Change temp directory
@dataclass
class GradioConfig:
    temp_dir: str = "/path/to/custom/temp"
```

## Running the Application

### Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run the dashboard
python dashboard.py
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs GradioTEMP

# Expose port
EXPOSE 7861

# Set environment variables
ENV GRADIO_SHARE=false
ENV SERVER_NAME=0.0.0.0
ENV SERVER_PORT=7861

# Run the application
CMD ["python", "dashboard.py"]
```

### 2. Build and Run

```bash
# Build image
docker build -t sentiment-dashboard .

# Run container
docker run -d \
  --name sentiment-dashboard \
  -p 7861:7861 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --gpus all \
  sentiment-dashboard
```

## Nginx Reverse Proxy

For production deployment behind Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:7861;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
    }
}
```

## Performance Tuning

### 1. GPU Memory Optimization

```python
# In config/settings.py
@dataclass
class PerformanceConfig:
    max_cached_chunks: int = 20  # Reduce for limited GPU memory
    clear_cache_after_chunks: int = 5
```

### 2. CPU Optimization

```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 3. Batch Size Tuning

Monitor GPU memory usage and adjust batch sizes:

```python
# In config/settings.py
@dataclass
class AudioConfig:
    batch_size: int = 4  # Start small and increase
```

## Monitoring

### 1. Application Logs

```bash
# View logs
tail -f logs/dashboard.log

# Log rotation is automatic (configured in LoggingConfig)
```

### 2. System Monitoring

```bash
# GPU monitoring
nvidia-smi -l 1

# CPU and memory
htop

# Disk usage
df -h
```

### 3. Health Check Endpoint

Add a health check route for monitoring:

```python
# In dashboard.py
@interface.app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   - Reduce batch size in configuration
   - Clear GPU cache more frequently
   - Use CPU for some operations

2. **Model Loading Errors**
   - Verify model paths in configuration
   - Check model file permissions
   - Ensure compatible model versions

3. **Audio Processing Errors**
   - Verify FFmpeg installation
   - Check audio file formats
   - Ensure sufficient disk space for temp files

4. **AWS Credentials Error**
   - Verify AWS credentials in `.env`
   - Check AWS region settings
   - Ensure IAM permissions for Bedrock

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python dashboard.py
```

## Security Considerations

1. **Authentication**: Always enable authentication in production
   ```bash
   DASHBOARD_AUTH=admin:strong_password_here
   ```

2. **HTTPS**: Use SSL/TLS certificates with Nginx
   ```bash
   certbot --nginx -d your-domain.com
   ```

3. **Firewall**: Configure firewall rules
   ```bash
   ufw allow 22/tcp
   ufw allow 443/tcp
   ufw enable
   ```

4. **Secrets Management**: Use environment variables or secrets manager for sensitive data

## Backup and Recovery

### Backup Strategy

1. **Models**: Store models in version control or S3
2. **Logs**: Regular backup of log files
3. **Configuration**: Version control for all config files

### Backup Script

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/sentiment-dashboard"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" logs/

# Backup configuration
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" config/ .env

# Clean old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```yaml
# docker-compose.yml
version: '3.8'

services:
  dashboard1:
    image: sentiment-dashboard
    ports:
      - "7861:7861"
    deploy:
      replicas: 3
      
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - dashboard1
```

### Vertical Scaling

- Increase GPU memory
- Add more CPU cores
- Increase system RAM
- Use faster storage (NVMe SSD) 