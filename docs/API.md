# Sentiment Analysis Dashboard API Documentation

## Overview

The Sentiment Analysis Dashboard provides a comprehensive API for analyzing customer support calls with transcription, sentiment analysis, and topic classification. The system supports multi-user sessions, real-time processing, and advanced caching mechanisms.

## Core Modules

### Configuration (`config.settings`)

The configuration system uses dataclasses to provide type-safe configuration management.

```python
from config import config

# Access configuration values
model_path = config.model.transcription_model_path
device = config.performance.gpu_device
```

#### Configuration Sections

- `ModelConfig`: AI model paths and weights
- `AudioConfig`: Audio processing parameters
- `GradioConfig`: Web interface settings
- `AWSConfig`: AWS service credentials
- `DataConfig`: Data file locations
- `LoggingConfig`: Logging settings
- `PerformanceConfig`: Performance optimization settings
- `SentimentConfig`: Sentiment analysis configurations

### State Management (`core.state_manager`)

Manages session-specific state and caching for multi-user support.

```python
from core.state_manager import get_session_state, GlobalState

# Get session state
state = get_session_state(request)

# Access state properties
audio_path = state.original_audio_path
chunks = state.chunks_metadata
```

#### Key Classes

- `GlobalState`: Session-specific state container
- `SessionManager`: Manages multiple user sessions
- `ChunkMetadata`: Metadata for audio chunks
- `ChunkData`: Cached chunk data

### Audio Processing (`core.audio`)

Handles audio loading, validation, transcription, and voice activity detection.

```python
from core.audio import audio_processor

# Validate audio file
is_valid, message = audio_processor.validate_audio_file(audio_path)

# Transcribe audio
transcription = audio_processor.transcribe_audio(audio_path)

# Get speech segments
segments = audio_processor.get_speech_segments(waveform, sample_rate, "Agent")
```

#### Key Methods

- `validate_audio_file(file_path: str) -> Tuple[bool, str]`
- `load_audio(file_path: str) -> Tuple[torch.Tensor, int]`
- `load_audio_channels(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]`
- `transcribe_audio(audio_path: str) -> str`
- `get_speech_segments(waveform: torch.Tensor, sample_rate: int, speaker_label: str) -> List[Dict]`

### Sentiment Analysis (`core.sentiment`)

Provides sentiment analysis for both agent and client speakers.

```python
from core.sentiment import sentiment_analyzer

# Analyze sentiment
results = sentiment_analyzer.analyze_sentiment_client_agent(
    agent_text="Agent transcription",
    client_text="Client transcription", 
    audio_path="/path/to/audio.wav"
)

# Results tuple contains:
# (agent_text_sentiment, agent_acoustic_sentiment, agent_fusion_sentiment,
#  client_text_sentiment, client_acoustic_sentiment, client_fusion_sentiment,
#  agent_per_modality, client_per_modality, agent_fusion_prob, client_fusion_prob)
```

#### Sentiment Labels 

**Agent Sentiments:**
- `aggressive`: Aggressive tone
- `courtois`: Courteous/polite
- `sec`: Dry/curt

**Client Sentiments:**
- `content`: Satisfied
- `tres mecontent`: Very dissatisfied
- `neutre`: Neutral

### Chunk Processing (`core.audio.chunk_processor`)

Handles audio chunking and parallel processing.

```python
from core.audio.chunk_processor import chunk_processor

# Process audio chunks
chunk_keys, sentiments, topic = chunk_processor.optimized_chunker(
    audio_path="/path/to/audio.wav",
    topic=True,
    request=gradio_request
)

# Extract specific chunk
chunk_data = chunk_processor.extract_chunk_from_original(
    chunk_id="chunk_0",
    request=gradio_request
)
```

### Model Management (`core.models`)

Manages loading and caching of AI models.

```python
from core.models import model_manager, load_all_models

# Load all models at startup
results = load_all_models()

# Get specific models
model, processor = model_manager.load_transcription_model()
vad_pipeline = model_manager.load_vad_pipeline()
evaluator = model_manager.load_sentiment_evaluator()
```

## API Usage Examples

### Basic Audio Analysis

```python
from core.audio import audio_processor
from core.sentiment import sentiment_analyzer

# Load and validate audio
audio_path = "/path/to/call.wav"
is_valid, message = audio_processor.validate_audio_file(audio_path)

if is_valid:
    # Load channels
    left, right, sr = audio_processor.load_audio_channels(audio_path)
    
    # Get speech segments
    agent_segments = audio_processor.get_speech_segments(left, sr, "Agent")
    client_segments = audio_processor.get_speech_segments(right, sr, "Client")
    
    # Transcribe
    agent_text = " ".join([
        audio_processor.transcribe_chunk(seg["chunk"], sr) 
        for seg in agent_segments
    ])
    client_text = " ".join([
        audio_processor.transcribe_chunk(seg["chunk"], sr)
        for seg in client_segments
    ])
    
    # Analyze sentiment
    results = sentiment_analyzer.analyze_sentiment_client_agent(
        agent_text, client_text, audio_path
    )
```

### Chunk-Based Analysis

```python
from core.audio.chunk_processor import chunk_processor
from core.state_manager import get_session_state

# Process file in chunks
audio_path = "/path/to/long_call.wav"
chunk_keys, sentiments, topic = chunk_processor.optimized_chunker(audio_path)

# Analyze specific chunk
state = get_session_state()
chunk_id = chunk_keys[0]  # First chunk

chunk_data = chunk_processor.extract_chunk_from_original(chunk_id)
print(f"Agent: {chunk_data['agent_text']}")
print(f"Client: {chunk_data['client_text']}")
print(f"Sentiments: {chunk_data['sentiments']}")
```

### Session Management

```python
from core.state_manager import session_manager, get_session_state

# Get session for user
session = get_session_state(request)

# Store analysis results
session.full_call_analysis = {
    'chunk_keys': ['chunk_0', 'chunk_1'],
    'sentiments': [(...), (...)],
    'topic_result': 'Service - Technical Support'
}

# Clear session when done
session_manager.clear_session(session_id)
```

## Performance Optimization

### Caching Strategy

The system implements multiple levels of caching:

1. **Model Caching**: Models are loaded once and reused
2. **Session Caching**: Analysis results cached per user session
3. **Chunk Caching**: Audio chunks cached to avoid re-extraction
4. **Transcription Caching**: Full transcriptions cached for reuse

### GPU Memory Management

```python
from core.models import model_manager

# Get GPU memory usage
info = model_manager.get_model_info()
print(f"GPU Memory: {info['gpu_memory_allocated']}")

# Clear GPU cache
model_manager.clear_cache()
```

### Batch Processing

```python
# Process multiple files efficiently
from utils.six_sentiments import BatchOptimizedEvaluator

evaluator = BatchOptimizedEvaluator()
batch_data = [
    {'agent_text': 'text1', 'client_text': 'text1', 'audio_path': 'file1.wav'},
    {'agent_text': 'text2', 'client_text': 'text2', 'audio_path': 'file2.wav'},
]

results = evaluator.predict_batch_optimized(batch_data)
```

## Error Handling

All API functions include comprehensive error handling and logging:

```python
try:
    result = audio_processor.transcribe_audio(audio_path)
except Exception as e:
    logger.error(f"Transcription failed: {e}")
    # Fallback handling
```

## Logging

The system uses structured logging with different levels:

```python
from utils.logging_config import get_logger

logger = get_logger(__name__)

logger.info("Processing started")
logger.debug("Detailed information")
logger.error("Error occurred", exc_info=True)
```

## WebSocket Events (Gradio)

The dashboard uses Gradio's event system for real-time updates:

```python
# Chain events for processing indication
btn.click(
    fn=lambda: gr.update(visible=True),
    outputs=[processing_indicator]
).then(
    fn=process_audio,
    inputs=[audio_input],
    outputs=[results]
).then(
    fn=lambda: gr.update(visible=False),
    outputs=[processing_indicator]
)
``` 