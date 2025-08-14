"""
State management for Sentiment Analysis Dashboard
Handles session-specific state and caching
"""
import os
import threading
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from pathlib import Path
import gradio as gr

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for an audio chunk"""
    id: str
    start_time: float
    end_time: float
    agent_text: str
    client_text: str
    sentiments: tuple


@dataclass
class ChunkData:
    """Data for a cached chunk"""
    audio_path: str
    waveform: Any  # torch.Tensor
    sample_rate: int
    agent_text: str
    client_text: str
    sentiments: tuple


class GlobalState:
    """Session-specific state management"""
    
    def __init__(self):
        self.original_audio_path: Optional[str] = None
        self.chunks_metadata: List[ChunkMetadata] = []
        self.chunks_in_memory: Dict[str, ChunkData] = {}
        self.current_chunk_sentiments: Optional[tuple] = None
        self.full_call_analysis: Optional[Dict[str, Any]] = None
        self.last_analyzed_file: Optional[str] = None
        self.full_transcription_cache: Dict[str, Dict[str, Any]] = {}
        self.temp_files: Set[str] = set()
        
        logger.debug("Initialized new GlobalState instance")
        
    def clear_all(self) -> None:
        """Clear all cached data"""
        logger.info("Clearing all cached data for session")
        self.clear_chunk_cache()
        # Remove uploaded/original audio file if it lives in the Gradio temp dir
        try:
            if self.original_audio_path and os.path.exists(self.original_audio_path):
                from config import config as _app_config  # local import to avoid cycles
                temp_dir = str(_app_config.gradio.temp_dir)
                try:
                    is_under_temp = os.path.commonpath([
                        os.path.abspath(self.original_audio_path),
                        os.path.abspath(temp_dir),
                    ]) == os.path.abspath(temp_dir)
                except Exception:
                    is_under_temp = False
                if is_under_temp:
                    os.remove(self.original_audio_path)
                    logger.debug(f"Removed original uploaded audio: {self.original_audio_path}")
        except Exception as e:
            logger.warning(f"Failed to remove original audio file: {e}")
        # Remove any other session-scoped temp files
        for path in list(self.temp_files):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed session temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove session temp file: {e}")
        self.temp_files.clear()
        self.chunks_metadata.clear()
        self.original_audio_path = None
        self.current_chunk_sentiments = None
        self.full_call_analysis = None
        self.last_analyzed_file = None
        self.full_transcription_cache.clear()
        
    def clear_chunk_cache(self) -> None:
        """Clear chunk cache and temporary files"""
        logger.debug(f"Clearing {len(self.chunks_in_memory)} cached chunks")
        
        for chunk_id, chunk_data in self.chunks_in_memory.items():
            if hasattr(chunk_data, 'audio_path') and os.path.exists(chunk_data.audio_path):
                try:
                    os.remove(chunk_data.audio_path)
                    logger.debug(f"Removed temporary file: {chunk_data.audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
                    
        self.chunks_in_memory.clear()
        
    def is_same_file(self, audio_path: str) -> bool:
        """Check if this is the same file as previously analyzed"""
        return self.last_analyzed_file == audio_path
        
    def set_current_file(self, audio_path: str) -> None:
        """Set current file and clear cache if different"""
        if not self.is_same_file(audio_path):
            logger.info(f"New file detected: {audio_path}")
            self.clear_all()
        self.last_analyzed_file = audio_path
        self.original_audio_path = audio_path
        
    def cache_chunk(self, chunk_id: str, chunk_data: ChunkData) -> None:
        """Cache a chunk with memory management"""
        from config import config
        
        # Check if we need to clear old chunks
        if len(self.chunks_in_memory) >= config.performance.max_cached_chunks:
            logger.warning(f"Chunk cache full ({config.performance.max_cached_chunks}), clearing oldest chunks")
            # Remove oldest chunks (simple FIFO)
            to_remove = list(self.chunks_in_memory.keys())[:10]
            for old_id in to_remove:
                old_data = self.chunks_in_memory.pop(old_id, None)
                if old_data and hasattr(old_data, 'audio_path') and os.path.exists(old_data.audio_path):
                    try:
                        os.remove(old_data.audio_path)
                    except:
                        pass
                        
        self.chunks_in_memory[chunk_id] = chunk_data
        logger.debug(f"Cached chunk {chunk_id}, total cached: {len(self.chunks_in_memory)}")

    def register_temp_file(self, file_path: str) -> None:
        """Register a temp file path owned by this session for later cleanup."""
        if file_path:
            self.temp_files.add(file_path)


class SessionManager:
    """Manages multiple user sessions"""
    
    def __init__(self):
        self._sessions: Dict[str, GlobalState] = {}
        self._lock = threading.Lock()
        logger.info("Initialized SessionManager")
    
    def get_session(self, session_id: str) -> GlobalState:
        """Get or create session state"""
        with self._lock:
            if session_id not in self._sessions:
                logger.info(f"Creating new session: {session_id}")
                self._sessions[session_id] = GlobalState()
            return self._sessions[session_id]
    
    def clear_session(self, session_id: str) -> None:
        """Clear a specific session"""
        with self._lock:
            if session_id in self._sessions:
                logger.info(f"Clearing session: {session_id}")
                self._sessions[session_id].clear_all()
                del self._sessions[session_id]
                
    def cleanup_old_sessions(self, max_sessions: int = 100) -> None:
        """Clean up old sessions if too many exist"""
        with self._lock:
            if len(self._sessions) > max_sessions:
                logger.warning(f"Too many sessions ({len(self._sessions)}), cleaning up oldest")
                # Simple cleanup - remove first half
                to_remove = list(self._sessions.keys())[:len(self._sessions)//2]
                for session_id in to_remove:
                    self._sessions[session_id].clear_all()
                    del self._sessions[session_id]
                logger.info(f"Cleaned up {len(to_remove)} sessions")

    def clear_all_sessions(self) -> None:
        """Clear and delete all sessions (for app shutdown)."""
        with self._lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                try:
                    self._sessions[session_id].clear_all()
                finally:
                    del self._sessions[session_id]
            logger.info("All sessions cleared")


# Global session manager instance
session_manager = SessionManager()


def get_session_state(request: Optional[gr.Request] = None) -> GlobalState:
    """Get session-specific state from Gradio request"""
    session_id = getattr(request, 'session_hash', 'default') if request else 'default'
    return session_manager.get_session(session_id)


def clear_session_state(request: Optional[gr.Request] = None) -> None:
    """Clear session state"""
    session_id = getattr(request, 'session_hash', 'default') if request else 'default'
    session_manager.clear_session(session_id) 