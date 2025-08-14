"""
Temporary files cleanup utilities.

Removes files in Gradio temp/cache directories and any session-specific
temporary artifacts.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Removed directory: {path}")
    except Exception as exc:
        logger.warning(f"Failed to remove directory {path}: {exc}")


def _delete_children(path: Path) -> None:
    try:
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                try:
                    child.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception as exc:
                    logger.warning(f"Failed to remove file {child}: {exc}")
            else:
                _safe_rmtree(child)
    except Exception as exc:
        logger.warning(f"Error while cleaning children of {path}: {exc}")


def cleanup_temp_dirs_for_session(request: Optional[object] = None) -> None:
    """Delete only temp files registered for the given session."""
    try:
        from core.state_manager import get_session_state
        state = get_session_state(request)
        # clear_all() already removes session temp files; call it safely
        state.clear_all()
        # Also remove the session-scoped CSV file if present
        try:
            from core.utils import get_session_csv_path
            csv_path = Path(get_session_csv_path(request))
            if csv_path.exists():
                csv_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                logger.info(f"Removed session CSV: {csv_path}")
        except Exception:
            pass
        logger.info("Session-scoped temp files cleaned.")
    except Exception as exc:
        logger.warning(f"Failed to cleanup session temp files: {exc}")


def cleanup_temp_dirs(remove_cache: bool = False) -> None:
    """Delete runtime temporary files/directories (global sweep).

    - Cleans Gradio temp dir contents (config.gradio.temp_dir)
    - Optionally cleans Gradio cache dir (config.gradio.cache_dir)
    Use with caution; prefer cleanup_temp_dirs_for_session when possible.
    """
    try:
        from config import config
    except Exception as exc:
        logger.warning(f"Could not import config for cleanup: {exc}")
        return

    temp_dir = Path(config.gradio.temp_dir)
    cache_dir = Path(config.gradio.cache_dir)

    logger.info("Cleaning Gradio temp directory contents...")
    _delete_children(temp_dir)

    if remove_cache:
        logger.info("Cleaning Gradio cache directory contents...")
        _delete_children(cache_dir)


def register_process_cleanup_handlers() -> None:
    """Register atexit and signal handlers to cleanup on shutdown.

    This ensures temp files are removed if the app is stopped via Ctrl+C (SIGINT)
    or SIGTERM (container stop), and also at normal interpreter exit.
    """
    import atexit
    import signal

    def _cleanup_wrapper() -> None:
        try:
            # Clear sessions to delete per-chunk temp files
            from core.state_manager import session_manager  # local import to avoid cycles
            try:
                session_manager.clear_all_sessions()
            except Exception:
                pass
        finally:
            # Remove temp directory contents
            try:
                cleanup_temp_dirs(remove_cache=False)
            except Exception:
                pass

    # atexit always runs on normal interpreter exit
    atexit.register(_cleanup_wrapper)

    # Also handle common termination signals
    def _signal_handler(signum, frame):  # type: ignore[unused-argument]
        _cleanup_wrapper()
        # Re-raise default to terminate process promptly
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass


