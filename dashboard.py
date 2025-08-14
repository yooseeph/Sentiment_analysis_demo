#!/usr/bin/env python
# coding: utf-8
"""
Professional Sentiment Analysis Dashboard for Audio Call Analysis
=================================================================

This dashboard provides comprehensive analysis of customer support calls with:
- Transcription and sentiment analysis (text + acoustic fusion)
- Topic classification
- Multi-user session support
- Optimized caching and performance

Compatible with WAV and OGG audio formats.
"""

import os
import sys
from typing import Optional, List, Dict, Any, Tuple
import gradio as gr

# Set up environment before imports
from config import config
from utils.temp_cleanup import register_process_cleanup_handlers, cleanup_temp_dirs
os.environ["GRADIO_TEMP_DIR"] = config.gradio.temp_dir
os.environ["TMPDIR"] = config.gradio.temp_dir
os.environ["GRADIO_CACHE_DIR"] = config.gradio.cache_dir

# Ensure cleanup handlers are registered as early as possible
register_process_cleanup_handlers()

# Clean temp dir on startup to recover from previous unclean exits
try:
    cleanup_temp_dirs(remove_cache=False)
except Exception:
    pass

# Import core modules
from utils.logging_config import logger, log_performance
from core.models import load_all_models
from core.state_manager import get_session_state, clear_session_state
from core.audio import audio_processor
from core.audio.chunk_processor import chunk_processor
from core.sentiment import sentiment_analyzer
from core.utils import (
    get_base64_image,
    get_custom_css,
    darija_converter,
    append_rows_to_csv
)

# Import UI views
from core.ui.views import (
    create_vue1_interface,
    create_vue2_interface,
    create_vue3_interface,
    optimized_inference
)

# Global UI component references (for cross-view communication)
UI_COMPONENTS = {
    'audio_input_vue2': None,
    'output_html_vue2': None,
    'processing_html_vue2': None,
    'chunk_selector': None,
    'sections': {
        'vue1': None,
        'vue2': None,
        'vue3': None
    }
}


def initialize_application():
    """Initialize the application and load all required models"""
    logger.info("=" * 60)
    logger.info("Initializing Sentiment Analysis Dashboard")
    logger.info("=" * 60)
    
    # Load all models
    model_results = load_all_models()
    
    # Check if critical models loaded successfully
    if not model_results.get('transcription'):
        logger.critical("Failed to load transcription model - aborting")
        sys.exit(1)
        
    if not model_results.get('sentiment'):
        logger.warning("Sentiment evaluator not loaded - sentiment analysis will be limited")
        
    # Load logos
    logo_inwi = get_base64_image(str(config.data.inwi_logo))
    logo_clever = get_base64_image(str(config.data.clever_logo))
    
    logger.info("Application initialized successfully")
    
    return logo_inwi, logo_clever


def create_main_interface():
    """Create the main dashboard interface"""
    global UI_COMPONENTS
    
    # Initialize application
    logo_inwi, logo_clever = initialize_application()
    
    with gr.Blocks(
        theme=gr.themes.Default(),
        title="Sentiment Analysis Dashboard",
        css=get_custom_css()
    ) as interface:
        
        # Register shutdown cleanup
        try:
            from core.state_manager import session_manager
            from utils.temp_cleanup import cleanup_temp_dirs
            def _shutdown_cleanup():
                try:
                    session_manager.clear_all_sessions()
                finally:
                    # Do not remove cache by default; only temp files
                    cleanup_temp_dirs(remove_cache=False)
            interface.on_shutdown(_shutdown_cleanup)
        except Exception:
            pass
        
        # State to carry precomputed sentiments between events
        precomputed_state = gr.State(value=None)
        
        # Header with logos and title
        gr.HTML(f"""
            <div class="header-container" style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0;">
                <img src="data:image/png;base64,{logo_clever}" width="150" alt="Logo Cleverlytics">
                <h2 class="header-title" style="margin: 0; color: var(--body-text-color, #000) !important; font-weight: 600;">Sentiment Analysis Dashboard</h2>
                <img src="data:image/png;base64,{logo_inwi}" width="120" alt="Logo Inwi">
            </div>
        """)
        
        # Navigation buttons
        with gr.Row():
            btn_vue1 = gr.Button(
                "Vue 1: Analyse Appel",
                scale=1,
                elem_classes=["primary-button"]
            )
            btn_vue2 = gr.Button(
                "Vue 2: Analyse Chunk",
                scale=1,
                elem_classes=["primary-button"]
            )
        
        # Create views (order matters for cross-references)
        with gr.Column(visible=False) as section_vue2:
            UI_COMPONENTS['sections']['vue2'] = section_vue2
            vue2_components = create_vue2_interface()
            UI_COMPONENTS.update(vue2_components)
        
        with gr.Column(visible=True) as section_vue1:
            UI_COMPONENTS['sections']['vue1'] = section_vue1
            vue1_components = create_vue1_interface(UI_COMPONENTS)
            UI_COMPONENTS.update(vue1_components)
        
        # Remove Vue3 entirely
        # Navigation logic
        def show_vue1():
            return (
                gr.update(visible=True),
                gr.update(visible=False)
            )
            
        def show_vue2():
            return (
                gr.update(visible=False),
                gr.update(visible=True)
            )
        
        btn_vue1.click(
            fn=show_vue1,
            inputs=[],
            outputs=[section_vue1, section_vue2]
        )
        
        btn_vue2.click(
            fn=show_vue2,
            inputs=[],
            outputs=[section_vue1, section_vue2]
        )

        # After all components are created, wire chunk selection
        if UI_COMPONENTS.get('chunk_selector') and \
           UI_COMPONENTS.get('audio_input_vue2') and \
           UI_COMPONENTS['sections'].get('vue1') is not None and \
           UI_COMPONENTS['sections'].get('vue2') is not None:

            def on_chunk_selected(chunk_line, request: gr.Request = None):
                from core.state_manager import get_session_state
                from core.audio.chunk_processor import chunk_processor
                state = get_session_state(request)
                if not chunk_line:
                    state.current_chunk_sentiments = None
                    return gr.update(), gr.update(), gr.update(), None
                chunk_id = chunk_line.split("  |")[0].strip()
                try:
                    chunk_data = chunk_processor.extract_chunk_from_original(chunk_id, request)
                    chunk_path = chunk_data['audio_path']
                    state.current_chunk_sentiments = chunk_data['sentiments']
                    return (
                        gr.update(value=chunk_path),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        state.current_chunk_sentiments,
                    )
                except Exception:
                    state.current_chunk_sentiments = None
                    return gr.update(), gr.update(), gr.update(), None

            UI_COMPONENTS['chunk_selector'].change(
                fn=on_chunk_selected,
                inputs=[UI_COMPONENTS['chunk_selector']],
                outputs=[
                    UI_COMPONENTS['audio_input_vue2'],
                    UI_COMPONENTS['sections']['vue2'],
                    UI_COMPONENTS['sections']['vue1'],
                    precomputed_state,
                ],
                show_progress=False,
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=[],
                outputs=[UI_COMPONENTS['processing_html_vue2']],
                show_progress=False,
            ).then(
                fn=lambda audio_path, pre_sents, request=None: optimized_inference(
                    audio_path,
                    pre_sents,
                    request,
                ),
                inputs=[UI_COMPONENTS['audio_input_vue2'], precomputed_state],
                outputs=[UI_COMPONENTS['output_html_vue2']],
                show_progress=False,
            ).then(
                fn=lambda: gr.update(visible=False),
                inputs=[],
                outputs=[UI_COMPONENTS['processing_html_vue2']],
                show_progress=False,
            )
    
    return interface


def main():
    """Main entry point"""
    try:
        logger.info("Starting Sentiment Analysis Dashboard...")
        
        # Create interface
        interface = create_main_interface()
        
        logger.info("Dashboard created successfully")
        logger.info(f"Configuration: {config.to_dict()}")
        
        # Launch interface
        interface.launch(
            share=config.gradio.share,
            server_name=config.gradio.server_name,
            server_port=config.gradio.server_port,
            show_error=config.gradio.show_error,
            quiet=config.gradio.quiet,
            max_threads=config.gradio.max_threads,
            auth=config.gradio.auth,
            show_api=config.gradio.show_api
        )
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested by user")
    except Exception as e:
        logger.critical(f"Fatal error launching dashboard: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 