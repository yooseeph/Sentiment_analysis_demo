"""
UI views for Sentiment Analysis Dashboard
"""
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from config import config
from utils.logging_config import get_logger
from core.state_manager import get_session_state
from core.audio import audio_processor
from core.audio.chunk_processor import chunk_processor
from core.sentiment import sentiment_analyzer
from core.utils import darija_converter, get_custom_css
from utils.temp_cleanup import cleanup_temp_dirs

logger = get_logger(__name__)


# =============================================================================
# VUE 1: Call Analysis (chunk by chunk)
# =============================================================================

def create_vue1_interface(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create Vue 1: Call analysis interface"""
    
    gr.Markdown("## **Analyse d'appel chunk par chunk: sentiment et topic**")
    
    audio_input = gr.Audio(
        type="filepath",
        label="📁 Uploader un appel (WAV/OGG)"
    )
    
    chunk_selector = gr.Radio(
        label="Sélectionner un chunk à analyser",
        interactive=True,
        choices=[]
    )
    
    global_client = gr.Textbox(
        label="Sentiment global appel client",
        interactive=False
    )
    global_agent = gr.Textbox(
        label="Sentiment global appel agent",
        interactive=False
    )
    topic_output = gr.Textbox(
        label="Topic de l'appel",
        interactive=False
    )
    
    # Processing indicator
    processing_html_v1 = gr.HTML(
        "<div class='processing-indicator'>⏳ Traitement en cours...</div>",
        visible=False
    )
    
    # Analysis button
    btn = gr.Button("Analyser", elem_classes=["primary-button"])
    
    # Event handlers
    def on_audio_change(audio_path, request: gr.Request = None):
        """Clear session cache when audio is removed"""
        state = get_session_state(request)
        
        if not audio_path:
            # Prefer session-scoped cleanup; fallback to state.clear_all
            try:
                from utils.temp_cleanup import cleanup_temp_dirs_for_session
                cleanup_temp_dirs_for_session(request)
            except Exception:
                state.clear_all()
            logger.info("Audio removed: cleared session cache")
            return (
                gr.update(choices=[], value=None),  # chunk selector
                "",  # global client
                "",  # global agent
                "",  # topic
                gr.update(value=None),  # audio input vue2
                ""   # output html vue2
            )
        
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    def process_audio(audio_path, request: gr.Request = None):
        """Process audio file and extract chunks"""
        state = get_session_state(request)
        
        if not audio_path:
            return (
                gr.update(choices=[]),
                "Aucun fichier audio",
                "Aucun fichier audio",
                "Aucun fichier audio"
            )
        
        # Clear previous chunk sentiments
        state.current_chunk_sentiments = None
        logger.info("Starting fresh analysis")
        
        # Validate audio
        is_valid, validation_message = audio_processor.validate_audio_file(audio_path)
        if not is_valid:
            error_msg = f"Erreur de validation: {validation_message}"
            return gr.update(choices=[]), error_msg, error_msg, error_msg
        
        # Process chunks
        chunks, sentiments, topic_call = chunk_processor.optimized_chunker(
            audio_path, request=request
        )
        
        # Extract sentiment predictions
        sentiments_a = [s[2] if s[2] else "Inconnu" for s in sentiments]
        sentiments_c = [s[5] if s[5] else "Inconnu" for s in sentiments]
        
        # Create display names
        chunk_names = []
        for i, (chunk_id, s_c, s_a) in enumerate(zip(chunks, sentiments_c, sentiments_a)):
            chunk_meta = next(
                (c for c in state.chunks_metadata if c.id == chunk_id),
                None
            )
            display_name = f"{chunk_id}  |  Client: {s_c} | Agent: {s_a}"
            chunk_names.append(display_name)
        
        # Get overall sentiments
        global_c = sentiment_analyzer.sentiment_appel_client(sentiments_c)
        global_a = sentiment_analyzer.sentiment_appel_agent(sentiments_a)
        
        return gr.update(choices=chunk_names, value=None), global_c, global_a, topic_call
    
    def on_chunk_selected(chunk_line, request: gr.Request = None):
        """Handle chunk selection"""
        state = get_session_state(request)
        
        if not chunk_line:
            state.current_chunk_sentiments = None
            return gr.update(), gr.update(), gr.update(), gr.update()
        
        # Extract chunk ID
        chunk_id = chunk_line.split("  |")[0].strip()
        logger.info(f"Chunk selected: {chunk_id}")
        
        try:
            # Extract chunk
            chunk_data = chunk_processor.extract_chunk_from_original(chunk_id, request)
            chunk_path = chunk_data['audio_path']
            state.current_chunk_sentiments = chunk_data['sentiments']
            
            # Switch to Vue 2 and load chunk
            return (
                gr.update(value=chunk_path),  # audio_input_vue2
                gr.update(visible=True),       # section_vue2
                gr.update(visible=False),      # section_vue1
                gr.update(visible=False)       # section_vue3
            )
            
        except Exception as e:
            logger.error(f"Error extracting chunk: {e}")
            state.current_chunk_sentiments = None
            return gr.update(), gr.update(), gr.update(), gr.update()
    
    # Wire up events
    btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[processing_html_v1],
        show_progress=False,
    ).then(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[chunk_selector, global_client, global_agent, topic_output],
        show_progress=False,
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[processing_html_v1],
        show_progress=False,
    )
    
    audio_input.change(
        fn=on_audio_change,
        inputs=[audio_input],
        outputs=[
            chunk_selector, global_client, global_agent, topic_output,
            ui_components['audio_input_vue2'], ui_components['output_html_vue2']
        ]
    )
    
    # Return selector; wiring is done after all sections are created in main
    return {'chunk_selector': chunk_selector}


# =============================================================================
# VUE 2: Chunk Analysis (transcription and sentiment)
# =============================================================================

def create_vue2_interface() -> Dict[str, Any]:
    """Create Vue 2: Chunk analysis interface"""
    
    gr.Markdown("## **Analyse de chunk: transcription et sentiment**")
    
    with gr.Row():
        audio_input_vue2 = gr.Audio(
            type="filepath",
            label="Télécharger un fichier audio (WAV/OGG)",
            interactive=True,
            elem_id="audio-upload"
        )
    
    with gr.Row():
        submit_btn = gr.Button("Analyser", elem_classes=["primary-button"], scale=3)
    
    # Processing indicator
    processing_html_vue2 = gr.HTML(
        "<div class='processing-indicator'>⏳ Traitement en cours...</div>",
        visible=False
    )
    
    output_html_vue2 = gr.HTML(label="Résultat")
    
    # Event handlers
    def on_audio_change_vue2(audio_path, request: gr.Request = None):
        """Clear output when audio is removed"""
        state = get_session_state(request)
        if not audio_path:
            state.current_chunk_sentiments = None
            logger.info("Audio removed in Vue2: cleared output")
            return ""
        return gr.update()
    
    # Wire up events
    submit_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[processing_html_vue2],
        show_progress=False,
    ).then(
        fn=lambda audio_path, request=None: optimized_inference(
            audio_path,
            get_session_state(request).current_chunk_sentiments,
            request
        ),
        inputs=[audio_input_vue2],
        outputs=[output_html_vue2],
        show_progress=False,
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[processing_html_vue2],
        show_progress=False,
    )
    
    audio_input_vue2.change(
        fn=on_audio_change_vue2,
        inputs=[audio_input_vue2],
        outputs=[output_html_vue2]
    )
    
    return {
        'audio_input_vue2': audio_input_vue2,
        'output_html_vue2': output_html_vue2,
        'processing_html_vue2': processing_html_vue2
    }


# =============================================================================
# VUE 3: Multi-call Analysis (trend graphs)
# =============================================================================

def create_vue3_interface():
    """Create Vue 3: Multi-call analysis interface"""
    
    gr.Markdown("## **Répartition des tons globaux de l'agent : analyse multi-appels**")
    
    with gr.Row():
        with gr.Column():
            audio_files = gr.File(
                file_types=[".wav", ".ogg"],
                file_count="multiple",
                label="📁 Fichiers audio WAV/OGG (1 par appel)"
            )
            audio_count_text = gr.Markdown("")
            
        with gr.Column():
            date_inputs = gr.Textbox(
                label="📅 Dates des appels (1 par ligne)",
                lines=10,
                placeholder="2025-06-01\n2025-06-01\n2025-06-02\n..."
            )
    
    period = gr.Radio(
        ["day", "week", "month"],
        value="day",
        label="📊 Période de regroupement"
    )
    
    submit_btn = gr.Button("Générer le graphe", elem_classes=["primary-button"])
    
    output_text = gr.Textbox(label="Résultat")
    output_plot = gr.Plot()
    
    # Event handlers
    def update_audio_count(file_list):
        """Update audio file count display"""
        if not file_list:
            return "📁 Aucun fichier sélectionné"
        
        n = len(file_list)
        return f"🎧 **{n} fichiers audio** sélectionnés.<br>📅 Merci d'entrer **{n} dates**, une par ligne."
    
    def get_agent_audio_tone(audio_path, request: gr.Request = None):
        """Get agent tone for an audio file"""
        state = get_session_state(request)
        
        # Check cache
        if state.is_same_file(audio_path) and state.full_call_analysis:
            logger.info("Using cached analysis for agent tone")
            sentiments = state.full_call_analysis['sentiments']
            sentiments_a = [s[2] for s in sentiments if s[2]]
            return sentiment_analyzer.sentiment_appel_agent(sentiments_a)
        
        # Perform analysis
        _, sentiments, _ = chunk_processor.optimized_chunker(
            audio_path, topic=False, request=request
        )
        sentiments_a = [s[2] for s in sentiments if s[2]]
        return sentiment_analyzer.sentiment_appel_agent(sentiments_a)
    
    def generate_sentiment_graph(audio_files, dates, period):
        """Generate sentiment trend graph"""
        try:
            records = []
            
            for file, date_str in zip(audio_files, dates):
                try:
                    tone = get_agent_audio_tone(file.name)
                    date = pd.to_datetime(date_str)
                    records.append({"date": date, "tone": tone})
                except Exception as e:
                    logger.error(f"Error processing file {file.name}: {e}")
                    continue
            
            if not records:
                return "❌ Aucune donnée exploitable.", None
            
            df = pd.DataFrame(records)
            
            # Group by period
            if period == "day":
                df["period"] = df["date"].dt.date
            elif period == "week":
                df["period"] = df["date"].dt.to_period("W").astype(str)
            elif period == "month":
                df["period"] = df["date"].dt.to_period("M").astype(str)
            else:
                df["period"] = df["date"].dt.date
            
            # Create grouped data
            grouped = df.groupby(["period", "tone"]).size().unstack(fill_value=0)
            grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            grouped_percent.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=["#fbbbe0", "#d66ba0", "#a83279", "#58114c"]
            )
            
            plt.title("Répartition des tons globaux de l'agent", fontsize=16, fontweight='bold')
            plt.xlabel("Période", fontsize=12)
            plt.ylabel("Pourcentage des appels (%)", fontsize=12)
            plt.legend(title="Ton", bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return "✅ Analyse terminée avec succès", fig
            
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            return f"❌ Erreur lors de la génération: {str(e)}", None
    
    def handler_graph_generation(audio_files, date_input_str, period):
        """Handle graph generation request"""
        if not audio_files:
            return "❌ Aucun fichier audio fourni.", None
        
        dates = [d.strip() for d in date_input_str.strip().splitlines() if d.strip()]
        
        if len(dates) != len(audio_files):
            return (
                f"❌ Le nombre de dates ({len(dates)}) ne correspond pas "
                f"au nombre de fichiers audio ({len(audio_files)})."
            ), None
        
        return generate_sentiment_graph(audio_files, dates, period)
    
    # Wire up events
    audio_files.change(
        fn=update_audio_count,
        inputs=[audio_files],
        outputs=[audio_count_text]
    )
    
    submit_btn.click(
        fn=handler_graph_generation,
        inputs=[audio_files, date_inputs, period],
        outputs=[output_text, output_plot]
    )


# =============================================================================
# Helper Functions
# =============================================================================

def optimized_inference(
    audio_path: str,
    precomputed_sentiments: Optional[tuple] = None,
    request: Optional[gr.Request] = None
) -> str:
    """
    Optimized inference function with intelligent caching
    
    Args:
        audio_path: Path to audio file
        precomputed_sentiments: Pre-computed sentiments if available
        request: Gradio request object
        
    Returns:
        HTML formatted transcription with sentiment analysis
    """
    state = get_session_state(request)
    
    try:
        # Validate audio
        is_valid, validation_message = audio_processor.validate_audio_file(audio_path)
        if not is_valid:
            return f"<div style='color: red;'>❌ {validation_message}</div>"
        
        # Check cache
        cache_key = audio_path
        if cache_key in state.full_transcription_cache and not precomputed_sentiments:
            logger.info("Using cached full transcription")
            cached_data = state.full_transcription_cache[cache_key]
            return format_transcription_with_sentiment(
                cached_data['transcript'],
                cached_data['text_voice_agent'],
                cached_data['text_voice_client'],
                cached_data['acstc_voice_agent'],
                cached_data['acstc_voice_client'],
                cached_data['global_voice_agent'],
                cached_data['global_voice_client'],
                used_precomputed=True
            )
        
        # Process audio
        left, right, sample_rate = audio_processor.load_audio_channels(audio_path)
        
        # Get speech segments
        left_segments = audio_processor.get_speech_segments(left, sample_rate, "Agent")
        right_segments = audio_processor.get_speech_segments(right, sample_rate, "Client")
        
        # Transcribe segments
        transcript = chunk_processor.transcribe_segments(
            left_segments, sample_rate, config.audio.target_sample_rate, request
        )
        transcript += chunk_processor.transcribe_segments(
            right_segments, sample_rate, config.audio.target_sample_rate, request
        )
        transcript_sorted = sorted(transcript, key=lambda x: x["start"])
        
        # Get text for sentiment analysis
        agent_text = " ".join([seg["text"] for seg in transcript_sorted if seg["speaker"] == "Agent"])
        client_text = " ".join([seg["text"] for seg in transcript_sorted if seg["speaker"] == "Client"])
        
        # Get sentiment analysis
        if precomputed_sentiments and len(precomputed_sentiments) >= 6:
            logger.info("Using precomputed sentiments")
            (text_voice_agent, acstc_voice_agent, global_voice_agent,
             text_voice_client, acstc_voice_client, global_voice_client) = precomputed_sentiments[:6]
            # Fallback labels if any are missing
            text_voice_agent = text_voice_agent or "Inconnu"
            acstc_voice_agent = acstc_voice_agent or "Inconnu"
            global_voice_agent = global_voice_agent or "Inconnu"
            text_voice_client = text_voice_client or "Inconnu"
            acstc_voice_client = acstc_voice_client or "Inconnu"
            global_voice_client = global_voice_client or "Inconnu"
            used_precomputed = True
        else:
            logger.info("Performing new sentiment analysis")
            results = sentiment_analyzer.analyze_sentiment_client_agent(
                agent_text, client_text, audio_path
            )
            
            # Defensive unpacking
            text_voice_agent = (results[0] if len(results) > 0 else None) or "Inconnu"
            acstc_voice_agent = (results[1] if len(results) > 1 else None) or "Inconnu"
            global_voice_agent = (results[2] if len(results) > 2 else None) or "Inconnu"
            text_voice_client = (results[3] if len(results) > 3 else None) or "Inconnu"
            acstc_voice_client = (results[4] if len(results) > 4 else None) or "Inconnu"
            global_voice_client = (results[5] if len(results) > 5 else None) or "Inconnu"
            used_precomputed = False
            
            # Cache results
            state.full_transcription_cache[cache_key] = {
                'transcript': transcript_sorted,
                'text_voice_agent': text_voice_agent,
                'text_voice_client': text_voice_client,
                'acstc_voice_agent': acstc_voice_agent,
                'acstc_voice_client': acstc_voice_client,
                'global_voice_agent': global_voice_agent,
                'global_voice_client': global_voice_client
            }
        
        # Clean up temporary files
        for entry in transcript:
            if "audio_path" in entry and os.path.exists(entry["audio_path"]):
                try:
                    os.remove(entry["audio_path"])
                except:
                    pass
        
        return format_transcription_with_sentiment(
            transcript_sorted, text_voice_agent, text_voice_client,
            acstc_voice_agent, acstc_voice_client, global_voice_agent, global_voice_client,
            used_precomputed
        )
        
    except Exception as e:
        logger.error(f"Error in optimized_inference: {e}", exc_info=True)
        return f"<div style='color: red;'>Erreur lors de l'analyse: {str(e)}</div>"


def format_transcription_with_sentiment(
    transcript: List[Dict],
    text_voice_agent: str,
    text_voice_client: str,
    acstc_voice_agent: str,
    acstc_voice_client: str,
    global_voice_agent: str,
    global_voice_client: str,
    used_precomputed: bool = False
) -> str:
    """Format transcription with sentiment analysis results"""
    
    html = f"""
    <div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <div style="flex: 1; text-align: left; font-weight: bold; color: orange;">Client</div>
            <div style="flex: 1; text-align: right; font-weight: bold; color: orange;">Agent</div>
        </div>
    """
    
    # Add conversation bubbles
    for turn in transcript:
        speaker = str(turn.get("speaker", "")).strip()
        is_agent = speaker.lower() == "agent"
        speaker_class = "agent" if is_agent else "client"
        bubble_align = "margin-left:auto;" if is_agent else "margin-right:auto;"
        bubble_width = "max-width: 72%;"  # slightly narrower to avoid overlap
        html += f"""
        <div class=\"bubble-container\" style=\"direction: ltr; display:flex; width:100%;\">
            <div class=\"{speaker_class}\" style=\"{bubble_align} {bubble_width}\">{turn['text']}</div>
        </div>
        """
    
    # Add sentiment summary
    html += "<div class='result-container'>"

    # Helper for empty/undefined
    def fmt(v):
        return (v or "Inconnu").strip()

    # Client column
    html += "<div class='tone-column'>"
    html += "<div class='tone-title'>Client :</div>"
    html += f"<div class='final-tone'><b>Émotion du texte:</b> {fmt(text_voice_client)}</div>"
    html += f"<div class='final-tone'><b>Émotion de la voix:</b> {fmt(acstc_voice_client)}</div>"
    html += f"<div class='final-tone'><b>Émotion globale:</b> {fmt(global_voice_client)}</div>"
    html += "</div>"
    
    # Agent column
    html += "<div class='tone-column'>"
    html += "<div class='tone-title'>Agent :</div>"
    html += f"<div class='final-tone'><b>Émotion du texte:</b> {fmt(text_voice_agent)}</div>"
    html += f"<div class='final-tone'><b>Émotion de la voix:</b> {fmt(acstc_voice_agent)}</div>"
    html += f"<div class='final-tone'><b>Émotion globale:</b> {fmt(global_voice_agent)}</div>"
    html += "</div>"
    
    html += "</div></div>"
    
    return html 