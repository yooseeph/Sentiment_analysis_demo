"""
Utility functions for Sentiment Analysis Dashboard
"""
import os
import re
import base64
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import pandas as pd
import gradio as gr

from config import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DarijaFrenchConverter:
    """Handles Darija to French text conversion"""
    
    def __init__(self):
        self.mapping: Dict[str, str] = {}
        self.sorted_keys: List[str] = []
        self._load_dictionary()
        
    def _load_dictionary(self):
        """Load Darija to French conversion dictionary"""
        try:
            df = pd.read_excel(config.data.darija_french_dict)
            
            # First column contains French words
            french_words = df.iloc[:, 0]
            
            # Other columns contain Darija variants
            for col in df.columns[1:]:
                for french, darija in zip(french_words, df[col]):
                    if pd.notna(darija):
                        self.mapping[darija.strip()] = french.strip()
                        
            # Sort keys by length (longest first) for better matching
            self.sorted_keys = sorted(self.mapping.keys(), key=len, reverse=True)
            
            logger.info(f"Loaded Darija dictionary with {len(self.mapping)} entries")
            
        except Exception as e:
            logger.error(f"Error loading Darija dictionary: {e}")
            
    def convert_text(self, text: str) -> str:
        """
        Convert embedded Darija words to French
        
        Args:
            text: Input text with Darija words
            
        Returns:
            Text with Darija words converted to French
        """
        if not self.mapping:
            return text
            
        for darija_variant in self.sorted_keys:
            pattern = re.compile(rf"\b{re.escape(darija_variant)}\b", flags=re.IGNORECASE)
            text = pattern.sub(f" {self.mapping[darija_variant]} ", text)
            
        # Clean up extra spaces
        return ' '.join(text.split())


# Global converter instance
darija_converter = DarijaFrenchConverter()


def get_base64_image(image_path: str) -> str:
    """
    Convert image to base64 for HTML embedding
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return ""


def get_session_csv_path(request: Optional[gr.Request] = None) -> str:
    """
    Get session-specific CSV path for logging
    
    Args:
        request: Gradio request object
        
    Returns:
        Path to session CSV file
    """
    session_id = getattr(request, 'session_hash', 'default') if request else 'default'
    csv_path = os.path.join(config.gradio.temp_dir, f"chunks_analysis_log_{session_id}.csv")
    return csv_path


def get_session_csv_path_by_id(session_id: str) -> str:
    """Get session-specific CSV path from a known session id."""
    return os.path.join(config.gradio.temp_dir, f"chunks_analysis_log_{session_id}.csv")


def append_rows_to_csv(rows: List[Dict], request: Optional[gr.Request] = None):
    """
    Append rows to session-specific CSV file
    
    Args:
        rows: List of dictionaries to append
        request: Gradio request object
    """
    if not rows:
        return
        
    try:
        csv_path = get_session_csv_path(request)
        df = pd.DataFrame(rows)
        
        # Check if file exists to determine if we need headers
        write_header = not os.path.exists(csv_path)
        
        # Append to CSV
        df.to_csv(csv_path, mode="a", header=write_header, index=True, encoding="utf-8")
        
        logger.debug(f"Appended {len(rows)} rows to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error appending to CSV: {e}")


def analyze_topic(transcription: str, business_type: str = "B2C") -> str:
    """
    Analyze the topic of the transcription
    
    Args:
        transcription: Transcribed text
        business_type: Type of business analysis ("B2B" or "B2C")
        
    Returns:
        Topic classification result
    """
    try:
        from utils.topics_inf_clean import TopicClassifier
        classifier = TopicClassifier(business_type=business_type)
        _, cat, typ = classifier.infer(transcription)
        topic = f"{cat} - {typ}"
        logger.info(f"Topic analysis result [{business_type}]: {topic}")
        return topic
    except Exception as e:
        logger.error(f"Error in topic analysis [{business_type}]: {e}")
        return "VIDE"

def get_custom_css() -> str:
    """Return custom CSS styles for the interface with proper dark mode support"""
    return """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Force Poppins as default for all elements with maximum specificity */
    *, *::before, *::after,
    body, div, span, p, h1, h2, h3, h4, h5, h6, 
    button, input, textarea, label, select, option,
    .gradio-container, .gradio-container *,
    .gr-button, .gr-input, .gr-textbox, .gr-radio,
    .gr-form, .gr-panel, .gr-block, .gr-box,
    [class*="gradio"], [class*="gr-"],
    .svelte-1ed2p3z, .svelte-1rjryqp, .svelte-nlb5t9,
    .app, .container, .main {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
    }
    
    /* Keep Cairo specifically for Arabic transcription bubbles with higher specificity */
    .bubble-container,
    .bubble-container *,
    .bubble-container .agent,
    .bubble-container .client,
    .agent,
    .client {
        font-family: 'Cairo', 'Amiri', 'Times New Roman', serif !important;
    }

    /* ---------------- BiDi fixes (RTL/LTR mixing) ---------------- */
    /* Give Arabic bubbles an RTL base and isolate their content so LTR tokens don't reorder */
    .bubble-container {
        direction: rtl;              /* base direction is RTL */
        unicode-bidi: isolate;       /* isolate from siblings/parents */
        text-align: start;           /* aligns correctly for RTL */
        line-height: 2.0;
        margin-top: 10px;
        margin-bottom: 10px;
        padding-bottom: 10px;
        display: flex;
        align-items: flex-start;
        width: 100%;
        box-sizing: border-box;
    }
    .bubble-container:last-child { margin-bottom: 0; }

    .agent, .client {
        direction: rtl;              /* ensure each bubble itself is RTL */
        unicode-bidi: isolate;       /* or 'plaintext' if you prefer; isolate is widely supported */
        background-color: #851868;
        color: #fff !important;
        padding: 12px 16px;
        border-radius: 14px;
        max-width: 68%;
        margin-left: 12px;
        margin-right: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: start;           /* keeps natural alignment for RTL text */
        white-space: pre-wrap;       /* preserve spaces/line breaks if present */
    }

    /* Optional helper class: wrap Latin fragments if/when you can in HTML:
       <span class="ltr">inwi</span> */
    .ltr { direction: ltr; unicode-bidi: isolate; }
    /* ------------------------------------------------------------- */
        
    .tag {
        background-color: #58114c;
        color: #fff !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin: 0 5px;
    }
        
    /* Result Container - Fixed for dark mode */
    .result-container {
        display: flex;
        justify-content: space-between;
        gap: 40px;
        margin-top: 40px;
        flex-wrap: wrap;
        filter: none !important;
        mix-blend-mode: normal !important;
    }
        
    .tone-column {
        flex: 1;
        min-width: 280px;
        padding: 16px;
        border-left: 3px solid var(--border-color-primary, #ddd) !important;
        background-color: var(--background-fill-primary, #f9f9f9) !important;
        border-radius: 8px;
        color: var(--body-text-color, #222) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    .tone-column * {
        color: var(--body-text-color, #222) !important;
        font-family: 'Poppins', sans-serif !important;
    }
        
    .tone-title {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 14px !important;
        color: #F25C05 !important;
        font-weight: 500;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Labels above transcription bubbles (Client/Agent) */
    .transcription-label,
    .speaker-label,
    h3:contains("Client"),
    h3:contains("Agent"),
    .client-header,
    .agent-header {
        color: #F25C05 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        margin-bottom: 8px !important;
        text-transform: uppercase;
    }
        
    .final-tone {
        margin-bottom: 10px;
        font-size: 15px;
        color: var(--body-text-color, #222) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Dark mode overrides */
    .dark .tone-column,
    [data-theme="dark"] .tone-column {
        background-color: var(--background-fill-secondary, rgba(255,255,255,0.05)) !important;
        border-left-color: var(--border-color-accent, #555) !important;
    }
    
    .dark .tone-column *,
    [data-theme="dark"] .tone-column * {
        color: var(--body-text-color, #fff) !important;
    }
    
    .dark .tone-title,
    [data-theme="dark"] .tone-title {
        color: var(--body-text-color, #fff) !important;
    }
    
    .dark .final-tone,
    [data-theme="dark"] .final-tone {
        color: var(--body-text-color-subdued, #ccc) !important;
    }
        
    /* Button Styling */
    .primary-button, #analyser-button {
        background-color: #A62182 !important;
        color: white !important;
        font-weight: 500 !important;
        font-family: 'Poppins', sans-serif !important;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-size: 16px;
        text-align: center;
        transition: background-color 0.3s ease;
        width: 100%;
        margin-top: 20px;
    }

    .primary-button:hover, #analyser-button:hover {
        background-color: #db1ac2 !important;
        cursor: pointer;
    }
            
    /* Radio Group Styling - Fixed for dark mode */
    div[data-testid="radio-group"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px;
        max-width: 600px;
        width: 100%;
        flex-wrap: nowrap !important;
    }
        
    div[data-testid="radio-option"] {
        background: var(--background-fill-secondary, #f9f9f9);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid var(--border-color-primary, #ddd);
        width: 100% !important;
        box-sizing: border-box;
        transition: background-color 0.2s ease;
        color: var(--body-text-color, #000);
        font-family: 'Poppins', sans-serif !important;
    }
        
    div[data-testid="radio-option"]:hover {
        background-color: var(--background-fill-primary, #eee);
    }
    
    /* Dark mode radio options */
    .dark div[data-testid="radio-option"],
    [data-theme="dark"] div[data-testid="radio-option"] {
        background: var(--background-fill-secondary, rgba(255,255,255,0.05));
        border-color: var(--border-color-primary, #555);
        color: var(--body-text-color, #fff);
    }
    
    .dark div[data-testid="radio-option"]:hover,
    [data-theme="dark"] div[data-testid="radio-option"]:hover {
        background-color: var(--background-fill-primary, rgba(255,255,255,0.1));
    }
        
    /* Header Styling - Fixed for dark mode */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        padding: 10px 0;
        margin-bottom: 20px;
    }
        
    .header-title {
        text-align: center;
        color: var(--body-text-color, #333) !important;
        margin: 0;
        font-weight: 600;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Dark mode header */
    .dark .header-title,
    [data-theme="dark"] .header-title {
        color: var(--body-text-color, #fff) !important;
    }
    
    /* Media query fallback for header */
    @media (prefers-color-scheme: dark) {
        .header-title { color: #fff !important; }
    }
        
    /* Status indicator - Fixed for dark mode */
    .status-indicator {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 10px 0;
        font-weight: bold;
        font-family: 'Poppins', sans-serif !important;
    }
        
    .status-cached {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
        
    .status-fresh {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    /* Dark mode status indicators */
    .dark .status-cached,
    [data-theme="dark"] .status-cached {
        background-color: rgba(40, 167, 69, 0.2);
        color: #90ee90;
        border-color: rgba(40, 167, 69, 0.3);
    }
    
    .dark .status-fresh,
    [data-theme="dark"] .status-fresh {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffd700;
        border-color: rgba(255, 193, 7, 0.3);
    }

    /* Hide global Gradio queue status */
    [data-testid="queue-status"],
    [data-testid="queue"],
    div.svelte-queue,
    div#queue {
        display: none !important;
    }
    
    /* Processing indicator */
    .processing-indicator {
        text-align: center;
        padding: 20px;
        font-size: 18px;
        color: #F25C05 !important;
        font-weight: normal;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Additional fixes for any remaining white backgrounds */
    .dark div[style*="background-color: white"],
    .dark div[style*="background: white"],
    [data-theme="dark"] div[style*="background-color: white"],
    [data-theme="dark"] div[style*="background: white"] {
        background: var(--background-fill-primary, transparent) !important;
        color: var(--body-text-color, #fff) !important;
    }
    
    /* Force Poppins on common Gradio elements that might be stubborn */
    .gr-text-input,
    .gr-textbox,
    .gr-dropdown,
    .gr-slider,
    .gr-number,
    .gr-checkbox,
    .gr-radio,
    .gr-button,
    .gr-file,
    .gr-image,
    .gr-audio,
    .gr-video,
    .gr-dataframe,
    .gr-html,
    .gr-json,
    .gr-markdown,
    .gr-code,
    .gr-plot,
    .gr-3d,
    .gr-chatbot,
    .gr-model3d {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Target Gradio's internal classes more specifically */
    .svelte-1ed2p3z *,
    .svelte-1rjryqp *,
    .svelte-nlb5t9 *,
    .gradio-container .wrap *,
    .gradio-container .block *,
    .gradio-container .form * {
        font-family: 'Poppins', sans-serif !important;
    }
</style>
"""