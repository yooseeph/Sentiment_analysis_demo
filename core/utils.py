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


def analyze_topic(transcription: str) -> str:
    """
    Analyze the topic of the transcription
    
    Args:
        transcription: Transcribed text
        
    Returns:
        Topic classification result
    """
    try:
        from utils.topics_inf_clean import infer
        # _, cat, typ = infer(transcription)
        _, cat, typ = "Appel blanc", "Appel blanc", "Appel blanc"
        topic = f"{cat} - {typ}"
        logger.info(f"Topic analysis result: {topic}")
        return topic
    except Exception as e:
        logger.error(f"Error in topic analysis: {e}")
        return "Appel blanc"


def get_custom_css() -> str:
    """Return custom CSS styles for the interface"""
    return """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    .bubble-container {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
        line-height: 1.8;
        margin-top: 10px;
        margin-bottom: 22px;
        padding-bottom: 6px;
        display: flex;
        align-items: flex-start;
        width: 100%;
        box-sizing: border-box;
    }
    .bubble-container:last-child {
        margin-bottom: 0;
    }
        
    .agent, .client {
        background-color: #A62182;
        color: #fff !important;
        padding: 12px 16px;
        border-radius: 14px;
        max-width: 68%;
        margin-left: 12px;
        margin-right: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.08);
    }
        
    .tag {
        background-color: #58114c;
        color: #fff !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin: 0 5px;
    }
        
    /* Result Container */
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
        border-left: 3px solid #ddd !important;
        background-color: #f9f9f9 !important;
        border-radius: 8px;
        color: #222 !important;
    }
    .tone-column * {
        color: #222 !important;
    }
        
    .tone-title {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 18px;
        color: #111 !important;
        font-weight: 600;
    }
        
    .final-tone {
        margin-bottom: 10px;
        font-size: 15px;
        color: #222 !important;
    }
        
    /* Button Styling */
    .primary-button, #analyser-button {
        background-color: #A62182 !important;
        color: white !important;
        font-weight: bold;
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
            
    /* Radio Group Styling */
    div[data-testid="radio-group"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px;
        max-width: 600px;
        width: 100%;
        flex-wrap: nowrap !important;
    }
        
    div[data-testid="radio-option"] {
        background: #f9f9f9;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #ddd;
        width: 100% !important;
        box-sizing: border-box;
        transition: background-color 0.2s ease;
    }
        
    div[data-testid="radio-option"]:hover {
        background-color: #eee;
    }
        
    /* Header Styling */
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
        color: #333 !important;
        margin: 0;
    }
        
    /* Status indicator */
    .status-indicator {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 10px 0;
        font-weight: bold;
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
        color: #A62182;
        font-weight: bold;
    }
</style>
""" 