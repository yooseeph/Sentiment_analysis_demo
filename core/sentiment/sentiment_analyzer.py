"""
Sentiment analysis wrapper for Sentiment Analysis Dashboard
"""
from typing import Tuple, Optional, Dict, Any, List
from collections import Counter

from config import config
from utils.logging_config import get_logger, log_sentiment_result, log_performance
from core.models import get_sentiment_evaluator

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Wrapper for sentiment analysis operations"""
    
    def __init__(self):
        self.evaluator = None
        self._load_evaluator()
        
    def _load_evaluator(self):
        """Load sentiment evaluator"""
        self.evaluator = get_sentiment_evaluator()
        if not self.evaluator:
            logger.error("Failed to load sentiment evaluator")
            
    def pretty_sentiment(self, label: Optional[str]) -> str:
        """Convert sentiment label to display format"""
        if label is None:
            return "Vide"
        return config.sentiment.sentiment_display.get(label, str(label).capitalize())
    
    @log_performance
    def analyze_sentiment_client_agent(
        self,
        agent_text: str,
        client_text: str,
        audio_path: str
    ) -> Tuple[str, ...]:
        """
        Analyze sentiment for both client and agent
        
        Args:
            agent_text: Agent's transcribed text
            client_text: Client's transcribed text
            audio_path: Path to audio file
            
        Returns:
            Tuple of sentiment results (6 values + metadata)
        """
        if not self.evaluator:
            logger.error("Sentiment evaluator not loaded")
            return tuple(["Erreur"] * 6) + ({}, {}, None, None)
        
        try:
            logger.info(f"Analyzing sentiment for audio: {audio_path}")
            results = self.evaluator.predict_dual_sentiment_optimized(
                agent_text, client_text, audio_path
            )

            # Extract agent sentiments
            agent_r = results['agent']
            text_pred_a = agent_r.get('text_prediction')
            acoustic_pred_a = agent_r.get('acoustic_prediction')
            fusion_pred_a = agent_r.get('fusion_prediction')
            per_modality_a = agent_r.get('per_modality', {})
            fusion_prob_a = agent_r.get('fusion_prob')
            
            # Extract client sentiments
            client_r = results['client']
            text_pred_c = client_r.get('text_prediction')
            acoustic_pred_c = client_r.get('acoustic_prediction')
            fusion_pred_c = client_r.get('fusion_prediction')
            per_modality_c = client_r.get('per_modality', {})
            fusion_prob_c = client_r.get('fusion_prob')
            
            # Log results
            log_sentiment_result("Agent", text_pred_a, acoustic_pred_a, fusion_pred_a)
            log_sentiment_result("Client", text_pred_c, acoustic_pred_c, fusion_pred_c)
            
            return (
                self.pretty_sentiment(text_pred_a),
                self.pretty_sentiment(acoustic_pred_a),
                self.pretty_sentiment(fusion_pred_a),
                self.pretty_sentiment(text_pred_c),
                self.pretty_sentiment(acoustic_pred_c),
                self.pretty_sentiment(fusion_pred_c),
                per_modality_a,
                per_modality_c,
                fusion_prob_a,
                fusion_prob_c
            )
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
            return tuple(["Erreur"] * 6) + ({}, {}, None, None)
    
    def sentiment_appel_client(self, sentiments: List[str]) -> str:
        """
        Determine overall client sentiment from list of sentiments
        
        Args:
            sentiments: List of sentiment labels
            
        Returns:
            Overall sentiment
        """
        if not sentiments:
            return "Inconnu"

        sentiments = [s.strip() for s in sentiments if s and s.strip()]
        if not sentiments:
            return "Inconnu"
        
        count = Counter(sentiments)
        total = len(sentiments)

        # Priority 1: Last emotion is "Content"
        if sentiments[-1] == "Content":
            return "Content"

        # Priority 2: Presence of "Très Mécontent"
        if "Très Mécontent" in count: 
            return "Très Mécontent"

        # Priority 3: Presence of "Mécontent"
        if "Mécontent" in count:
            return "Mécontent"

        # Priority 4: ≥50% "Neutre" AND no negative emotions
        if count.get("Neutre", 0) / total >= 0.5:
            return "Neutre"

        # Priority 5: Most common emotion
        candidates = ["Content", "Mécontent", "Très Mécontent", "Neutre"]
        dominant = max(candidates, key=lambda x: count.get(x, 0))
        
        logger.info(f"Client overall sentiment: {dominant} from {dict(count)}")
        return dominant
    
    def sentiment_appel_agent(self, sentiments: List[str]) -> str:
        """
        Determine overall agent sentiment from list of sentiments
        
        Args:
            sentiments: List of sentiment labels
            
        Returns:
            Overall sentiment
        """
        if not sentiments:
            return "Inconnu"

        sentiments = [s.strip() for s in sentiments if s and s.strip()]
        if not sentiments:
            return "Inconnu"
        
        count = Counter(sentiments)
        total = len(sentiments)

        # Priority 1: ≥1 occurrence of "Agressif"
        if "Agressif" in count:
            return "Agressif"

        # Priority 2: Last tone is "Sec" OR ≥30% "Sec"
        if sentiments[-1] == "Sec" or count.get("Sec", 0) / total >= 0.3:
            return "Sec"

        # Priority 3: Last tone is "Courtois" AND ≥50% "Courtois"
        if sentiments[-1] == "Courtois" and count.get("Courtois", 0) / total >= 0.5:
            return "Courtois"

        # Priority 4: Last tone is "Neutre" AND no "Sec"
        if sentiments[-1] == "Neutre" and "Sec" not in count:
            return "Neutre"

        # Priority 5: Most common tone
        candidates = ["Agressif", "Sec", "Courtois", "Neutre"]
        dominant = max(candidates, key=lambda x: count.get(x, 0))
        
        logger.info(f"Agent overall sentiment: {dominant} from {dict(count)}")
        return dominant


# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer() 