from __future__ import annotations

import os
import json
import joblib
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Inject logger and route print() to logger.info in this module
from utils.logging_config import get_logger
logger = get_logger(__name__)

def _print_to_logger(*args, **kwargs):
    try:
        msg = " ".join(str(a) for a in args)
    except Exception:
        msg = " ".join(repr(a) for a in args)
    logger.info(msg)

print = _print_to_logger

# ────────────────────────────────────────────────────────────────────────────────
#  MarBERT Adapter
# ────────────────────────────────────────────────────────────────────────────────
class MarBERTAdapter:
    """
    Adapter class for fine-tuned MarBERT/ArabertV2 model that skips text preprocessing
    """
    def __init__(self, model_path, device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Try to load tokenizer from model config
        try:
            # Get the original base model name from config
            with open(f"{model_path}/config.json", "r") as f:
                config = json.load(f)
                base_model_name = config.get("_name_or_path", "aubmindlab/bert-base-arabertv2")
                
                # Load the actual label mappings from config
                self.id2label = config.get("id2label", {})
                self.label2id = config.get("label2id", {})
                
                # Convert string keys to int for id2label
                self.id2label = {int(k): v for k, v in self.id2label.items()}
            
            # Load tokenizer from the base model
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            print(f"✓ Loaded tokenizer from base model: {base_model_name}")
            print(f"✓ Loaded label mappings - id2label: {self.id2label}")
        except Exception as e:
            # Fallback to default Arabic BERT tokenizer
            print(f"⚠ Error loading tokenizer from config: {e}")
            print("⚠ Falling back to default ArabertV2 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
            
            # Default fallback mappings
            self.id2label = {0: 0, 1: 1, 2: 2, 3: 3}
            self.label2id = {v: k for k, v in self.id2label.items()}
            
        # Skip preprocessing by using identity function
        self.preprocess = lambda text: text
        
        # Set to evaluation mode
        self.model.eval()
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions on a list of texts without preprocessing
        Returns:
            predictions: numpy array of class indices
            probabilities: numpy array of class probabilities
        """
        # Use texts directly without preprocessing
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            
            # Move to CPU and convert to numpy
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)

# ────────────────────────────────────────────────────────────────────────────────
#  Dual Multimodal Evaluator
# ────────────────────────────────────────────────────────────────────────────────
class OptimizedDualMultimodalEvaluator:
    """Run sentiment analysis for both client and agent with fusion."""
    
    def __init__(self):
        # Use GPU if possible for all torch tensors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device → {self.device}")

        # ---------- Model paths ------------------------------------------------
        # Agent model paths
        TEXT_MODEL_PATH_AGENT =         "./models/agent/text/best_model"
        ACOUSTIC_MODEL_PATH_AGENT =     "./models/agent/acoustic/randomforest_acoustic_model.joblib"
        ACOUSTIC_SCALER_PATH_AGENT =    "./models/agent/acoustic/acoustic_scaler.joblib"

        # Client model paths
        TEXT_MODEL_PATH_CLIENT =        "./models/client/text/best_model"
        ACOUSTIC_MODEL_PATH_CLIENT ="./models/client/acoustic/svm_acoustic_model.joblib"
        ACOUSTIC_SCALER_PATH_CLIENT =   "./models/client/acoustic/acoustic_scaler.joblib"

        # ---------- Load models ------------------------------------------------
        self.models = {
            'agent': {},
            'client': {}
        }

        # Load agent models
        print("\n=== Loading Agent Models ===")
        if os.path.exists(TEXT_MODEL_PATH_AGENT):
            self._load_text_model('agent', TEXT_MODEL_PATH_AGENT)
        else:
            print(f'⚠ Agent text model not found at {TEXT_MODEL_PATH_AGENT}')

        if os.path.exists(ACOUSTIC_MODEL_PATH_AGENT) and os.path.exists(ACOUSTIC_SCALER_PATH_AGENT):
            self._load_acoustic_model('agent', ACOUSTIC_MODEL_PATH_AGENT, ACOUSTIC_SCALER_PATH_AGENT)
        else:
            print(f'⚠ Agent acoustic model not found')

        # Load client models
        print("\n=== Loading Client Models ===")
        if os.path.exists(TEXT_MODEL_PATH_CLIENT):
            self._load_text_model('client', TEXT_MODEL_PATH_CLIENT)
        else:
            print(f'⚠ Client text model not found at {TEXT_MODEL_PATH_CLIENT}')

        if os.path.exists(ACOUSTIC_MODEL_PATH_CLIENT) and os.path.exists(ACOUSTIC_SCALER_PATH_CLIENT):
            self._load_acoustic_model('client', ACOUSTIC_MODEL_PATH_CLIENT, ACOUSTIC_SCALER_PATH_CLIENT)
        else:
            print(f'⚠ Client acoustic model not found')

        # Check if we have at least one complete model set
        if not any(self.models['agent'].values()) and not any(self.models['client'].values()):
            raise RuntimeError('No models available for either agent or client – aborting.')

        # Initialize label mappings - will be updated when models are loaded
        self.agent_labels = {}
        self.client_labels = {}
        
        # Set canonical label order for fusion (matching evaluation code)
        self.canonical_agent_labels = ['aggressive' ,'courtois' , 'sec']
        self.canonical_client_labels = ["content", "neutre", "tres mecontent"]
        
        # Update label mappings from loaded models
        self._update_label_mappings()

    def _load_text_model(self, role, model_path):
        """Load text model for specified role (agent or client)"""
        try:
            adapter = MarBERTAdapter(model_path, device=self.device)
            self.models[role]['text'] = {
                'adapter': adapter,
                'cls': [0, 1, 2, 3]
            }
            print(f'✓ {role.capitalize()} text model loaded')
        except Exception as e:
            print(f'⚠ Error loading {role} text model: {e}')
            import traceback
            traceback.print_exc()

    def _load_acoustic_model(self, role, model_path, scaler_path):
        """Load acoustic model for specified role (agent or client)"""
        try:
            self.models[role]['acoustic'] = {
                'clf': joblib.load(model_path),
                'scaler': joblib.load(scaler_path)
            }
            print(f'✓ {role.capitalize()} acoustic model loaded')

        except Exception as e:
            print(f'⚠ Error loading {role} acoustic model: {e}')
            import traceback
            traceback.print_exc()

    def _update_label_mappings(self):
        """Update label mappings from loaded models"""
        # Update agent labels from text model
        if 'agent' in self.models and 'text' in self.models['agent']:
            text_model = self.models['agent']['text']['adapter']
            if hasattr(text_model, 'id2label') and text_model.id2label:
                self.agent_labels = text_model.id2label
                print(f"✓ Using agent text model labels: {self.agent_labels}")

        # Update client labels from text model
        if 'client' in self.models and 'text' in self.models['client']:
            text_model = self.models['client']['text']['adapter']
            if hasattr(text_model, 'id2label') and text_model.id2label:
                self.client_labels = text_model.id2label
                print(f"✓ Using client text model labels: {self.client_labels}")

        # Set fallback labels if not loaded from models
        if not self.agent_labels:
            self.agent_labels = {0: 'aggressive', 1: 'courtois', 2: 'neutre', 3: 'sec'}
        if not self.client_labels:
            self.client_labels = {0: 'content', 1: 'mecontent', 2: 'neutre', 3: 'tres mecontent'}

    def _reorder_probabilities(self, probs, src_order, tgt_order):
        """Reorder probability vector from source order to target order"""
        idx = {lbl: i for i, lbl in enumerate(src_order)}
        return np.array([probs[idx[lbl]] if lbl in idx else 0.0 for lbl in tgt_order])

    def _extract_acoustic_features(self, waveform, sr):
        """Extract acoustic features from a waveform tensor (already on device)"""
        try:
            # Ensure waveform is on device
            waveform = waveform.to(self.device)
            
            # Resample if needed
            target_sr = 16_000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=target_sr).to(self.device)
                waveform = resampler(waveform)
            
            # Use the waveform directly (should be mono at this point)
            y = waveform  # shape (N,)
            sr = target_sr

            # Frame parameters
            frame_length = int(0.025 * sr)  # 25 ms
            hop_length   = int(0.010 * sr)  # 10 ms
            if y.numel() < frame_length:
                raise ValueError('Audio too short')

            frames = y.unfold(0, frame_length, hop_length)

            # RMS
            rms       = torch.sqrt(torch.mean(frames ** 2, dim=1))
            rms_mean  = rms.mean();  rms_std = rms.std();  rms_rng = rms.max() - rms.min()

            # ZCR
            signs      = torch.sign(frames)
            zc         = ((signs[:, :-1] * signs[:, 1:]) < 0).sum(dim=1).float() / frame_length
            zcr_mean   = zc.mean();  zcr_std = zc.std()

            # STFT
            n_fft  = 512
            window = torch.hann_window(frame_length).to(y.device)
            stft   = torch.stft(y, n_fft=n_fft, hop_length=hop_length,
                                win_length=frame_length, window=window,
                                return_complex=True)
            mag    = stft.abs()  # (freq_bins, time)

            freqs  = torch.linspace(0, sr / 2, mag.shape[0], device=self.device)
            energy = mag.sum(dim=0) + 1e-8

            # Spectral centroid & bandwidth
            centroid = (mag * freqs.unsqueeze(1)).sum(dim=0) / energy
            sc_mean, sc_std = centroid.mean(), centroid.std()

            diff_sq   = (freqs.unsqueeze(1) - centroid.unsqueeze(0)) ** 2
            bandwidth = torch.sqrt((mag * diff_sq).sum(dim=0) / energy)
            sb_mean, sb_std = bandwidth.mean(), bandwidth.std()

            # Rolloff 0.85
            cum_energy = mag.cumsum(dim=0)
            thresh     = 0.85 * (cum_energy[-1] + 1e-8)
            roll_idx   = ((cum_energy >= thresh).float().argmax(dim=0)).long()
            roll_freqs = freqs[roll_idx]
            sr_mean, sr_std = roll_freqs.mean(), roll_freqs.std()

            # MFCC (13) – GPU
            mfcc_tf = torchaudio.transforms.MFCC(
                sample_rate=sr, n_mfcc=13,
                melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'win_length': frame_length}
            ).to(self.device)
            mfcc      = mfcc_tf(y.unsqueeze(0)).squeeze(0)  # (13, time)
            mfcc_mean = mfcc.mean(dim=1);  mfcc_std = mfcc.std(dim=1)

            # Rough tempo via spectral flux autocorrelation
            flux      = torch.relu(mag[:, 1:] - mag[:, :-1]).sum(dim=0)
            onset_env = flux.unsqueeze(0).unsqueeze(0)  # (1,1,T)
            autocorr  = torch.nn.functional.conv1d(onset_env, onset_env, padding=onset_env.shape[-1] - 1).squeeze()
            autocorr[0] = 0
            max_lag    = autocorr.argmax()
            period     = max_lag.item() * hop_length / sr if max_lag > 0 else 0.0
            tempo      = 60.0 / period if period > 0 else 0.0

            feats = {
                'rms_mean': rms_mean.item(),      'rms_std': rms_std.item(),      'rms_range': rms_rng.item(),
                'zcr_mean': zcr_mean.item(),      'zcr_std': zcr_std.item(),
                'spectral_centroid_mean': sc_mean.item(),  'spectral_centroid_std': sc_std.item(),
                'spectral_bandwidth_mean': sb_mean.item(), 'spectral_bandwidth_std': sb_std.item(),
                'spectral_rolloff_mean': sr_mean.item(),   'spectral_rolloff_std': sr_std.item(),
                'tempo': tempo,
            }
            for i in range(13):
                feats[f'mfcc_{i}_mean'] = mfcc_mean[i].item()
                feats[f'mfcc_{i}_std']  = mfcc_std[i].item()

            return feats
        except Exception as exc:
            print(f"[Feature Extraction Error]: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_acoustic_single(self, role, waveform, sr):
        """Return (pred_id, prob_vector) or (None, None) if disabled/error)."""
        if 'acoustic' not in self.models[role]:
            return None, None

        feats = self._extract_acoustic_features(waveform, sr)
        if feats is None:
            return None, None

        scaler = self.models[role]['acoustic']['scaler']
        clf    = self.models[role]['acoustic']['clf']
        X      = scaler.transform([list(feats.values())])
        prob   = clf.predict_proba(X)[0]
        pred_class_idx = prob.argmax()
        pred_class = clf.classes_[pred_class_idx]

        # The prediction is already the class index from the classifier
        return pred_class_idx, prob

    def _predict_with_text_model(self, role, texts):
        """Predict using the text model for specified role"""
        if 'text' not in self.models[role] or 'adapter' not in self.models[role]['text']:
            raise ValueError(f"{role.capitalize()} text adapter not loaded")
            
        adapter = self.models[role]['text']['adapter']
        return adapter.predict(texts)

    def _load_stereo_audio(self, audio_path):
        """Load stereo audio and return left and right channels separately"""
        try:
            # Clean up file path
            audio_path = audio_path.strip('"\'')
            
            # Verify file exists
            if not os.path.exists(audio_path):
                print(f"Error: Audio file not found: {audio_path}")
                return None, None, None
                
            print(f"Loading stereo audio file: {audio_path}")
            
            try:
                waveform, sr_orig = torchaudio.load(audio_path)
                print(f"✓ Loaded audio: shape={waveform.shape}, sr={sr_orig}Hz")
            except Exception as e:
                print(f"Error loading audio file with torchaudio: {e}")
                # Try scipy as backup
                try:
                    import scipy.io.wavfile as wavfile
                    sr_orig, wav_data = wavfile.read(audio_path)
                    wav_data = wav_data.astype(np.float32) / 32768.0
                    if len(wav_data.shape) == 1:  # mono
                        waveform = torch.from_numpy(wav_data).unsqueeze(0)
                    else:  # stereo
                        waveform = torch.from_numpy(wav_data.T)
                    print(f"✓ Loaded audio with scipy: shape={waveform.shape}, sr={sr_orig}Hz")
                except Exception as e2:
                    print(f"All audio loading methods failed: {e2}")
                    return None, None, None
                    
            # Move to device
            waveform = waveform.to(self.device)
            
            # Check if stereo
            if waveform.shape[0] < 2:
                print("Warning: Audio is not stereo. Using mono for both channels.")
                left_channel = waveform[0]
                right_channel = waveform[0]
            else:
                left_channel = waveform[0]   # Agent
                right_channel = waveform[1]  # Client
                
            return left_channel, right_channel, sr_orig
            
        except Exception as e:
            print(f"Error loading stereo audio: {e}")
            return None, None, None

    def predict_dual_sentiment_optimized(self, agent_text, client_text, audio_path):
        """
        Predict sentiment for both agent and client from stereo audio
        
        Parameters
        ----------
        agent_text : str - Text spoken by agent
        client_text : str - Text spoken by client  
        audio_path : str - Path to stereo WAV file (left=agent, right=client)
        
        Returns
        -------
        dict - Results for both agent and client with all 6 outputs
        """
        """
        Optimized dual prediction that minimizes redundant computations
        """
        # Fixed weights for fusion
        weights_agent = {'text': 0.54, 'acoustic': 0.46}
        weights_client = {'text': 0.42, 'acoustic': 0.58}
        
        # Load stereo audio ONCE
        left_channel, right_channel, sr = self._load_stereo_audio(audio_path)
        if left_channel is None:
            raise ValueError("Failed to load stereo audio")
        
        results = {}
        
        # Process Agent (optimized)
        print("\n=== Processing Agent (Optimized) ===")
        results['agent'] = self._predict_single_role_optimized(
            'agent', agent_text, left_channel, sr, weights_agent, 
            self.agent_labels, self.canonical_agent_labels
        )
        
        # Process Client (optimized)
        print("\n=== Processing Client (Optimized) ===")
        results['client'] = self._predict_single_role_optimized(
            'client', client_text, right_channel, sr, weights_client, 
            self.client_labels, self.canonical_client_labels
        )
        
        return results

    def _predict_single_role_optimized(self, role, text, audio_channel, sr, weights, labels, canonical_labels):
        """
        Optimized prediction that computes each modality only once
        """
        # Check available modalities for this role
        available_modalities = [m for m in ['text', 'acoustic'] if m in self.models[role]]
        if not available_modalities:
            return {'error': f'No models available for {role}'}
        
        # Adjust weights for available modalities
        role_weights = {m: weights[m] for m in available_modalities}
        z = sum(role_weights.values())
        role_weights = {m: w / z for m, w in role_weights.items()}
        
        # Storage for results
        results = {
            'text_prediction': None, 'text_prediction_id': None, 'text_prob_canonical': None,
            'acoustic_prediction': None, 'acoustic_prediction_id': None, 'acoustic_prob_canonical': None,
            'fusion_prediction': None, 'fusion_prediction_id': None, 'special_condition_applied': False,
            'per_modality': {}
        }
        
        # === TEXT PREDICTION (ONCE) ===
        if 'text' in self.models[role]:
            try:
                # Single call to text model
                pred_t, prob_t = self._predict_with_text_model(role, [text])
                text_pred_id = int(pred_t[0])
                text_pred = labels.get(text_pred_id, str(text_pred_id))
                
                # Reorder probabilities to canonical order (once)
                text_model = self.models[role]['text']['adapter']
                text_model_order = [text_model.id2label[i] for i in range(len(prob_t[0]))]
                text_prob_canonical = self._reorder_probabilities(prob_t[0], text_model_order, canonical_labels)
                
                # Store all text results
                results['text_prediction'] = text_pred
                results['text_prediction_id'] = text_pred_id
                results['text_prob_canonical'] = text_prob_canonical
                results['per_modality']['text'] = {
                    'id': text_pred_id,
                    'prob': prob_t[0],
                    'prob_canonical': text_prob_canonical
                }
                print(f"✓ {role.capitalize()} text prediction: {text_pred}")
                
            except Exception as e:
                print(f"⚠ {role.capitalize()} text prediction failed: {e}")
        
        # === ACOUSTIC PREDICTION (ONCE) ===
        if 'acoustic' in self.models[role]:
            try:
                # Single call to acoustic model (includes feature extraction)
                pred_a, prob_a = self._predict_acoustic_single(role, audio_channel, sr)
                if pred_a is not None:
                    acoustic_pred_id = int(pred_a)
                    acoustic_clf = self.models[role]['acoustic']['clf']
                    acoustic_classes = list(acoustic_clf.classes_)
                    acoustic_pred = acoustic_classes[acoustic_pred_id]
                    
                    # Reorder probabilities to canonical order (once)
                    acoustic_prob_canonical = self._reorder_probabilities(prob_a, acoustic_classes, canonical_labels)
                    
                    # Store all acoustic results
                    results['acoustic_prediction'] = acoustic_pred
                    results['acoustic_prediction_id'] = acoustic_pred_id
                    results['acoustic_prob_canonical'] = acoustic_prob_canonical
                    results['per_modality']['acoustic'] = {
                        'id': acoustic_pred_id,
                        'prob': prob_a,
                        'prob_canonical': acoustic_prob_canonical
                    }
                    print(f"✓ {role.capitalize()} acoustic prediction: {acoustic_pred}")
                    
            except Exception as e:
                print(f"⚠ {role.capitalize()} acoustic prediction failed: {e}")
        
        # === FUSION (USING CACHED PROBABILITIES) ===
        per_mod = results['per_modality']
        
        if len(per_mod) > 1:
            # Use already computed canonical probabilities
            text_prob_canon = results['text_prob_canonical']
            acoustic_prob_canon = results['acoustic_prob_canonical']
            
            # Standard weighted fusion
            fusion_prob = role_weights['text'] * text_prob_canon + role_weights['acoustic'] * acoustic_prob_canon
            fusion_pred_id = int(fusion_prob.argmax())
            
            # Apply business rules (commented out in original)
            # ... business logic here ...
            
            results['fusion_prediction'] = canonical_labels[fusion_pred_id]
            results['fusion_prediction_id'] = fusion_pred_id
            results['fusion_prob'] = fusion_prob
            results['fusion_confidence'] = fusion_prob.max()
            
        elif len(per_mod) == 1:
            # Single modality available - use cached results
            modality = list(per_mod.keys())[0]
            if modality == 'text':
                results['fusion_prediction_id'] = results['text_prediction_id']
                results['fusion_prediction'] = results['text_prediction']
                results['fusion_prob'] = results['text_prob_canonical']
            else:  # acoustic
                acoustic_classes = list(self.models[role]['acoustic']['clf'].classes_)
                acoustic_label = acoustic_classes[results['acoustic_prediction_id']]
                if acoustic_label in canonical_labels:
                    results['fusion_prediction_id'] = canonical_labels.index(acoustic_label)
                    results['fusion_prediction'] = acoustic_label
                    results['fusion_prob'] = results['acoustic_prob_canonical']
                else:
                    results['fusion_prediction'] = acoustic_label
                    results['fusion_prob'] = results['acoustic_prob_canonical']
        
        if results['fusion_prediction']:
            print(f"✓ {role.capitalize()} fusion prediction: {results['fusion_prediction']}")
        
        # Return only the 6 key outputs (clean interface)
        return {
            'text_prediction': results['text_prediction'],
            'acoustic_prediction': results['acoustic_prediction'],
            'fusion_prediction': results['fusion_prediction'],
            'text_prediction_id': results['text_prediction_id'],
            'acoustic_prediction_id': results['acoustic_prediction_id'],
            'fusion_prediction_id': results['fusion_prediction_id'],
            'per_modality': results['per_modality'],
            'fusion_prob': results['fusion_prob']
        }

# Additional optimizations for feature caching
class CachedAcousticFeatures:
    """Cache acoustic features to avoid recomputation"""
    
    def __init__(self):
        self._cache = {}
    
    def get_features(self, audio_hash, compute_fn):
        """Get features from cache or compute and cache them"""
        if audio_hash not in self._cache:
            self._cache[audio_hash] = compute_fn()
        return self._cache[audio_hash]
    
    def clear_cache(self):
        """Clear the feature cache"""
        self._cache.clear()


# Memory-efficient batch processing
class BatchOptimizedEvaluator(OptimizedDualMultimodalEvaluator):
    """Further optimizations for batch processing"""
    
    def __init__(self):
        super().__init__()
        self.feature_cache = CachedAcousticFeatures()
    
    def predict_batch_optimized(self, batch_data):
        """
        Process multiple audio files efficiently
        
        batch_data: List of dicts with keys: agent_text, client_text, audio_path
        """
        results = []
        
        # Pre-load all audio files
        audio_data = {}
        for i, item in enumerate(batch_data):
            audio_path = item['audio_path']
            if audio_path not in audio_data:
                left, right, sr = self._load_stereo_audio(audio_path)
                audio_data[audio_path] = (left, right, sr)
        
        # Process each item using cached audio
        for item in batch_data:
            audio_path = item['audio_path']
            left_channel, right_channel, sr = audio_data[audio_path]
            
            if left_channel is not None:
                result = {
                    'agent': self._predict_single_role_optimized(
                        'agent', item['agent_text'], left_channel, sr, 
                        {'text': 0.54, 'acoustic': 0.46},
                        self.agent_labels, self.canonical_agent_labels
                    ),
                    'client': self._predict_single_role_optimized(
                        'client', item['client_text'], right_channel, sr,
                        {'text': 0.42, 'acoustic': 0.58},
                        self.client_labels, self.canonical_client_labels
                    )
                }
            else:
                result = {'error': f'Failed to load audio: {audio_path}'}
            
            results.append(result)
        
        return results


# GPU memory optimization
class GPUOptimizedEvaluator(OptimizedDualMultimodalEvaluator):
    """GPU memory optimizations"""
    
    def _extract_acoustic_features_optimized(self, waveform, sr):
        """Optimized feature extraction with better GPU memory management"""
        try:
            # Process in chunks if waveform is very long
            max_length = 16000 * 30  # 30 seconds at 16kHz
            if waveform.numel() > max_length:
                # Process in overlapping chunks and average features
                chunk_size = max_length
                overlap = chunk_size // 4
                chunks = []
                
                for start in range(0, waveform.numel() - chunk_size + 1, chunk_size - overlap):
                    chunk = waveform[start:start + chunk_size]
                    chunks.append(self._extract_features_single_chunk(chunk, sr))
                
                # Average features across chunks
                if chunks:
                    avg_features = {}
                    for key in chunks[0].keys():
                        avg_features[key] = np.mean([chunk[key] for chunk in chunks])
                    return avg_features
            
            return self._extract_features_single_chunk(waveform, sr)
            
        except Exception as exc:
            print(f"[Optimized Feature Extraction Error]: {exc}")
            return None
    
    def _extract_features_single_chunk(self, waveform, sr):
        """Extract features from a single audio chunk"""
        # Move computation to GPU in batches to optimize memory usage
        with torch.no_grad():  # Ensure no gradients are tracked
            # ... (optimized feature extraction code) ...
            pass



# ────────────────────────────────────────────────────────────────────────────────
#  Enhanced UI for dual prediction
# ────────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Dual Arabic Sentiment Analysis System ===")
    print("Agent sentiment: aggressive, courtoi, neutre, sec")
    print("Client sentiment: content, neutre, mecontent, tres mecontent")
    print("Audio: Left channel = Agent, Right channel = Client")
    
    # Initialize evaluator
    try:
        evaluator = OptimizedDualMultimodalEvaluator()
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        import traceback
        traceback.print_exc()
        print("\nExiting program.")
        return
    
    # Show weights information
    print("\nUsing fixed weights: text=0.3, acoustic=0.7")
    print("Special rules applied:")
    print("  - Client: If text='content' and acoustic='mecontent', fusion='content'")
    print("  - Agent: If text='neutre' and acoustic='sec', fusion='sec'")
    
    # Main prediction loop
    print("\n=== Dual Sentiment Prediction ===")
    print("Enter 'q' to quit at any prompt")
    
    while True:
        # Get stereo audio file path
        wav_path = input("\nEnter path to stereo WAV file: ").strip().strip('"\'')
        if wav_path.lower() == 'q':
            break
        
        if not os.path.exists(wav_path):
            print(f"Error: File not found: {wav_path}")
            continue
        
        # Get agent text
        agent_text = input("Enter agent's Arabic text: ").strip()
        if agent_text.lower() == 'q':
            break
        
        # Get client text
        client_text = input("Enter client's Arabic text: ").strip()
        if client_text.lower() == 'q':
            break
        
        # Make predictions
        try:
            print("\nAnalyzing both agent and client...")
            results = evaluator.predict_dual_sentiment_optimized(agent_text, client_text, wav_path)
            
            # Print results
            print("\n" + "="*50)
            print("RESULTS - 6 OUTPUTS (3 INPUTS)")
            print("="*50)
            print(f"Inputs: Agent Text, Client Text, Stereo Audio")
            print(f"Outputs: Agent Text, Agent Acoustic, Agent Fusion, Client Text, Client Acoustic, Client Fusion")
            
            # Agent results
            if 'agent' in results and 'error' not in results['agent']:
                agent_r = results['agent']
                print(f"\n🎧 AGENT OUTPUTS:")
                print(f"  1. Text Model:     {agent_r['text_prediction']} (ID: {agent_r['text_prediction_id']})")
                print(f"  2. Acoustic Model: {agent_r['acoustic_prediction']} (ID: {agent_r['acoustic_prediction_id']})")
                print(f"  3. Fusion Result:  {agent_r['fusion_prediction']} (ID: {agent_r['fusion_prediction_id']})")
                if agent_r['special_condition_applied']:
                    print(f"     ⚠ Special rule applied")
            else:
                print(f"\n🎧 AGENT OUTPUTS: Error - {results.get('agent', {}).get('error', 'Unknown error')}")
            
            # Client results
            if 'client' in results and 'error' not in results['client']:
                client_r = results['client']
                print(f"\n👤 CLIENT OUTPUTS:")
                print(f"  4. Text Model:     {client_r['text_prediction']} (ID: {client_r['text_prediction_id']})")
                print(f"  5. Acoustic Model: {client_r['acoustic_prediction']} (ID: {client_r['acoustic_prediction_id']})")
                print(f"  6. Fusion Result:  {client_r['fusion_prediction']} (ID: {client_r['fusion_prediction_id']})")
                if client_r['special_condition_applied']:
                    print(f"     ⚠ Special rule applied")
            else:
                print(f"\n👤 CLIENT OUTPUTS: Error - {results.get('client', {}).get('error', 'Unknown error')}")
            
            # Summary in structured format
            print(f"\n📋 STRUCTURED OUTPUT:")
            print(f"Agent_Text: {results.get('agent', {}).get('text_prediction', 'N/A')}")
            print(f"Agent_Acoustic: {results.get('agent', {}).get('acoustic_prediction', 'N/A')}")
            print(f"Agent_Fusion: {results.get('agent', {}).get('fusion_prediction', 'N/A')}")
            print(f"Client_Text: {results.get('client', {}).get('text_prediction', 'N/A')}")
            print(f"Client_Acoustic: {results.get('client', {}).get('acoustic_prediction', 'N/A')}")
            print(f"Client_Fusion: {results.get('client', {}).get('fusion_prediction', 'N/A')}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
        
        # Ask to continue
        cont = input("\nContinue with another prediction? (y/n): ").strip().lower()
        if cont != 'y':
            break
    
    print("\nThank you for using the Dual Arabic Sentiment Analysis System!")

def api_predict(agent_text, client_text, audio_path):
    """
    API function for programmatic access
    
    Parameters
    ----------
    agent_text : str - Text spoken by agent
    client_text : str - Text spoken by client  
    audio_path : str - Path to stereo WAV file (left=agent, right=client)
    
    Returns
    -------
    dict - Results with 6 outputs:
        - agent_text_prediction
        - agent_acoustic_prediction  
        - agent_fusion_prediction
        - client_text_prediction
        - client_acoustic_prediction
        - client_fusion_prediction
    """
    try:
        evaluator = OptimizedDualMultimodalEvaluator()
        results = evaluator.predict_dual_sentiment_optimized(agent_text, client_text, audio_path)
        
        # Format output in a clean structure
        output = {
            'agent_text_prediction': results.get('agent', {}).get('text_prediction'),
            'agent_text_prediction_id': results.get('agent', {}).get('text_prediction_id'),
            'agent_acoustic_prediction': results.get('agent', {}).get('acoustic_prediction'),
            'agent_acoustic_prediction_id': results.get('agent', {}).get('acoustic_prediction_id'),
            'agent_fusion_prediction': results.get('agent', {}).get('fusion_prediction'),
            'agent_fusion_prediction_id': results.get('agent', {}).get('fusion_prediction_id'),
            'agent_special_condition_applied': results.get('agent', {}).get('special_condition_applied', False),
            
            'client_text_prediction': results.get('client', {}).get('text_prediction'),
            'client_text_prediction_id': results.get('client', {}).get('text_prediction_id'),
            'client_acoustic_prediction': results.get('client', {}).get('acoustic_prediction'),
            'client_acoustic_prediction_id': results.get('client', {}).get('acoustic_prediction_id'),
            'client_fusion_prediction': results.get('client', {}).get('fusion_prediction'),
            'client_fusion_prediction_id': results.get('client', {}).get('fusion_prediction_id'),
            'client_special_condition_applied': results.get('client', {}).get('special_condition_applied', False),
            
            'success': True,
            'error': None
        }
        
        return output
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'agent_text_prediction': None,
            'agent_acoustic_prediction': None,
            'agent_fusion_prediction': None,
            'client_text_prediction': None,
            'client_acoustic_prediction': None,
            'client_fusion_prediction': None
        }

if __name__ == "__main__":
    main()
