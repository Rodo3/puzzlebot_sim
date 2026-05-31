"""
voice_inference.py — Motor de inferencia de voz sin dependencias de ROS ni audio.

Uso:
    engine = VoiceInferenceEngine.load('artifacts_final')
    result = engine.infer(pcm_float32_array)
    # result['command'], result['confidence'], result['ranked_predictions'], result['inference_time_ms']
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class InferenceResult:
    command: str
    confidence: float          # margen KMeans (diferencia con 2do lugar, positivo = más seguro)
    inference_time_ms: float
    ranked_kmeans: List[Tuple[str, float]]   # [(label, dist), ...] menor dist = mejor
    ranked_hmm: List[Tuple[str, float]]      # [(label, log_lik), ...] mayor = mejor


class VoiceInferenceEngine:
    """Carga modelos KMeans + HMM y ejecuta inferencia sobre PCM float32."""

    def __init__(self, km_model, km_cfg, hmm_model, hmm_cfg):
        self._km_model = km_model
        self._km_cfg = km_cfg
        self._hmm_model = hmm_model
        self._hmm_cfg = hmm_cfg

    @classmethod
    def load(cls, artifact_dir) -> 'VoiceInferenceEngine':
        """Carga ambos modelos desde artifact_dir.

        Raises:
            FileNotFoundError: si algún archivo de modelo no existe.
        """
        artifact_dir = Path(artifact_dir)

        from .models.kmeans_codebook import KMeansCodebookClassifier
        from .models.hmm import HiddenMarkovModelClassifier
        from .config import MFCCConfig

        # --- KMeans ---
        km_path = artifact_dir / 'kmeans_model.pkl'
        km_cfg_path = artifact_dir / 'kmeans_feature_config.json'
        if not km_path.exists():
            raise FileNotFoundError(f"Modelo KMeans no encontrado: {km_path}")
        if not km_cfg_path.exists():
            raise FileNotFoundError(f"Config KMeans no encontrada: {km_cfg_path}")

        km_cfg_data = json.loads(km_cfg_path.read_text())
        km_cfg = MFCCConfig(
            sample_rate=km_cfg_data.get('sample_rate', 16000),
            pre_emphasis=km_cfg_data.get('pre_emphasis', 0.97),
            frame_size=km_cfg_data.get('frame_size', 0.025),
            frame_stride=km_cfg_data.get('frame_stride', 0.01),
            n_fft=km_cfg_data.get('n_fft', 512),
            n_filters=km_cfg_data.get('n_filters', 26),
            n_mfcc=km_cfg_data.get('n_mfcc', 13),
            include_delta=km_cfg_data.get('include_delta', False),
            include_delta_delta=km_cfg_data.get('include_delta_delta', False),
            cmvn=km_cfg_data.get('cmvn', False),
        )
        km_model = KMeansCodebookClassifier.load(km_path)

        # --- HMM ---
        hmm_path = artifact_dir / 'hmm_model.pkl'
        hmm_cfg_path = artifact_dir / 'hmm_config.json'
        if not hmm_path.exists():
            raise FileNotFoundError(f"Modelo HMM no encontrado: {hmm_path}")
        if not hmm_cfg_path.exists():
            raise FileNotFoundError(f"Config HMM no encontrada: {hmm_cfg_path}")

        hmm_cfg_data = json.loads(hmm_cfg_path.read_text())
        hmm_mfcc = hmm_cfg_data.get('mfcc', {})
        hmm_cfg = MFCCConfig(
            sample_rate=hmm_mfcc.get('sample_rate', 16000),
            n_mfcc=hmm_mfcc.get('n_mfcc', 13),
            n_filters=hmm_mfcc.get('n_filters', 26),
            cmvn=hmm_mfcc.get('cmvn', False),
            include_delta=hmm_mfcc.get('include_delta', False),
            include_delta_delta=hmm_mfcc.get('include_delta_delta', False),
            use_librosa=hmm_mfcc.get('use_librosa', False),
            include_zcr=hmm_mfcc.get('include_zcr', False),
            include_rms=hmm_mfcc.get('include_rms', False),
            include_contrast=hmm_mfcc.get('include_contrast', False),
        )
        hmm_model = HiddenMarkovModelClassifier.load(hmm_path)

        return cls(km_model, km_cfg, hmm_model, hmm_cfg)

    def infer(self, signal: np.ndarray, sample_rate: int = 16000) -> InferenceResult:
        """Ejecuta inferencia sobre una señal PCM float32.

        Args:
            signal:      array float32 mono, ya en el rango [-1, 1].
            sample_rate: debe coincidir con el SR de entrenamiento (16000).

        Returns:
            InferenceResult con comando, confianza y predicciones rankeadas.
        """
        from .audio_io import normalize, _resample
        from .mfcc import extract_mfcc_frames
        from .librosa_features import extract_librosa_frames

        signal = signal.flatten().astype(np.float32)
        if sample_rate != 16000:
            signal = _resample(signal, sample_rate, 16000)
        signal = normalize(signal)

        t0 = time.perf_counter()

        # KMeans (más preciso — decide el comando final)
        frames_km = extract_mfcc_frames(signal, self._km_cfg)
        ranked_km = self._km_model.predict_ranked(frames_km)

        # HMM
        if self._hmm_cfg.use_librosa:
            frames_hmm = extract_librosa_frames(signal, self._hmm_cfg)
        else:
            frames_hmm = extract_mfcc_frames(signal, self._hmm_cfg)
        ranked_hmm = self._hmm_model.predict_ranked(frames_hmm)

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        label = ranked_km[0][0]
        confidence = float(ranked_km[1][1] - ranked_km[0][1]) if len(ranked_km) > 1 else 0.0

        return InferenceResult(
            command=label,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            ranked_kmeans=[(lbl, float(dist)) for lbl, dist in ranked_km],
            ranked_hmm=[(lbl, float(ll)) for lbl, ll in ranked_hmm],
        )
