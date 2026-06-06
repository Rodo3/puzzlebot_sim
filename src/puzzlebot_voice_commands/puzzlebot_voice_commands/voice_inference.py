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
    confidence: float          # margen log-lik entre 1er y 2do lugar (positivo = más seguro)
    inference_time_ms: float
    ranked_kmeans: List[Tuple[str, float]]   # siempre vacío (KMeans eliminado)
    ranked_hmm: List[Tuple[str, float]]      # [(label, log_lik), ...] mayor = mejor


class VoiceInferenceEngine:
    """Motor de inferencia HMM sobre PCM float32."""

    def __init__(self, hmm_model, hmm_cfg):
        self._hmm_model = hmm_model
        self._hmm_cfg = hmm_cfg

    @classmethod
    def load(cls, artifact_dir) -> 'VoiceInferenceEngine':
        """Carga el modelo HMM desde artifact_dir.

        Raises:
            FileNotFoundError: si los archivos de HMM no existen.
        """
        artifact_dir = Path(artifact_dir)

        from .models.hmm import HiddenMarkovModelClassifier
        from .config import MFCCConfig

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

        return cls(hmm_model, hmm_cfg)

    def infer(self, signal: np.ndarray, sample_rate: int = 16000) -> InferenceResult:
        """Ejecuta inferencia sobre una señal PCM float32.

        Args:
            signal:      array float32 mono, ya en el rango [-1, 1].
            sample_rate: debe coincidir con el SR de entrenamiento (16000).

        Returns:
            InferenceResult con comando, confianza y predicciones HMM rankeadas.
        """
        from .audio_io import normalize, _resample
        from .mfcc import extract_mfcc_frames
        from .librosa_features import extract_librosa_frames

        signal = signal.flatten().astype(np.float32)
        if sample_rate != 16000:
            signal = _resample(signal, sample_rate, 16000)
        signal = normalize(signal)

        t0 = time.perf_counter()

        if self._hmm_cfg.use_librosa:
            frames = extract_librosa_frames(signal, self._hmm_cfg)
        else:
            frames = extract_mfcc_frames(signal, self._hmm_cfg)
        ranked_hmm = self._hmm_model.predict_ranked(frames)

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        label = ranked_hmm[0][0]
        confidence = float(ranked_hmm[0][1] - ranked_hmm[1][1]) if len(ranked_hmm) > 1 else 0.0

        return InferenceResult(
            command=label,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            ranked_kmeans=[],
            ranked_hmm=[(lbl, float(ll)) for lbl, ll in ranked_hmm],
        )
