"""Descriptores acústicos por clip para el EDA."""
import warnings

import librosa
import numpy as np

SR = 22050
TOP_DB = 30  # umbral para recortar silencio
N_MFCC = 20
FRAME_LENGTH = 2048  # ~93 ms a 22 050 Hz
HOP_LENGTH = 512     # ~23 ms, 75 % de traslape
ENV_SECONDS = 4.0    # duración fija para comparar envolventes


def clip_descriptors(path):
    warnings.filterwarnings("ignore")
    y, sr = librosa.load(path, sr=SR, mono=True)
    _, (start, end) = librosa.effects.trim(y, top_db=TOP_DB)
    voiced = y[start:end]

    fh = dict(frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=voiced, **fh)[0]
    zcr = librosa.feature.zero_crossing_rate(voiced, **fh)[0]
    centroid = librosa.feature.spectral_centroid(y=voiced, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=voiced, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    f0, voiced_flag, _ = librosa.pyin(voiced, fmin=65, fmax=600, sr=sr, frame_length=2048)
    f0 = f0[voiced_flag] if voiced_flag.any() else np.array([np.nan])
    mfcc = librosa.feature.mfcc(y=voiced, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH).mean(axis=1)

    out = {
        "dur_total": len(y) / sr,
        "dur_voz": len(voiced) / sr,
        "sil_inicio": start / sr,
        "sil_final": (len(y) - end) / sr,
        "rms_media": rms.mean(),
        "rms_std": rms.std(),
        "zcr_media": zcr.mean(),
        "centroide_media": centroid.mean(),
        "rolloff_media": rolloff.mean(),
        "f0_media": np.nanmean(f0),
        "f0_std": np.nanstd(f0),
        "f0_rango": np.nanpercentile(f0, 95) - np.nanpercentile(f0, 5),
        "frac_sonora": voiced_flag.mean(),
    }
    out.update({f"mfcc_{i + 1}": v for i, v in enumerate(mfcc)})
    return out


def rms_envelope(path):
    """Envolvente RMS cuadro a cuadro de los primeros ENV_SECONDS del clip (sin recortar)."""
    warnings.filterwarnings("ignore")
    y, _ = librosa.load(path, sr=SR, mono=True, duration=ENV_SECONDS)
    y = librosa.util.fix_length(y, size=int(SR * ENV_SECONDS))
    return librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
