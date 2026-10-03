"""Pipeline de clasificación de emociones basado en el notebook de referencia
"Speech Emotion Recognition 90%" (gemmin, Kaggle):

- 2.5 s de audio desde el segundo 0.6, a 22 050 Hz.
- Vector por clip = [ZCR | RMS | MFCC(20)] cuadro a cuadro, aplanado (2376 valores).
- Aumentación: original, + ruido, + pitch-shift con ruido (3 versiones por clip).
- Red Conv1D de 5 bloques (Conv-BN-MaxPool) + Dense 512 + softmax.

Diferencia deliberada con la referencia: la aumentación se aplica *después* de
dividir los datos y solo al entrenamiento, y la división es por actor, para que
el conjunto de prueba contenga hablantes nunca vistos.
"""
import warnings

import librosa
import numpy as np
import torch
from torch import nn

SR = 22050
DURATION = 2.5
OFFSET = 0.6
FRAME, HOP = 2048, 512
N_SAMPLES = int(SR * DURATION)


def load_clip(path):
    warnings.filterwarnings("ignore")
    y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION, offset=OFFSET)
    return librosa.util.fix_length(y, size=N_SAMPLES)


def noise(y, rng, rate=0.035):
    return y + rate * rng.uniform() * np.max(np.abs(y)) * rng.normal(size=y.shape)


def pitch(y, steps=0.7):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)


def frame_features(y):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=FRAME, hop_length=HOP).T.ravel()
    return np.concatenate([zcr, rms, mfcc]).astype(np.float32)


def clip_variants(args):
    """(path, augment, seed) -> lista de vectores: original [+ ruido, + pitch con ruido]."""
    path, augment, seed = args
    y = load_clip(path)
    out = [frame_features(y)]
    if augment:
        rng = np.random.default_rng(seed)
        out.append(frame_features(noise(y, rng)))
        out.append(frame_features(noise(pitch(y), rng)))
    return out


class ConvSER(nn.Module):
    """Equivalente en PyTorch de la red Conv1D de la referencia (Keras)."""

    def __init__(self, n_features, n_classes, width=(512, 512, 256, 256, 128)):
        super().__init__()

        def block(cin, cout, k, pool_k, drop):
            layers = [nn.Conv1d(cin, cout, k, padding="same"), nn.ReLU(), nn.BatchNorm1d(cout),
                      nn.MaxPool1d(pool_k, stride=2, padding=pool_k // 2)]
            return layers + ([nn.Dropout(drop)] if drop else [])

        c1, c2, c3, c4, c5 = width
        self.features = nn.Sequential(
            *block(1, c1, 5, 5, 0), *block(c1, c2, 5, 5, 0.2), *block(c2, c3, 5, 5, 0),
            *block(c3, c4, 3, 5, 0.2), *block(c4, c5, 3, 3, 0.2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, n_features)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 512), nn.ReLU(), nn.BatchNorm1d(512),
                                  nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))
