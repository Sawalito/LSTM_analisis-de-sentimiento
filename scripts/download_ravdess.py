"""Descarga solo el audio de habla (Audio_Speech_Actors_01-24) del dataset
orvile/ravdess-dataset. El dataset completo pesa ~24 GB por los videos."""
from concurrent.futures import ThreadPoolExecutor

import os
import time

import kagglehub

HANDLE = "orvile/ravdess-dataset"
PREFIX = "Audio_Speech_Actors_01-24/"
CACHE = os.path.expanduser(f"~/.cache/kagglehub/datasets/{HANDLE}/versions/1/")


def list_files():
    """Nombres 03-01-EE-II-SS-RR-AA.wav: neutral (01) solo tiene intensidad normal."""
    names = []
    for actor in range(1, 25):
        for emo in range(1, 9):
            for inten in ([1] if emo == 1 else [1, 2]):
                for stmt in (1, 2):
                    for rep in (1, 2):
                        names.append(f"{PREFIX}Actor_{actor:02d}/03-01-{emo:02d}-{inten:02d}-{stmt:02d}-{rep:02d}-{actor:02d}.wav")
    return names


def fetch(name):
    if os.path.exists(CACHE + name):
        return CACHE + name
    for attempt in range(6):
        try:
            return kagglehub.dataset_download(HANDLE, path=name)
        except Exception as e:
            err = e
            time.sleep(2 ** attempt)
    raise err


if __name__ == "__main__":
    names = list_files()
    print(f"{len(names)} archivos")
    with ThreadPoolExecutor(4) as ex:
        paths = list(ex.map(fetch, names))
    print(paths[0].split(PREFIX)[0] + PREFIX)
