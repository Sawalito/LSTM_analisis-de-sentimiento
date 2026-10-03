"""Acceso a RAVDESS (habla) almacenado en Supabase.

- Metadatos: tabla public.ravdess_clips (lectura pública vía RLS).
- Audio: bucket público 'ravdess-audio', ruta Actor_XX/<archivo>.wav.

Los .wav se descargan bajo demanda a una caché local (por defecto
~/.cache/ravdess_supabase) que puede borrarse en cualquier momento.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
BUCKET = "ravdess-audio"
TABLE = "ravdess_clips"
CACHE_DIR = Path(os.environ.get("RAVDESS_CACHE", Path.home() / ".cache" / "ravdess_supabase"))

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
EMOTIONS_ES = {"neutral": "neutral", "calm": "calma", "happy": "alegría", "sad": "tristeza",
               "angry": "enojo", "fearful": "miedo", "disgust": "asco", "surprised": "sorpresa"}
STATEMENTS = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}


def parse_filename(filename):
    """03-01-EE-II-SS-RR-AA.wav -> diccionario de metadatos."""
    modality, channel, emo, inten, stmt, rep, actor = (int(p) for p in Path(filename).stem.split("-"))
    return {
        "filename": Path(filename).name,
        "storage_path": f"Actor_{actor:02d}/{Path(filename).name}",
        "actor": actor,
        "gender": "male" if actor % 2 else "female",
        "emotion_code": emo,
        "emotion": EMOTIONS[emo],
        "intensity": "normal" if inten == 1 else "strong",
        "statement": STATEMENTS[stmt],
        "repetition": rep,
    }


def public_url(storage_path):
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"


def load_metadata():
    """Lee toda la tabla de metadatos vía PostgREST (paginado)."""
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    rows, start, step = [], 0, 1000
    while True:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{TABLE}", params={"select": "*", "order": "filename"},
                         headers={**headers, "Range": f"{start}-{start + step - 1}"}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        rows += batch
        if len(batch) < step:
            break
        start += step
    df = pd.DataFrame(rows)
    df["emotion_es"] = df["emotion"].map(EMOTIONS_ES)
    return df


def _fetch(storage_path):
    dest = CACHE_DIR / storage_path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                r = requests.get(public_url(storage_path), timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.rename(dest)
    return str(dest)


def ensure_audio(df, workers=8):
    """Descarga (si hace falta) los audios de df y agrega la columna 'path'."""
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(_fetch, df["storage_path"]))
    return df.assign(path=paths)
