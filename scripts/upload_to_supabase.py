"""Sube los .wav de RAVDESS (habla) al bucket 'ravdess-audio' y sus metadatos
a la tabla public.ravdess_clips. Requiere las políticas temporales de carga."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ravdess_data import BUCKET, SUPABASE_KEY, SUPABASE_URL, TABLE, parse_filename  # noqa: E402

SRC = Path.home() / ".cache/kagglehub/datasets/orvile/ravdess-dataset/versions/1/Audio_Speech_Actors_01-24"

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload(path):
    meta = parse_filename(path)
    info = sf.info(path)
    meta.update(size_bytes=path.stat().st_size, sample_rate=info.samplerate,
                channels=info.channels, duration_s=round(info.duration, 4))
    for attempt in range(3):
        try:
            client.storage.from_(BUCKET).upload(meta["storage_path"], path.read_bytes(),
                                                {"content-type": "audio/wav", "upsert": "true"})
            break
        except Exception:
            if attempt == 2:
                raise
    return meta


if __name__ == "__main__":
    files = sorted(SRC.glob("Actor_*/*.wav"))
    print(f"{len(files)} archivos")
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(upload, files))
    for i in range(0, len(rows), 500):
        client.table(TABLE).upsert(rows[i:i + 500]).execute()
    print(f"subidos {len(rows)} audios y {len(rows)} filas")
