"""Genera notebooks/02_modelo_kaggle.ipynb: versión autocontenida de 02_modelo
para ejecutar en Kaggle (GPU). Incrusta el código de src/ y lee los datos de
Supabase; si el dataset orvile/ravdess-dataset está añadido como input, lo usa
directamente y no necesita internet."""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_model_notebook as base  # noqa: E402


def src(name):
    return (ROOT / "src" / name).read_text()


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = list(base.cells)

intro = cells[0].source + """

---
### Cómo ejecutarlo en Kaggle
1. *File → Import Notebook* y sube este archivo.
2. *Settings → Accelerator*: **GPU T4 x2** o **P100**.
3. Fuente de datos, elige una:
   - **Supabase**: *Settings → Internet* **On** y en *Add-ons → Secrets* crea `SUPABASE_URL` y `SUPABASE_KEY`.
   - **Input de Kaggle**: *Add Input* → `orvile/ravdess-dataset`. Se detecta solo y no necesita internet.
4. *Run All*. Los resultados quedan en `/kaggle/working/` (pestaña *Output*).

Con GPU se usa el **ancho original de la referencia** (512-512-256-256-128); en CPU, 1/4 de ese ancho.
"""
cells[0] = md(intro)

setup = '''
import os, sys, json, time, copy, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import requests
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if KAGGLE else Path.cwd()
OUT = WORK / "resultados_modelo"; OUT.mkdir(exist_ok=True)
MODELS = WORK / "models"; MODELS.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH = (512, 512, 256, 256, 128) if DEVICE.type == "cuda" else (128, 128, 64, 64, 32)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
print("Dispositivo:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", "| ancho:", WIDTH)
mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
'''
cells[1] = code(setup.strip())

# Código de src/ incrustado. Las credenciales NO se incrustan: se leen de Kaggle Secrets
# (Add-ons → Secrets: SUPABASE_URL y SUPABASE_KEY) o de variables de entorno.
data_src = src("ravdess_data.py")
data_src = data_src.replace("from dotenv import load_dotenv\n", "").replace(
    'load_dotenv(Path(__file__).resolve().parents[1] / ".env")\n\n', '''try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    for _name in ("SUPABASE_URL", "SUPABASE_KEY"):
        os.environ.setdefault(_name, _secrets.get_secret(_name))
except Exception:
    pass  # fuera de Kaggle o sin secretos: se usan variables de entorno

''')
data_src = data_src.replace('SUPABASE_URL = os.environ["SUPABASE_URL"]', 'SUPABASE_URL = os.environ.get("SUPABASE_URL")')
data_src = data_src.replace('SUPABASE_KEY = os.environ["SUPABASE_KEY"]', 'SUPABASE_KEY = os.environ.get("SUPABASE_KEY")')
data_src = data_src.replace('Path.home() / ".cache" / "ravdess_supabase"', 'Path("/tmp/ravdess_supabase")')
model_src = src("ser_model.py")
embedded = [
    md("## 0. Código del proyecto (incrustado de `src/`)"),
    code("# --- src/ravdess_data.py ---\n" + data_src),
    code("# --- src/ser_model.py ---\n" + model_src),
]

data_cell = '''
ORDER = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
ORDER_ES = [EMOTIONS_ES[e] for e in ORDER]

KAGGLE_INPUT = next(Path("/kaggle/input").rglob("Audio_Speech_Actors_01-24"), None) if Path("/kaggle/input").exists() else None
if KAGGLE_INPUT is not None:
    # Input de Kaggle: metadatos a partir de los nombres de archivo (mismo esquema que la tabla de Supabase)
    files = sorted(KAGGLE_INPUT.glob("Actor_*/*.wav"))
    df = pd.DataFrame([{**parse_filename(f), "path": str(f)} for f in files])
    print("Fuente: input de Kaggle")
else:
    assert SUPABASE_URL and SUPABASE_KEY, "Configura los secretos SUPABASE_URL y SUPABASE_KEY o añade el input orvile/ravdess-dataset"
    df = ensure_audio(load_metadata(), workers=16)
    print("Fuente: Supabase")
assert len(df) == 1440, len(df)
df = df.sort_values("filename").reset_index(drop=True)
df["y"] = df["emotion"].map({e: i for i, e in enumerate(ORDER)})
with ProcessPoolExecutor() as ex:
    variants = list(ex.map(clip_variants, [(p, True, SEED + i) for i, p in enumerate(df["path"])], chunksize=8))
V = np.stack([np.stack(v) for v in variants])          # (1440, 3, 2376): original, ruido, pitch+ruido
print(V.shape)
'''
cells[3] = code(data_cell.strip())

# Entrenamiento en GPU
train = cells[5].source
train = train.replace("torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).numpy()",
                      "torch.cat([model(X[i:i + bs].to(DEVICE)).argmax(1).cpu() for i in range(0, len(X), bs)]).numpy()")
train = train.replace("model = ConvSER(tr[0].shape[1], len(ORDER), width=(128, 128, 64, 64, 32))",
                      "model = ConvSER(tr[0].shape[1], len(ORDER), width=WIDTH).to(DEVICE)")
train = train.replace("model.train(); perm = torch.randperm(len(tr[0]));",
                      "Xg, yg = tr[0].to(DEVICE), tr[1].to(DEVICE)\n        model.train(); perm = torch.randperm(len(tr[0]), device=DEVICE);")
train = train.replace("loss = nn.functional.cross_entropy(model(tr[0][b]), tr[1][b])",
                      "loss = nn.functional.cross_entropy(model(Xg[b]), yg[b])")
assert train.count("DEVICE") == 5, train.count("DEVICE")
cells[5] = code(train)

# orden final: intro, imports, código incrustado, resto
cells = cells[:2] + embedded + cells[2:]

cells.append(md("## 4. Descargar resultados\nEn Kaggle, todo lo que está en `/kaggle/working` aparece en la pestaña *Output*."))
cells.append(code('''
import shutil
shutil.make_archive(str(WORK / "resultados_modelo_kaggle"), "zip", WORK, "resultados_modelo")
print(sorted(p.name for p in OUT.iterdir()), sorted(p.name for p in MODELS.iterdir()))
'''.strip()))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True,
               "dataSources": [{"sourceType": "datasetVersion", "datasetSlug": "ravdess-dataset", "ownerSlug": "orvile"}]},
})
nbf.write(nb, ROOT / "notebooks" / "02_modelo_kaggle.ipynb")
print("ok", len(cells), "celdas")
