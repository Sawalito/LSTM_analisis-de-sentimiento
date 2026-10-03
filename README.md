# Reconocimiento de emociones en la voz — RAVDESS

EDA y modelo de clasificación de emociones sobre el subconjunto de **habla** de RAVDESS
(1440 clips, 24 actores, 8 emociones), con los datos alojados en **Supabase**.

**Autores:** Saúl Ubaldo Rojas Vázquez, Karols Cisneros y Mónica.

## Documentos

- [Reporte del análisis exploratorio (PDF)](latex/reporte.pdf)
- [Presentación (PDF)](latex/presentacion.pdf)
- Fuentes LaTeX en `latex/` y paquete listo para Overleaf en `overleaf_ravdess_eda.zip`

## Datos en Supabase (proyecto `ser-voz-emocional`)

| Recurso | Contenido |
|---|---|
| Bucket `ravdess-audio` (público, solo lectura) | 1440 WAV en `Actor_XX/<archivo>.wav` |
| Tabla `public.ravdess_clips` (RLS, solo lectura) | Metadatos por clip: actor, género, emoción, intensidad, frase, repetición, duración… |

Desde Python:

```python
import sys; sys.path.insert(0, "src")
from ravdess_data import load_metadata, ensure_audio
df = load_metadata()        # tabla completa desde Supabase
df = ensure_audio(df)       # descarga los WAV a ~/.cache/ravdess_supabase (borrable)
```

Las credenciales no están en el repositorio. Pídelas a los autores y defínelas como variables de entorno
`SUPABASE_URL` y `SUPABASE_KEY` (o en un archivo local `.env`, que está ignorado por git).

## Estructura

```
src/ravdess_data.py            acceso a Supabase (metadatos + audio bajo demanda)
src/features.py                descriptores acústicos para el EDA
src/ser_model.py               features, aumentación y red Conv1D (estilo notebook de referencia)
notebooks/01_eda.ipynb         análisis exploratorio (genera figuras y cifras para LaTeX)
notebooks/02_modelo.ipynb      entrenamiento y evaluación del modelo (local, CPU)
notebooks/02_modelo_kaggle.ipynb  misma versión, autocontenida para Kaggle con GPU
scripts/download_ravdess.py    descarga solo el audio de habla desde Kaggle (el dataset completo pesa ~24 GB)
scripts/upload_to_supabase.py  carga inicial a Supabase (ya ejecutada)
scripts/build_*_notebook.py    generan los notebooks (build_kaggle_notebook.py incrusta src/)
latex/                         reporte.tex y presentacion.tex (Overleaf)
overleaf_ravdess_eda.zip       el proyecto LaTeX listo para subir a Overleaf
resultados_modelo/             métricas y matrices de confusión del modelo
models/                        pesos entrenados (PyTorch)
data/                          descriptores acústicos por clip (CSV) y cifras del EDA (JSON)
```

## Reproducir

```bash
uv venv .venv && uv pip install -r requirements.txt   # torch: --index-url https://download.pytorch.org/whl/cpu
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb     # actualiza latex/figures y latex/resultados_eda.tex
jupyter nbconvert --to notebook --execute --inplace 02_modelo.ipynb
```

## Resultados preliminares del modelo (CPU, red a 1/4 de ancho)

| Protocolo | Accuracy | F1 macro |
|---|---|---|
| A — actores no vistos (21–24 en prueba) | 34.2 % | 0.30 |
| B — división aleatoria por clip | 63.9 % | 0.64 |

Pendiente: entrenamiento en Kaggle con GPU y el ancho original de la referencia.

## Entrenar en Kaggle

1. *File → Import Notebook* → `notebooks/02_modelo_kaggle.ipynb`.
2. *Settings → Accelerator*: GPU (T4 o P100).
3. Datos: *Add Input* → `orvile/ravdess-dataset` (sin internet), **o** *Settings → Internet* **On** y los secretos
   `SUPABASE_URL` y `SUPABASE_KEY` en *Add-ons → Secrets* para leer de Supabase.
4. *Run All*. Resultados en la pestaña *Output*: `resultados_modelo/`, `models/` y `resultados_modelo_kaggle.zip`.

Con GPU usa el ancho original de la referencia (512-512-256-256-128).

## Overleaf

Subir `overleaf_ravdess_eda.zip` (New Project → Upload Project). Compilador: pdfLaTeX.
Ambos documentos quedan en el mismo proyecto (`reporte.tex` y `presentacion.tex` en la raíz);
se alterna entre ellos con *Menu → Main document*. Las cifras del texto vienen de
`latex/resultados_eda.tex`, generado por el notebook de EDA.

## Licencia de los datos

RAVDESS — Livingstone & Russo (2018), PLoS ONE 13(5): e0196391. CC BY-NC-SA 4.0 (uso no comercial).
# LSTM_analisis-de-sentimiento
