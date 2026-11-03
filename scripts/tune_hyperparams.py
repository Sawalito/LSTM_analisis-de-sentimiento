"""Búsqueda de hiperparámetros con validación cruzada anidada por actor.

Datos: Supabase (tabla ravdess_clips + bucket ravdess-audio), vía src/ravdess_data.py.

Protocolo (independiente del hablante):
- Externo: 6 folds de 4 actores consecutivos (1-4, 5-8, ..., 21-24). Cada actor cae en prueba una vez.
- Interno: la selección de hiperparámetros usa SOLO los 20 actores de entrenamiento de cada fold
  (GroupKFold por actor). Los 4 actores de prueba se usan una sola vez, para medir.

Etapas:
  1. Extracción de características (se cachea en data/cache/, ignorado por git):
     - "plano": vector cuadro a cuadro de la referencia (ZCR | RMS | MFCC20), 3 variantes por clip.
     - "resumen": estadísticos por clip (media/desv. de MFCC40, deltas, RMS, ZCR, contraste
       espectral, croma, centroide, rolloff, F0) — compacto y robusto entre hablantes.
  2. Modelos clásicos (rápidos): LogReg y SVM-RBF con GridSearchCV interno.
  3. CNN (opcional, --cnn N): N configuraciones aleatorias; en cada fold externo se elige la
     mejor por validación interna (4 actores del entrenamiento) y se mide en prueba.

Uso:
  .venv/bin/python scripts/tune_hyperparams.py               # etapas 1-2
  .venv/bin/python scripts/tune_hyperparams.py --cnn 6       # + búsqueda de la CNN
Resultados en resultados_tuning/ (no sobrescribe resultados_modelo/ ni models/).
"""
import argparse
import copy
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "resultados_tuning"
SEED = 42
WORKERS = 6
ORDER = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


# ---------------------------------------------------------------- características
def extract(df):
    from ser_model import clip_variants, summary_features
    CACHE.mkdir(parents=True, exist_ok=True)
    def cached(name, compute):
        f = CACHE / f"{name}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if list(z["filename"]) == list(df["filename"]):
                print("características desde caché:", f)
                return z["X"]
        t0 = time.time()
        # pocos procesos: con 16 workers el equipo se queda sin RAM (BrokenProcessPool)
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            X = compute(ex)
        np.savez_compressed(f, X=X, filename=df["filename"].values)
        print(f"{name}: {X.shape} en {time.time() - t0:.0f}s", flush=True)
        return X

    V = cached("plano", lambda ex: np.stack([np.stack(v) for v in ex.map(
        clip_variants, [(p, True, SEED + i) for i, p in enumerate(df["path"])], chunksize=8)]))
    S = cached("resumen", lambda ex: np.stack(list(ex.map(summary_features, df["path"], chunksize=8))))
    return V, S


def outer_folds(actor):
    return [np.isin(actor, range(4 * k + 1, 4 * k + 5)) for k in range(6)]


# ---------------------------------------------------------------- modelos clásicos
def classical(S, V, y, actor):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA

    candidates = {
        # baseline estilo E3: LogReg sobre el vector plano original (sin aumentación)
        "logreg_plano": (V[:, 0], make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                         {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1]}),
        "logreg_resumen": (S, make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                           {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1]}),
        "svm_resumen": (S, make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                        {"svc__C": [1, 3, 10, 30], "svc__gamma": [1e-4, 3e-4, 1e-3, 3e-3]}),
        "svm_pca_plano": (V[:, 0], make_pipeline(StandardScaler(), PCA(200, random_state=SEED), SVC(kernel="rbf")),
                          {"svc__C": [1, 10, 30], "svc__gamma": [3e-4, 1e-3, "scale"]}),
    }
    rows = []
    for name, (X, pipe, grid) in candidates.items():
        t0 = time.time()
        for k, te in enumerate(outer_folds(actor)):
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(5), scoring="accuracy", n_jobs=WORKERS)
            gs.fit(X[~te], y[~te], groups=actor[~te])
            pred = gs.predict(X[te])
            rows.append({"modelo": name, "fold": k + 1, "actores": f"{4*k+1}-{4*k+4}",
                         "accuracy": accuracy_score(y[te], pred), "f1_macro": f1_score(y[te], pred, average="macro"),
                         "inner_cv": gs.best_score_, "params": json.dumps(gs.best_params_)})
        r = pd.DataFrame([x for x in rows if x["modelo"] == name])
        print(f"{name:16s} acc {r.accuracy.mean():.3f} ± {r.accuracy.std():.3f}  "
              f"f1 {r.f1_macro.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), candidates


def fit_final(name, candidate, y, actor):
    """Elige hiperparámetros con GroupKFold(6) sobre los 24 actores y entrena con los 1440 clips."""
    import joblib
    from sklearn.model_selection import GridSearchCV, GroupKFold
    X, pipe, grid = candidate
    gs = GridSearchCV(pipe, grid, cv=GroupKFold(6), scoring="accuracy", n_jobs=WORKERS, refit=True)
    gs.fit(X, y, groups=actor)
    dest = ROOT / "models" / f"{name}.joblib"
    joblib.dump({"modelo": gs.best_estimator_, "params": gs.best_params_, "clases": ORDER,
                 "descriptores": "ser_model.summary_features" if "resumen" in name else "ser_model.frame_features"}, dest)
    print(f"\nmodelo final {name}: {gs.best_params_} (CV por actor {gs.best_score_:.3f}) → {dest}")
    return gs.best_params_


# ---------------------------------------------------------------- CNN
def sample_cnn_configs(n, rng, epochs=80, patience=15, widths=None):
    widths = widths or [(64, 64, 32, 32, 16), (128, 128, 64, 64, 32), (256, 256, 128, 128, 64)]
    cfgs = []
    for _ in range(n):
        cfgs.append({"lr": float(10 ** rng.uniform(-3.7, -2.7)), "weight_decay": float(10 ** rng.uniform(-4, -1.5)),
                     "width": widths[rng.integers(len(widths))], "label_smoothing": float(rng.choice([0, 0.1])),
                     "bs": int(rng.choice([32, 64])), "epochs": epochs, "patience": patience})
    return cfgs


def train_cnn(cfg, Xtr, ytr, Xva, yva, deadline=None, log=None):
    import torch
    from torch import nn
    from ser_model import ConvSER
    torch.manual_seed(SEED)
    model = ConvSER(Xtr.shape[1], len(ORDER), width=tuple(cfg["width"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=cfg["epochs"],
                                                steps_per_epoch=-(-len(Xtr) // cfg["bs"]))
    best, best_state, wait, ep = -1, None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr)); loss_sum = 0.0
        for i in range(0, len(perm), cfg["bs"]):
            b = perm[i:i + cfg["bs"]]
            if len(b) < 2:
                sched.step(); continue
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xtr[b]), ytr[b], label_smoothing=cfg["label_smoothing"])
            loss.backward(); opt.step(); sched.step(); loss_sum += loss.item() * len(b)
        acc = (predict_cnn(model, Xva) == yva.numpy()).mean()
        if log is not None:
            sub = slice(0, len(Xtr), 3)  # una variante por clip para medir el ajuste en entrenamiento
            acc_tr = (predict_cnn(model, Xtr[sub]) == ytr[sub].numpy()).mean()
            log.append({"epoca": ep + 1, "loss": loss_sum / len(perm), "acc_train": acc_tr, "acc_val": acc,
                        "lr": opt.param_groups[0]["lr"]})
        if acc > best:
            best, best_state, wait = acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
        if deadline and time.time() > deadline:
            print("  ⏱ límite de tiempo alcanzado; se detiene este entrenamiento", flush=True)
            break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def predict_cnn(model, X, bs=256):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).numpy()


def cnn_search(V, y, actor, n_cfg, epochs=80, patience=15, widths=None, budget_min=None):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(16)
    cfgs = sample_cnn_configs(n_cfg, np.random.default_rng(SEED), epochs, patience, widths)
    deadline = time.time() + 60 * budget_min if budget_min else None
    rows, trials, history, saved = [], [], [], {}
    for k, te in enumerate(outer_folds(actor)):
        train_actors = np.unique(actor[~te])
        va = np.isin(actor, np.random.default_rng(SEED + k).choice(train_actors, 4, replace=False))
        tr = ~te & ~va
        Xtr = V[tr].reshape(-1, V.shape[-1]); ytr = np.repeat(y[tr], V.shape[1])  # aumentado solo en train
        sc = StandardScaler().fit(Xtr)
        T = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
        Xtr_t, ytr_t = T(Xtr), torch.tensor(ytr)
        Xva_t, yva_t = T(V[va, 0]), torch.tensor(y[va])
        best = None
        for c, cfg in enumerate(cfgs):
            t0 = time.time()
            log = []
            model, acc_va, n_ep = train_cnn(cfg, Xtr_t, ytr_t, Xva_t, yva_t, deadline, log)
            history += [{"fold": k + 1, "config": c, **h} for h in log]
            trials.append({"fold": k + 1, "config": c, **cfg, "acc_val": acc_va, "epocas": n_ep})
            print(f"  fold {k+1} cfg {c} val {acc_va:.3f} ({n_ep} ép., {time.time()-t0:.0f}s) {cfg}", flush=True)
            if best is None or acc_va > best[0]:
                best = (acc_va, c, model)
        pred = predict_cnn(best[2], T(V[te, 0]))
        rows.append({"modelo": "cnn_tuned", "fold": k + 1, "actores": f"{4*k+1}-{4*k+4}",
                     "accuracy": accuracy_score(y[te], pred), "f1_macro": f1_score(y[te], pred, average="macro"),
                     "inner_cv": best[0], "params": json.dumps(cfgs[best[1]])})
        print(f"fold {k+1}: cfg {best[1]} → test acc {rows[-1]['accuracy']:.3f}", flush=True)
        saved[k + 1] = {"state_dict": best[2].state_dict(), "config": cfgs[best[1]],
                        "scaler_mean": sc.mean_, "scaler_scale": sc.scale_}
    torch.save({"folds": saved, "clases": ORDER, "n_features": V.shape[-1]}, ROOT / "models" / "cnn_tuned_folds.pt")
    return pd.DataFrame(rows), pd.DataFrame(trials), pd.DataFrame(history)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=int, default=0, help="nº de configuraciones aleatorias de la CNN (0 = omitir)")
    ap.add_argument("--epochs", type=int, default=80, help="máximo de épocas por entrenamiento de la CNN")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--small", action="store_true", help="solo el ancho 64-64-32-32-16 (más rápido en CPU)")
    ap.add_argument("--budget-min", type=float, default=None, help="límite de tiempo total de la CNN, en minutos")
    args = ap.parse_args()

    from ravdess_data import load_metadata, ensure_audio
    df = load_metadata().sort_values("filename").reset_index(drop=True)
    print("metadatos desde Supabase:", len(df), "clips")
    df = ensure_audio(df, workers=16)
    y = df["emotion"].map({e: i for i, e in enumerate(ORDER)}).values
    actor = df["actor"].values.astype(int)
    V, S = extract(df)

    OUT.mkdir(exist_ok=True)
    res, candidates = classical(S, V, y, actor)
    if args.cnn:
        cnn, trials, history = cnn_search(V, y, actor, args.cnn, args.epochs, args.patience,
                                          [(64, 64, 32, 32, 16)] if args.small else None, args.budget_min)
        res = pd.concat([res, cnn])
        trials.to_csv(OUT / "cnn_trials.csv", index=False)
        history.to_csv(OUT / "cnn_historial.csv", index=False)
    res.to_csv(OUT / "folds.csv", index=False)

    piv = res.pivot(index="fold", columns="modelo", values="accuracy")
    print("\naccuracy por fold (actores de prueba nunca vistos):\n", piv.round(3).to_string())
    resumen = res.groupby("modelo").agg(acc_media=("accuracy", "mean"), acc_desv=("accuracy", "std"),
                                        f1_media=("f1_macro", "mean")).sort_values("acc_media", ascending=False)
    resumen["gana_a_logreg_plano"] = [int((piv[m] > piv["logreg_plano"]).sum()) for m in resumen.index]
    resumen.to_csv(OUT / "resumen.csv")
    print("\n", resumen.round(3).to_string())

    best = next(m for m in resumen.index if m in candidates)
    fit_final(best, candidates[best], y, actor)


if __name__ == "__main__":
    main()
