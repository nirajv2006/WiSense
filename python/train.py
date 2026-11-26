"""Train and evaluate occupancy classifiers, then save the deployed model.

    python train.py                       # evaluate all models, deploy Random Forest
    python train.py --deploy "Logistic regression"

Evaluation schemes (from most optimistic to most honest):
  random      shuffled 5-fold over windows. Overlapping windows from the same
              recording land in train AND test, so scores are inflated.
              Reported only to show how big that leakage effect is.
  recording   GroupKFold by recording: a whole recording is unseen.
  session     leave-one-session-out: train on other days/setups, test on an
              unseen one. This is the headline number.
  condition   leave-one-condition-out (if conditions vary independently of
              sessions), e.g. trained with door closed, tested with door open.

Outputs: reports/<timestamp>/ (also copied to reports/latest/) and models/wisense_model.joblib
"""
import argparse
import json
import shutil
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

from wisense import config
from wisense.features import FEATURE_GROUPS, FEATURE_NAMES
from wisense.models import MAIN_MODEL, make_models

CLASSES = config.CLASSES


# ── Cross-validation helpers ─────────────────────────────────────────────────
def schemes_for(df):
    out = {"random": (StratifiedKFold(5, shuffle=True, random_state=0), None)}
    n_rec = df.recording.nunique()
    if n_rec >= 3:
        out["recording"] = (GroupKFold(n_splits=min(5, n_rec)), df.recording.values)
    if df.session.nunique() >= 2:
        out["session"] = (LeaveOneGroupOut(), df.session.values)
    cond_per_session = df.groupby("session").condition.nunique().max()
    if df.condition.nunique() >= 2 and (cond_per_session > 1 or
                                        df.condition.nunique() < df.session.nunique()):
        out["condition"] = (LeaveOneGroupOut(), df.condition.values)
    return out


def run_cv(model, X, y, splitter, groups):
    pred = np.empty(len(y), dtype=object)
    fold_f1, fold_names = [], []
    for tr, te in splitter.split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        fold_f1.append(f1_score(y[te], pred[te], average="macro", labels=CLASSES, zero_division=0))
        fold_names.append(str(groups[te][0]) if groups is not None else f"fold{len(fold_f1)}")
    return pred, fold_f1, fold_names


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, average="macro", zero_division=0)
    _, _, fc, sup = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "f1_per_class": dict(zip(CLASSES, fc.round(4).tolist())),
            "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist()}


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_confusion(cm, title, path):
    cm = np.array(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(table, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    table.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_title("Model comparison by evaluation scheme")
    ax.legend(title="Evaluated on", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp, path):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Drop in macro F1 when feature is shuffled")
    ax.set_title("Permutation importance (unseen-session folds)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(df, feats, path):
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 3.4))
    for ax, f in zip(np.atleast_1d(axes), feats):
        ax.boxplot([df.loc[df.label == c, f] for c in CLASSES], showfliers=False)
        ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
        ax.set_title(f, fontsize=9)
    fig.suptitle("Top features by occupancy class", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.FEATURES_CSV))
    ap.add_argument("--deploy", default=MAIN_MODEL, help="model to save for live use")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = df[df.label.isin(CLASSES)].reset_index(drop=True)
    synthetic = bool(df.synthetic.any())
    X, y = df[FEATURE_NAMES].to_numpy(float), df.label.to_numpy(object)
    models = make_models()
    if args.deploy not in models:
        raise SystemExit(f"--deploy must be one of {list(models)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.REPORTS_DIR / stamp
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(df)} windows | {df.recording.nunique()} recordings | {df.session.nunique()} sessions"
          f"{' | SYNTHETIC DATA' if synthetic else ''}")
    print(df.label.value_counts().reindex(CLASSES).to_string(), "\n")

    schemes = schemes_for(df)
    primary = "session" if "session" in schemes else ("recording" if "recording" in schemes else "random")
    results, preds, folds = {}, {}, {}
    for sname, (splitter, groups) in schemes.items():
        for mname, model in models.items():
            pred, fold_f1, fold_names = run_cv(model, X, y, splitter, groups)
            results[(mname, sname)] = metrics(y, pred)
            results[(mname, sname)]["fold_f1"] = dict(zip(fold_names, np.round(fold_f1, 4).tolist()))
            preds[(mname, sname)] = pred
            print(f"{sname:10s} {mname:22s} acc {results[(mname, sname)]['accuracy']:.3f}  "
                  f"macro-F1 {results[(mname, sname)]['f1_macro']:.3f}")
        print()

    table = pd.DataFrame({s: {m: results[(m, s)]["f1_macro"] for m in models} for s in schemes})
    plot_comparison(table, out / "model_comparison.png")
    for m in [args.deploy, "Variance threshold"]:
        plot_confusion(results[(m, primary)]["confusion"], f"{m} - unseen {primary}",
                       out / f"confusion_{m.lower().replace(' ', '_')}.png")

    # Feature-group ablation for the deployed model on the primary scheme
    splitter, groups = schemes[primary]
    ablation = {}
    sets = {"all features": FEATURE_NAMES,
            "without level": [f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["level"]],
            "variation only": FEATURE_GROUPS["variation"],
            "variation + spectral": FEATURE_GROUPS["variation"] + FEATURE_GROUPS["spectral"],
            "level only": FEATURE_GROUPS["level"]}
    for name, feats in sets.items():
        idx = [FEATURE_NAMES.index(f) for f in feats]
        if args.deploy == "Variance threshold":
            break
        pred, _, _ = run_cv(models[args.deploy], X[:, idx], y, splitter, groups)
        ablation[name] = round(f1_score(y, pred, average="macro"), 4)

    # Permutation importance on held-out folds
    imps = []
    for tr, te in splitter.split(X, y, groups):
        m = clone(models[args.deploy]).fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], scoring="f1_macro", n_repeats=5, random_state=0, n_jobs=-1)
        imps.append(r.importances_mean)
    importance = pd.Series(np.mean(imps, axis=0), index=FEATURE_NAMES).round(4)
    plot_importance(importance, out / "feature_importance.png")
    plot_distributions(df, importance.sort_values(ascending=False).index[:4].tolist(),
                       out / "feature_distributions.png")

    # Deploy: fit on all data
    final = clone(models[args.deploy]).fit(X, y)
    config.MODELS_DIR.mkdir(exist_ok=True)
    meta = {
        "model_name": args.deploy, "trained_at": stamp, "feature_names": FEATURE_NAMES,
        "classes": CLASSES, "class_definitions": config.CLASS_DEFINITIONS,
        "window_s": config.WINDOW_S, "step_s": config.STEP_S, "n_windows": len(df),
        "sessions": sorted(df.session.unique().tolist()), "synthetic": synthetic,
        "primary_scheme": primary, "cv_f1_macro": results[(args.deploy, primary)]["f1_macro"],
    }
    joblib.dump({"model": final, "meta": meta}, config.MODEL_PATH)

    report = {
        "meta": meta,
        "schemes": list(schemes),
        "results": {f"{m}|{s}": v for (m, s), v in results.items()},
        "comparison_f1": table.round(4).to_dict(),
        "ablation_f1": ablation,
        "permutation_importance": importance.to_dict(),
        "dataset": {"windows_per_class": df.label.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
                    "windows_per_session": df.groupby("session").size().to_dict()},
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))

    best = results[(args.deploy, primary)]
    md = [f"# WiSense evaluation report ({stamp})", ""]
    if synthetic:
        md += ["> **SYNTHETIC DATA.** These numbers only test the software. Do not report them.", ""]
    md += [f"Dataset: {len(df)} windows ({config.WINDOW_S:.0f} s, step {config.STEP_S:.0f} s), "
           f"{df.recording.nunique()} recordings, {df.session.nunique()} sessions.", "",
           "## Macro-F1 by model and evaluation scheme", "", table.round(3).to_markdown(), "",
           f"Headline scheme: **unseen {primary}**. 'random' is shown only to quantify window leakage.", "",
           f"## {args.deploy} on unseen {primary}", "",
           f"Accuracy {best['accuracy']:.3f}, macro precision {best['precision_macro']:.3f}, "
           f"macro recall {best['recall_macro']:.3f}, macro F1 {best['f1_macro']:.3f}", "",
           "Per-fold macro F1: " + ", ".join(f"{k}: {v:.3f}" for k, v in best["fold_f1"].items()), "",
           "```", classification_report(y, preds[(args.deploy, primary)], labels=CLASSES, zero_division=0), "```", "",
           "## Feature-group ablation (macro F1)", "",
           pd.Series(ablation, name="macro F1").to_frame().to_markdown(), "",
           "## Most important features", "",
           importance.sort_values(ascending=False).head(8).to_frame("importance").to_markdown(), ""]
    (out / "report.md").write_text("\n".join(md))

    latest = config.REPORTS_DIR / "latest"
    if latest.exists():
        shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(out, latest, dirs_exist_ok=True)

    print(f"Headline ({args.deploy}, unseen {primary}): macro-F1 {best['f1_macro']:.3f}")
    print("Ablation:", ablation)
    print(f"Report: {out / 'report.md'}\nModel:  {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
