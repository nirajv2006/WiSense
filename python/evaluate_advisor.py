"""Evaluate the decision layer (power recommendations), not just the classifier.

    python evaluate_advisor.py

For every session: train on the OTHER sessions, predict this session window by
window (its recordings are played back to back like one continuous period),
pass the predictions through the same smoothing + PowerAdvisor used live, and
compare the recommended level with the level the TRUE occupancy calls for.

Reported per session and overall:
  matched         % of time the recommendation equals the ideal level
  under-provided  % of time below the ideal level   -> comfort risk
  over-provided   % of time above the ideal level   -> unused capacity
  changes / hour  how often the recommendation switches (stability)
Compared with acting directly on raw classifier output (no smoothing/hysteresis).
These are decision-quality metrics; they are not measured energy savings.
"""
import argparse
import collections

import numpy as np
import pandas as pd
from sklearn.base import clone

from wisense import config
from wisense.features import FEATURE_NAMES
from wisense.models import MAIN_MODEL, make_models
from wisense.power import PowerAdvisor

LEVEL_IDX = {lv: i for i, lv in enumerate(config.POWER_ORDER)}
IDEAL = {c: config.POWER_LEVELS[c]["level"] for c in config.CLASSES}


def score(rec_levels, true_labels, step_s):
    r = np.array([LEVEL_IDX[l] for l in rec_levels])
    t = np.array([LEVEL_IDX[IDEAL[l]] for l in true_labels])
    hours = len(r) * step_s / 3600
    return {"matched": (r == t).mean(), "under_provided": (r < t).mean(), "over_provided": (r > t).mean(),
            "changes_per_hour": (np.diff(r) != 0).sum() / max(hours, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.FEATURES_CSV))
    ap.add_argument("--model", default=MAIN_MODEL)
    ap.add_argument("--step-down", type=float, nargs="+", default=[0, 10, 30, config.STEP_DOWN_AFTER_S],
                    help="step-down delays (s) to compare")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = df[df.label.isin(config.CLASSES)]
    if df.session.nunique() < 2:
        raise SystemExit("Need at least 2 sessions (train on one, test on another).")
    model = make_models()[args.model]

    rows = []
    for session in sorted(df.session.unique()):
        train, test = df[df.session != session], df[df.session == session]
        test = test.sort_values(["recording", "window_start_s"])   # recordings back to back
        m = clone(model).fit(train[FEATURE_NAMES].to_numpy(float), train.label.to_numpy(object))
        proba = m.predict_proba(test[FEATURE_NAMES].to_numpy(float))
        order = [list(m.classes_).index(c) for c in config.CLASSES]
        proba = proba[:, order]

        truth = test.label.tolist()
        naive = [IDEAL[config.CLASSES[int(pr.argmax())]] for pr in proba]
        rows.append({"session": session, "policy": "raw classifier (no smoothing)",
                     **score(naive, truth, config.STEP_S)})
        for down in args.step_down:
            adv = PowerAdvisor(step_up_s=config.STEP_UP_AFTER_S, step_down_s=down)
            hist, rec = collections.deque(maxlen=config.PROBA_SMOOTHING), []
            for i, pr in enumerate(proba):
                hist.append(pr)
                sm = np.mean(hist, axis=0)
                rec.append(adv.update(config.CLASSES[int(sm.argmax())], float(sm.max()),
                                      i * config.STEP_S).level)
            rows.append({"session": session, "policy": f"advisor, step-down {down:g} s",
                         **score(rec, truth, config.STEP_S)})

    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(res.to_string(index=False))
    print("\nMean over sessions:")
    order = list(dict.fromkeys(res.policy))
    print(res.groupby("policy")[["matched", "under_provided", "over_provided", "changes_per_hour"]]
          .mean().reindex(order))
    if df.synthetic.any():
        print("\n(SYNTHETIC DATA - software test only)")
    print("\nRead this as a trade-off: longer step-down delays switch less often and rarely under-provide, "
          "but stay high longer after people leave.\nRecordings are only ~1 min long and played back to back, "
          "which is much choppier than real occupancy - record longer continuous sessions for a realistic test.")
    out = config.REPORTS_DIR / "latest"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "advisor_evaluation.csv", index=False)
    print(f"\nSaved {out / 'advisor_evaluation.csv'}")


if __name__ == "__main__":
    main()
