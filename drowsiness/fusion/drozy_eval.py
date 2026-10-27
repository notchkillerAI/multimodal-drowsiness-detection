"""Replay every DROZY test (EAR + CNN + EOG fused, no drawing) and compare the result with the label.

    python -m drowsiness.fusion.drozy_eval                 # all tests, 4 at a time; a finished test is not repeated
    python -m drowsiness.fusion.drozy_eval --report-only   # only the table and the plot from the finished logs

Each test writes `runs/drozy/s<subject>t<test>.csv` (one row per video frame: level, scores, confidences, KSS). The results are
made from those logs. Nothing is tuned on these tests: the cameras and the fusion rules are hand-set and the EOG model was
trained on another dataset, so this is a test on unseen data, but of 36 tests of 14 people (about 30 h of replay on one Mac), and the
label is one self-rating per test."""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from drowsiness.bio import drozy
from drowsiness.fusion.drozy_replay import kss_word, run

LOG_DIR = "runs/drozy"
ALARM = ("DROWSY", "MICROSLEEP", "SLEEP", "UNRESPONSIVE")
BRANCHES = ("ear", "cnn", "eog")


def log_path(subject, test):
    return f"{LOG_DIR}/s{subject}t{test}.csv"


def is_complete(path):
    """A log counts as finished if it reaches the end of the 10-minute test."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 400))
        last = f.read().decode(errors="ignore").strip().splitlines()[-1]
    try:
        return float(last.split(",")[0]) >= 595.0
    except ValueError:
        return False


def replay_one(job):
    subject, test = job
    path = log_path(subject, test)
    if not is_complete(path):
        run(subject, test, show=None, log_csv=path)
    return job


def read_log(subject, test):
    with open(log_path(subject, test)) as f:
        rows = list(csv.DictReader(f))
    number = lambda r, k: float(r[k]) if r[k] != "" else np.nan  # noqa: E731
    cols = {
        k: np.array([number(r, k) for r in rows])
        for k in ("time_s", "fused_score") + tuple(f"{b}_{x}" for b in BRANCHES for x in ("score", "conf"))
    }
    cols["level"] = np.array([r["level"] for r in rows])
    return cols


def summarise(subject, test):
    d = read_log(subject, test)
    n = len(d["level"])
    out = {"subject": subject, "test": test, "kss": drozy.kss(subject, test), "frames": n}
    out["fused_mean"] = float(np.nanmean(d["fused_score"]))
    for b in BRANCHES:  # confidence-weighted mean score where the branch had a score
        w = np.where(np.isnan(d[f"{b}_score"]), 0.0, d[f"{b}_conf"])
        out[f"{b}_mean"] = float(np.nansum(w * np.nan_to_num(d[f"{b}_score"])) / w.sum()) if w.sum() > 0 else np.nan
    out["alarm_share"] = float(np.isin(d["level"], ALARM).mean())
    out["warning_share"] = float((d["level"] == "EARLY WARNING").mean())
    out["alert_share"] = float((d["level"] == "ALERT").mean())
    out["unreliable_share"] = float((d["level"] == "SENSORS UNRELIABLE").mean())
    return out


def auc(score, positive):
    """Probability that a random sleepy test scores higher than a random not-sleepy one (ties count half)."""
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def comparison(rows):
    """AUC (KSS >= 7 vs <= 6) and Spearman correlation with KSS of each measure, over all tests."""
    kss = np.array([r["kss"] for r in rows])
    sleepy = kss >= 7
    table = {}
    for name, key in (
        ("EAR alone", "ear_mean"),
        ("CNN alone", "cnn_mean"),
        ("EOG alone", "eog_mean"),
        ("fused score", "fused_mean"),
        ("share of time in DROWSY or worse", "alarm_share"),
    ):
        v = np.array([r[key] for r in rows], float)
        ok = ~np.isnan(v)
        table[name] = {"auc": auc(v[ok], sleepy[ok]), "spearman": spearman(v[ok], kss[ok]), "n": int(ok.sum())}
    return table


def report(rows):
    rows = sorted(rows, key=lambda r: (r["subject"], r["test"]))
    pct = lambda x: f"{100 * x:5.0f}%"  # noqa: E731
    print(f"{'test':>5} {'KSS':>3}  {'label':9} {'ALERT':>6} {'WARN':>6} {'DROWSY+':>7} {'fused':>6} {'EAR':>5} {'CNN':>5} {'EOG':>5}")
    for r in rows:
        scores = " ".join(f"{r[k]:5.2f}" for k in ("ear_mean", "cnn_mean", "eog_mean"))
        print(
            f"{r['subject']:>2}-{r['test']:<2} {r['kss']:>3}  {kss_word(r['kss']):9} "
            f"{pct(r['alert_share'])} {pct(r['warning_share'])} {pct(r['alarm_share'])} {r['fused_mean']:6.2f} {scores}"
        )
    sleepy = [r for r in rows if r["kss"] >= 7]
    other = [r for r in rows if r["kss"] <= 6]
    print(f"\n{len(sleepy)} tests with KSS >= 7, {len(other)} with KSS <= 6.")
    print("per measure (one number per test):")
    for name, m in comparison(rows).items():
        print(f"  {name:34} AUC {m['auc']:.2f}   Spearman with KSS {m['spearman']:+.2f}   (n = {m['n']})")
    for label, group in (("KSS <= 6", other), ("KSS >= 7", sleepy)):
        alert = 100 * np.mean([r["alert_share"] for r in group])
        alarm = 100 * np.mean([r["alarm_share"] for r in group])
        print(f"  {label}: mean time in ALERT {alert:.0f} %, in DROWSY or worse {alarm:.0f} %")
    return rows


def plot(rows, path="docs/drozy_results.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kss = np.array([r["kss"] for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (title, key) in zip(
        axes, (("EAR alone", "ear_mean"), ("CNN alone", "cnn_mean"), ("EOG alone", "eog_mean"), ("fused", "fused_mean")), strict=True
    ):
        v = np.array([r[key] for r in rows], float)
        ax.scatter(kss + np.random.default_rng(0).uniform(-0.15, 0.15, len(kss)), v, c=np.where(kss >= 7, "#e53935", "#4c8c4a"), s=28)
        ax.axvline(6.5, color="grey", ls=":")
        ax.set(title=title, xlabel="KSS (own rating of the test)", xticks=range(1, 10))
    axes[0].set_ylabel("mean score of the test\n(0 alert, 1 drowsy)")
    fig.suptitle("DROZY, all tests replayed: red = KSS 7 or more", y=1.02)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def cli():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    jobs = drozy.tests()
    os.makedirs(LOG_DIR, exist_ok=True)
    if not a.report_only:
        todo = [j for j in jobs if not is_complete(log_path(*j))]
        print(f"{len(jobs) - len(todo)} of {len(jobs)} tests already replayed, {len(todo)} to do", flush=True)
        with ProcessPoolExecutor(a.workers) as pool:
            for done in pool.map(replay_one, todo):
                print("finished", done, flush=True)
    rows = report([summarise(*j) for j in jobs if is_complete(log_path(*j))])
    print("plot:", plot(rows))


if __name__ == "__main__":
    cli()
