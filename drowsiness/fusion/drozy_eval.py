"""Run the whole DROZY dataset through the pipelines and compare the result with the label.

    python -m drowsiness.fusion.drozy_eval                 # step 1 (slow, once) + step 2
    python -m drowsiness.fusion.drozy_eval --report-only   # step 2 only

Step 1, extraction (about 25 frames per second per worker): every test is replayed through the EAR, CNN and EOG branches
and what each branch said at every video frame is saved to `runs/drozy/s<subject>t<test>.npz`. A finished test is not repeated.
Step 2, fusion and report (seconds): the saved branch outputs are fed to `Fusion`, so the fusion rules and weights can be
changed and compared without replaying 6 hours of video. It prints the per-test table, compares the single branches, a majority
vote and the weighted fusion, tunes the weights with leave-one-person-out, and saves `docs/drozy_results.png`.

Nothing in the branches is tuned on these tests (the cameras are hand-set, the EOG model was trained on another dataset).
The label is one self-rating (KSS) per 10-minute test of 14 people: enough for a first honest look, not for an accuracy."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from scipy.stats import rankdata

from drowsiness.bio import drozy
from drowsiness.bio.drozy import kss_word
from drowsiness.fusion.fuse import DEFAULT_TRUST, Fusion

CACHE_DIR = "runs/drozy"
ALARM = ("DROWSY", "MICROSLEEP", "SLEEP", "UNRESPONSIVE")
BRANCHES = ("ear", "cnn", "eog")
SLEEPY_KSS = 7  # EU drowsiness-warning rules: KSS 7 optional, 8-9 required
VOTE_CONF, VOTE_SCORE = 0.5, 0.5  # majority vote: a confident branch (0.5) votes "drowsy" when its score is 0.5 or more
MIN_USABLE = 0.10  # as in Fusion: below this total of trust x confidence there is no fused score


# ---------------------------------------------------------------- step 1: extraction
def cache_path(subject, test):
    return f"{CACHE_DIR}/s{subject}t{test}.npz"


def extract_one(job):
    """Replay one test and save, per video frame, each branch's score, confidence and (cameras) current eye-closure time."""
    from drowsiness.fusion.drozy_replay import Replay  # imported here: the report does not need OpenCV, dlib or ONNX

    subject, test = job
    path = cache_path(subject, test)
    if os.path.exists(path):
        return job
    replay = Replay(subject, test)
    cols = {"t": []} | {f"{b}_{k}": [] for b in BRANCHES for k in ("score", "conf", "closure")}
    try:
        while (step := replay.step()) is not None:
            cols["t"].append(step[0])
            for name, branch in replay.branches.items():
                out = branch.latest()
                cols[f"{name}_score"].append(out["score"])
                cols[f"{name}_conf"].append(out["confidence"])
                cols[f"{name}_closure"].append(out["raw"].get("current_closure_s", np.nan))  # nan: the branch has no closure time
    finally:
        replay.stop()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path + ".part.npz", **{k: np.array(v, dtype=float) for k, v in cols.items()})
    os.replace(path + ".part.npz", path)  # a test that was interrupted leaves no half-written file
    return job


def load(subject, test):
    with np.load(cache_path(subject, test)) as data:
        cache = {k: data[k] for k in data.files}
    return cache | {"subject": subject, "test": test, "kss": drozy.kss(subject, test)}


# ---------------------------------------------------------------- step 2: fusion on the saved branch outputs
def fuse(cache, trust=None, **settings):
    """Feed the saved branch outputs to Fusion, frame by frame. Returns (levels, smoothed scores)."""
    fusion = Fusion(trust=trust, **settings)
    names = [b for b in BRANCHES if trust is None or trust.get(b, 0) > 0]
    levels, smoothed = [], []
    for i, t in enumerate(cache["t"]):
        outputs = {}
        for name in names:
            closure = cache[f"{name}_closure"][i]
            raw = {} if np.isnan(closure) else {"current_closure_s": float(closure)}
            outputs[name] = {
                "score": float(cache[f"{name}_score"][i]),
                "confidence": float(cache[f"{name}_conf"][i]),
                "raw": raw,
                "timestamp": t,
            }
        state = fusion.step(float(t), outputs)
        levels.append(state["level"])
        smoothed.append(np.nan if state["smoothed_score"] is None else state["smoothed_score"])
    return np.array(levels), np.array(smoothed)


def weighted_mean_score(cache, trust):
    """Mean over the test of the fused score  sum(w c s) / sum(w c)  (frames with too little usable weight are left out).
    The same vote as Fusion without its smoothing and rules, vectorised so that many weight settings can be compared."""
    w = np.array([trust.get(b, 0.0) for b in BRANCHES])[:, None]
    conf = np.array([cache[f"{b}_conf"] for b in BRANCHES])
    score = np.array([cache[f"{b}_score"] for b in BRANCHES])
    usable = (w * conf).sum(axis=0)
    ok = usable >= MIN_USABLE * w.sum() / sum(DEFAULT_TRUST[b] for b in BRANCHES)  # the limit scales with the size of the weights
    return float(((w * conf * score).sum(axis=0)[ok] / usable[ok]).mean()) if ok.any() else np.nan


def majority_share(cache):
    """Share of the test in which most of the confident branches vote "drowsy" (a tie is not a majority)."""
    conf = np.array([cache[f"{b}_conf"] for b in BRANCHES]) >= VOTE_CONF
    drowsy = (np.array([cache[f"{b}_score"] for b in BRANCHES]) >= VOTE_SCORE) & conf
    voted = conf.sum(axis=0) > 0
    return float((drowsy.sum(axis=0)[voted] > conf.sum(axis=0)[voted] / 2).mean()) if voted.any() else np.nan


def summarise(cache, trust=None, **settings):
    """One row per test: the label, each branch alone, the majority vote and the fusion (score and time in each level)."""
    levels, smoothed = fuse(cache, trust, **settings)
    row = {"subject": cache["subject"], "test": cache["test"], "kss": cache["kss"], "frames": len(levels)}
    for b in BRANCHES:  # a branch alone: its confidence-weighted mean score
        row[f"{b}_mean"] = weighted_mean_score(cache, {b: 1.0})
    row["majority_share"] = majority_share(cache)
    row["fused_mean"] = float(np.nanmean(smoothed)) if not np.isnan(smoothed).all() else np.nan
    row["alarm_share"] = float(np.isin(levels, ALARM).mean())
    row["warning_share"] = float((levels == "EARLY WARNING").mean())
    row["alert_share"] = float((levels == "ALERT").mean())
    row["unreliable_share"] = float((levels == "SENSORS UNRELIABLE").mean())
    return row


# ---------------------------------------------------------------- statistics
def auc(score, positive):
    """Probability that a random sleepy test scores higher than a random not-sleepy one (ties count half)."""
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def spearman(a, b):
    """Rank correlation; tied values (KSS has many) share their mean rank."""
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def auc_interval(score, positive, subjects, n=2000, seed=0):
    """95 % interval of the AUC by resampling PEOPLE (the tests of one person are not independent)."""
    rng, people, values = np.random.default_rng(seed), np.unique(subjects), []
    for _ in range(n):
        idx = np.concatenate([np.flatnonzero(subjects == p) for p in rng.choice(people, len(people))])
        if positive[idx].any() and not positive[idx].all():
            values.append(auc(score[idx], positive[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def measure(values, kss, subjects):
    values, ok = np.asarray(values, float), ~np.isnan(np.asarray(values, float))
    sleepy = kss >= SLEEPY_KSS
    low, high = auc_interval(values[ok], sleepy[ok], subjects[ok])
    return {"auc": auc(values[ok], sleepy[ok]), "low": low, "high": high, "spearman": spearman(values[ok], kss[ok]), "n": int(ok.sum())}


# ---------------------------------------------------------------- weights tuned with leave-one-person-out
def weight_grid(steps=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """All different mixes of the three branch weights (scaled so that they sum to 1)."""
    seen, grid = set(), []
    for w in product(steps, repeat=len(BRANCHES)):
        if sum(w) > 0 and (key := tuple(round(x / sum(w), 4) for x in w)) not in seen:
            seen.add(key)
            grid.append(dict(zip(BRANCHES, key, strict=True)))
    return grid


def tune_weights(caches):
    """For every person in turn: choose the weights that separate the OTHER people's sleepy tests best (AUC of the mean fused
    score), then score that person's tests with them. Returns the held-out scores, the weights chosen per person and the
    weights that are best on all the data (in-sample, so optimistic)."""
    grid = weight_grid()
    table = np.array([[weighted_mean_score(c, w) for c in caches] for w in grid])  # weights x tests
    kss, subjects = np.array([c["kss"] for c in caches]), np.array([c["subject"] for c in caches])
    sleepy = kss >= SLEEPY_KSS
    held_out, chosen = np.full(len(caches), np.nan), {}
    for person in np.unique(subjects):
        train = subjects != person
        best = int(np.argmax([auc(row[train], sleepy[train]) for row in table]))
        held_out[~train], chosen[int(person)] = table[best][~train], grid[best]
    in_sample = [auc(row, sleepy) for row in table]
    return held_out, chosen, grid[int(np.argmax(in_sample))], float(np.max(in_sample))


def test_level_decision(values, kss, subjects):
    """Is a whole test "sleepy"? The cut-off on the mean fused score is chosen on the other people (best balanced accuracy)
    and applied to the held-out person. Returns the confusion counts (tp, fn, fp, tn)."""
    sleepy, calls = kss >= SLEEPY_KSS, np.zeros(len(values), bool)
    for person in np.unique(subjects):
        train = subjects != person
        cuts = np.unique(values[train])
        balanced = [((values[train] >= c)[sleepy[train]].mean() + (values[train] < c)[~sleepy[train]].mean()) / 2 for c in cuts]
        calls[~train] = values[~train] >= cuts[int(np.argmax(balanced))]
    return int((calls & sleepy).sum()), int((~calls & sleepy).sum()), int((calls & ~sleepy).sum()), int((~calls & ~sleepy).sum())


# ---------------------------------------------------------------- report
def report(caches):
    rows = [summarise(c) for c in caches]
    kss, subjects = np.array([r["kss"] for r in rows]), np.array([r["subject"] for r in rows])
    pct = lambda x: f"{100 * x:5.0f}%"  # noqa: E731
    print(f"{'test':>5} {'KSS':>3}  {'label':9} {'ALERT':>6} {'WARN':>6} {'DROWSY+':>7} {'fused':>6} {'EAR':>5} {'CNN':>5} {'EOG':>5}")
    for r in rows:
        scores = " ".join(f"{r[k]:5.2f}" for k in ("ear_mean", "cnn_mean", "eog_mean"))
        print(
            f"{r['subject']:>2}-{r['test']:<2} {r['kss']:>3}  {kss_word(r['kss']):9} "
            f"{pct(r['alert_share'])} {pct(r['warning_share'])} {pct(r['alarm_share'])} {r['fused_mean']:6.2f} {scores}"
        )
    sleepy, other = [r for r in rows if r["kss"] >= SLEEPY_KSS], [r for r in rows if r["kss"] < SLEEPY_KSS]
    for label, group in ((f"KSS < {SLEEPY_KSS}", other), (f"KSS >= {SLEEPY_KSS}", sleepy)):
        alert, alarm = 100 * np.mean([r["alert_share"] for r in group]), 100 * np.mean([r["alarm_share"] for r in group])
        print(f"  {label} ({len(group)} tests): mean time in ALERT {alert:.0f} %, in DROWSY or worse {alarm:.0f} %")

    held_out, chosen, best_in_sample, best_auc = tune_weights(caches)
    measures = {
        "EAR alone": [r["ear_mean"] for r in rows],
        "CNN alone": [r["cnn_mean"] for r in rows],
        "EOG alone": [r["eog_mean"] for r in rows],
        "majority vote (share of time)": [r["majority_share"] for r in rows],
        "weighted fusion, default weights": [r["fused_mean"] for r in rows],
        "weighted fusion, weights tuned on the other people": held_out,
        "fusion levels: share of time in DROWSY or worse": [r["alarm_share"] for r in rows],
    }
    print("\none number per test; AUC = sleepy tests (KSS >= 7) against the rest, interval by resampling people:")
    results = {}
    for name, values in measures.items():
        results[name] = m = measure(values, kss, subjects)
        print(f"  {name:52} AUC {m['auc']:.2f} [{m['low']:.2f}, {m['high']:.2f}]   Spearman with KSS {m['spearman']:+.2f}   (n = {m['n']})")
    counts = {}
    for weights in chosen.values():
        key = ", ".join(f"{b} {weights[b]:.2f}" for b in BRANCHES)
        counts[key] = counts.get(key, 0) + 1
    print("weights chosen when one person is left out (times chosen):")
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key}   x{n}")
    print(f"best weights on all the data (in-sample, optimistic): {best_in_sample}, AUC {best_auc:.2f}")
    tp, fn, fp, tn = test_level_decision(np.array(measures["weighted fusion, default weights"], float), kss, subjects)
    print(
        f"whole-test decision from the mean fused score (cut-off chosen on the other people): sleepy tests found {tp} of {tp + fn}, "
        f"not-sleepy tests cleared {tn} of {tn + fp}   (confusion: tp {tp}, fn {fn}, fp {fp}, tn {tn})"
    )
    return rows, results


def plot(rows, path="docs/drozy_results.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kss = np.array([r["kss"] for r in rows])
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(kss))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (title, key) in zip(
        axes, (("EAR alone", "ear_mean"), ("CNN alone", "cnn_mean"), ("EOG alone", "eog_mean"), ("fused", "fused_mean")), strict=True
    ):
        ax.scatter(kss + jitter, [r[key] for r in rows], c=np.where(kss >= SLEEPY_KSS, "#e53935", "#4c8c4a"), s=28)
        ax.axvline(SLEEPY_KSS - 0.5, color="grey", ls=":")
        ax.set(title=title, xlabel="KSS (own rating of the test)", xticks=range(1, 10))
    axes[0].set_ylabel("mean score of the test\n(0 alert, 1 drowsy)")
    fig.suptitle(f"DROZY, {len(rows)} tests replayed: red = KSS {SLEEPY_KSS} or more", y=1.02)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def cli():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    jobs = drozy.tests()
    if not a.report_only:
        todo = [j for j in jobs if not os.path.exists(cache_path(*j))]
        print(f"{len(jobs) - len(todo)} of {len(jobs)} tests already extracted, {len(todo)} to do", flush=True)
        with ProcessPoolExecutor(a.workers) as pool:
            for done in pool.map(extract_one, todo):
                print("extracted", done, flush=True)
    done = [j for j in jobs if os.path.exists(cache_path(*j))]
    if len(done) < len(jobs):
        print(f"NOTE: only {len(done)} of {len(jobs)} tests are extracted; the numbers below are for those.")
    rows, _ = report([load(*j) for j in done])
    print("plot:", plot(rows))


if __name__ == "__main__":
    cli()
