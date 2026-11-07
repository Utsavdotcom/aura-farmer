"""
build_preflop.py - precompute how strong every starting hand is (run once, offline).

For every distinct 5-card starting hand (suits don't matter, only which
cards share a suit), play many random opponent hands + boards and record the
win chance ("equity"). Writes data/preflop.json, which agent.py loads at start.

    python tools/build_preflop.py            # about a minute on a laptop
"""
import json
import os
import sys
from collections import Counter
from itertools import combinations
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fasteval import sample_showdowns  # noqa: E402
from preflop import canon_key  # noqa: E402

SAMPLES = 6000


def equity(args):
    key, hand = args
    rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
    me, opp, _ = sample_showdowns(hand, [], SAMPLES, rng)
    return key, float(np.mean((me > opp) + 0.5 * (me == opp)))


def main():
    counts, example = Counter(), {}
    for hand in combinations(range(36), 5):
        k = canon_key(hand)
        counts[k] += 1
        example.setdefault(k, hand)
    print(len(counts), "classes")
    with Pool() as pool:
        eqs = dict(pool.map(equity, list(example.items()), chunksize=64))
    # percentile: share of all starting hands (by count) that are weaker
    order = sorted(eqs, key=eqs.get)
    total, below, out = sum(counts.values()), 0, {}
    for k in order:
        out[k] = [round(eqs[k], 4), round((below + counts[k] / 2) / total, 4), counts[k]]
        below += counts[k]
    with open(os.path.join(ROOT, "data", "preflop.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("best", order[-5:], "worst", order[:5])


if __name__ == "__main__":
    main()
