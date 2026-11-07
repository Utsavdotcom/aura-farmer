"""preflop.py - starting-hand lookup (table built by tools/build_preflop.py)."""
import json
import os

RANKS = "6789TJQKA"


def canon_key(hand) -> str:
    """Same key for hands that differ only by suit names, e.g. 'AK9|T|7'."""
    groups = {}
    for c in hand:
        groups.setdefault(c & 3, []).append(c >> 2)
    parts = sorted((tuple(sorted(g, reverse=True)) for g in groups.values()),
                   key=lambda g: (-len(g), [-r for r in g]))
    return "|".join("".join(RANKS[r] for r in g) for g in parts)


def load_table() -> dict:
    """{key: [equity vs random hand, percentile 0..1, count]}"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preflop.json")
    with open(path) as f:
        return json.load(f)
