"""
common.py - shared helpers for the "field" practice bots (tools/field/).

These bots imitate what other student teams will plausibly submit. They are
built only from the game rules, evaluator.py and the starter template.

Main pieces:
  * a numpy hand-score table keyed by an ADDITIVE card key, so thousands of
    5-card hands can be scored with a few array ops (Monte Carlo equity);
  * equity_vs_random(): Monte Carlo equity vs ONE random opponent hand;
  * best_score(): best made hand (2 hole + 3 board) on a 3/4/5 card board;
  * draw helpers (flush draw, straight-completing ranks);
  * action helpers that only ever return legal actions;
  * action-history helpers.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from evaluator import (FLUSH, HIGH_CARD, NUM_CARDS, ONE_PAIR, STRAIGHT, TWO_PAIR,  # noqa: F401
                       THREE_OF_A_KIND, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH,
                       category_of, score_five)
from game import Action, Street

# ---------------------------------------------------------------------------
# Additive score table
#   rank key = sum over the 5 cards of 5**rank   (at most 4 cards per rank)
#   suit key = sum over the 5 cards of 6**suit   (flush iff key == 5 * 6**s)
#   score    = TABLE[rank_key * 2 + is_flush]
# ---------------------------------------------------------------------------

RANK_W = np.array([5 ** (c >> 2) for c in range(NUM_CARDS)], dtype=np.int64)
SUIT_W = np.array([6 ** (c & 3) for c in range(NUM_CARDS)], dtype=np.int64)

_FLUSH_KEYS = {5 * 6 ** s for s in range(4)}
IS_FLUSH = np.zeros(5 * 6 ** 3 + 1, dtype=np.int64)
for _k in _FLUSH_KEYS:
    IS_FLUSH[_k] = 1


def _build_table() -> np.ndarray:
    table = np.zeros(2 * 5 ** 9, dtype=np.int32)
    # every multiset of 5 ranks with at most 4 copies of a rank
    for combo in combinations(range(9 + 4), 5):
        ranks = [c - i for i, c in enumerate(combo)]   # stars and bars -> non-decreasing ranks
        if max(ranks.count(r) for r in ranks) > 4:
            continue
        # non-flush representative: give each copy of a rank a different suit,
        # and make sure 5 distinct ranks are not all one suit
        cards, used = [], {}
        for r in ranks:
            s = used.get(r, 0)
            used[r] = s + 1
            cards.append(r * 4 + s)
        if len({c & 3 for c in cards}) == 1:
            cards[-1] = (cards[-1] >> 2) * 4 + 1
        key = sum(5 ** r for r in ranks)
        table[key * 2] = score_five(cards)
        if len(set(ranks)) == 5:
            table[key * 2 + 1] = score_five([r * 4 for r in ranks])
    return table


TABLE = _build_table()

HOLE_PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)   # (10, 2)
BOARD_TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)


def _scores_vec(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """holes (n,5) and boards (n,5) card arrays -> best Omaha score per row (n,)."""
    hr = RANK_W[holes]
    hs = SUIT_W[holes]
    br = RANK_W[boards]
    bs = SUIT_W[boards]
    pr = hr[:, HOLE_PAIRS[:, 0]] + hr[:, HOLE_PAIRS[:, 1]]                      # (n,10)
    ps = hs[:, HOLE_PAIRS[:, 0]] + hs[:, HOLE_PAIRS[:, 1]]
    tr = br[:, BOARD_TRIPLES[:, 0]] + br[:, BOARD_TRIPLES[:, 1]] + br[:, BOARD_TRIPLES[:, 2]]
    ts = bs[:, BOARD_TRIPLES[:, 0]] + bs[:, BOARD_TRIPLES[:, 1]] + bs[:, BOARD_TRIPLES[:, 2]]
    rk = pr[:, :, None] + tr[:, None, :]
    fl = IS_FLUSH[ps[:, :, None] + ts[:, None, :]]
    return TABLE[rk * 2 + fl].reshape(len(holes), -1).max(axis=1)


def equity_vs_random(hole, board, n: int, rng: np.random.Generator) -> float:
    """Monte Carlo equity of `hole` vs ONE uniformly random opponent hand,
    with a random runout of the missing board cards. Ties count half."""
    hole = list(hole)
    board = list(board)
    dead = set(hole) | set(board)
    deck = np.array([c for c in range(NUM_CARDS) if c not in dead], dtype=np.int64)
    need_board = 5 - len(board)
    k = 5 + need_board
    picks = np.argsort(rng.random((n, len(deck))), axis=1)[:, :k]
    drawn = deck[picks]                                   # (n, k)
    opp = drawn[:, :5]
    if need_board:
        boards = np.concatenate([np.tile(np.array(board, dtype=np.int64), (n, 1)), drawn[:, 5:]], axis=1) \
            if board else drawn[:, 5:]
    else:
        boards = np.tile(np.array(board, dtype=np.int64), (n, 1))
    mine = np.tile(np.array(hole, dtype=np.int64), (n, 1))
    s_me = _scores_vec(mine, boards)
    s_op = _scores_vec(opp, boards)
    return float(((s_me > s_op).sum() + 0.5 * (s_me == s_op).sum()) / n)


def best_score(hole, board) -> int:
    """Best score with exactly 2 hole + 3 board cards (board of 3, 4 or 5 cards)."""
    best = 0
    hr = [5 ** (c >> 2) for c in hole]
    hs = [6 ** (c & 3) for c in hole]
    br = [5 ** (c >> 2) for c in board]
    bs = [6 ** (c & 3) for c in board]
    trip = [(br[i] + br[j] + br[k], bs[i] + bs[j] + bs[k])
            for i, j, k in combinations(range(len(board)), 3)]
    for i, j in combinations(range(5), 2):
        r2 = hr[i] + hr[j]
        s2 = hs[i] + hs[j]
        for tr, ts in trip:
            s = int(TABLE[(r2 + tr) * 2 + (1 if (s2 + ts) in _FLUSH_KEYS else 0)])
            if s > best:
                best = s
    return best


def board_score(board) -> int:
    """Best 3-card-ish 'board only' strength proxy: max category reachable by the board
    alone is not defined in Omaha, so return the board's pairing level (0 none, 1 pair, 2 trips+)."""
    ranks = [c >> 2 for c in board]
    m = max(ranks.count(r) for r in ranks) if ranks else 0
    return 0 if m <= 1 else (1 if m == 2 else 2)


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------

def flush_draw(hole, board) -> int:
    """0 = none, 1 = flush draw (2+ hole and exactly 2 board of a suit, board not complete),
    2 = nut-ish flush draw (same, and we hold the ace or king of that suit)."""
    if len(board) >= 5:
        return 0
    best = 0
    for s in range(4):
        b = sum(1 for c in board if (c & 3) == s)
        h = [c >> 2 for c in hole if (c & 3) == s]
        if b == 2 and len(h) >= 2:
            best = max(best, 2 if max(h) >= 7 else 1)
    return best


_STRAIGHTS = [set(range(lo, lo + 5)) for lo in range(5)] + [{8, 0, 1, 2, 3}]


def straight_outs(hole, board) -> int:
    """Number of distinct ranks that would give us a straight next card
    (using exactly 2 hole + 3 board). 0 if the board is complete."""
    if len(board) >= 5:
        return 0
    hranks = {c >> 2 for c in hole}
    branks = [c >> 2 for c in board]
    outs = set()
    for r in range(9):
        nb = set(branks) | {r}
        for st in _STRAIGHTS:
            need_board = st & nb
            # need >=3 board ranks in the straight and the other 2 from hole
            if len(need_board) >= 3:
                for trio in combinations(sorted(need_board), 3):
                    if r not in trio:
                        continue
                    rest = st - set(trio)
                    if rest <= hranks:
                        outs.add(r)
                        break
            if r in outs:
                break
    # don't count ranks that already make a straight without the new card
    made = set()
    for st in _STRAIGHTS:
        bset = set(branks)
        if len(st & bset) >= 3:
            for trio in combinations(sorted(st & bset), 3):
                if set(st) - set(trio) <= hranks:
                    made = {1}
    return 0 if made else len(outs)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def safe_default(obs) -> Action:
    """Check if free, else fold (and only legal actions)."""
    legal = obs.legal_actions
    if obs.to_call == 0 and Action.CHECK_CALL in legal:
        return Action.CHECK_CALL
    if Action.FOLD in legal:
        return Action.FOLD
    return legal[0] if legal else Action.CHECK_CALL


def legalize(obs, want: Action) -> Action:
    """Turn a wanted action into a legal one."""
    legal = obs.legal_actions
    if want in legal:
        return want
    if want in (Action.RAISE_POT, Action.RAISE_HALF):
        other = Action.RAISE_HALF if want == Action.RAISE_POT else Action.RAISE_POT
        if other in legal:
            return other
        return Action.CHECK_CALL if Action.CHECK_CALL in legal else safe_default(obs)
    if want == Action.FOLD:  # folding when checking is free is silly
        return Action.CHECK_CALL
    return safe_default(obs)


def pot_odds(obs) -> float:
    if obs.to_call <= 0:
        return 0.0
    return obs.to_call / float(obs.pot + obs.to_call)


def can_raise(obs) -> bool:
    return Action.RAISE_HALF in obs.legal_actions or Action.RAISE_POT in obs.legal_actions


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def street_actions(obs, street=None):
    st = obs.street if street is None else street
    return [a for a in obs.action_history if a.street == st]


def opp_checked_to_me(obs) -> bool:
    """True if the opponent has passed to us: to_call == 0 and either the opponent
    already checked this street, or (we act first postflop) the opponent checked
    on the previous street's last action."""
    if obs.to_call != 0:
        return False
    cur = street_actions(obs)
    opp_cur = [a for a in cur if not a.is_me]
    if opp_cur:
        return opp_cur[-1].action == Action.CHECK_CALL
    if obs.street == Street.PREFLOP:
        return False
    prev = street_actions(obs, Street(obs.street - 1))
    if not prev:
        return False
    # previous street was passive (nobody bet or raised): the opponent checked / limped
    return all(a.action == Action.CHECK_CALL for a in prev)


def last_opp_action(obs):
    cur = [a for a in street_actions(obs) if not a.is_me]
    return cur[-1] if cur else None


def opp_bet_fraction(obs) -> float:
    """Size of what we face relative to the pot before the opponent's bet:
    about 0.5 for a half-pot bet/raise, about 1.0 for a pot-size one."""
    if obs.to_call <= 0:
        return 0.0
    base = obs.pot - obs.to_call
    return obs.to_call / float(base) if base > 0 else 1.0


# Warm numpy code paths now so the first real decision is not slow.
equity_vs_random((0, 5, 10, 15, 20), (), 50, np.random.default_rng(0))
equity_vs_random((0, 5, 10, 15, 20), (24, 28, 32), 50, np.random.default_rng(0))
