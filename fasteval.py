"""
fasteval.py - fast numpy hand evaluation and equity for Short Deck PLO5.

Scores are identical to evaluator.score_five / omaha_score (tested in
test_agent.py), but many hands are scored at once with numpy.

Trick: a 5-card score only depends on (a) which ranks it holds and how many
of each, and (b) whether all 5 cards share a suit. So each card becomes
  rank key  5 ** rank        (adding 5 keys gives a unique "rank count" index)
  suit bit  1 << suit        (AND of 5 suit bits is non-zero only for a flush)
and the score is BASE[index] or FLUSH[index].
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np

from evaluator import _score_ranks

POW5 = np.array([5 ** r for r in range(9)], dtype=np.int32)
CARD_RK = POW5[np.arange(36) >> 2]                       # rank key per card
CARD_SB = (1 << (np.arange(36) & 3)).astype(np.int32)    # suit bit per card

_SIZE = 5 ** 9
BASE = np.zeros(_SIZE, dtype=np.int32)
FLUSH = np.zeros(_SIZE, dtype=np.int32)
for _ranks in combinations_with_replacement(range(9), 5):
    if max(_ranks.count(r) for r in _ranks) > 4:
        continue
    _idx = sum(5 ** r for r in _ranks)
    _desc = tuple(sorted(_ranks, reverse=True))
    BASE[_idx] = _score_ranks(_desc, False)
    if len(set(_ranks)) == 5:
        FLUSH[_idx] = _score_ranks(_desc, True)

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)
TRIPLES4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)  # (4, 3)
TRIPLES3 = np.array([[0, 1, 2]], dtype=np.int64)                      # (1, 3)
TRIPLES = {3: TRIPLES3, 4: TRIPLES4, 5: TRIPLES5}


def _groups(cards: np.ndarray, idx: np.ndarray):
    """cards (..., n) -> rank-key sums and suit-bit ANDs for each index group (..., g)."""
    sel = cards[..., idx]                       # (..., g, k)
    rk = CARD_RK[sel].sum(axis=-1)
    sb = np.bitwise_and.reduce(CARD_SB[sel], axis=-1)
    return rk, sb


def best_scores(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Best Omaha score for each row. holes (N, 5); boards (N, b) or (b,), b in 3..5."""
    b = boards.shape[-1]
    hr, hs = _groups(holes, PAIRS)                     # (N, 10)
    br, bs = _groups(boards, TRIPLES[b])               # (N, t) or (t,)
    if br.ndim == 1:
        br, bs = br[None, :], bs[None, :]
    idx = hr[:, :, None] + br[:, None, :]
    flush = (hs[:, :, None] & bs[:, None, :]) != 0
    sc = np.where(flush, FLUSH[idx], BASE[idx])
    return sc.reshape(sc.shape[0], -1).max(axis=1)


def sample_showdowns(hole, board, n: int, rng: np.random.Generator):
    """Deal n random opponent hands and board runouts.

    Returns (my_final, opp_final, opp_now):
      my_final, opp_final  final 5-card-board scores per sample
      opp_now              opponent's made-hand score on the CURRENT board
                           (None preflop), used to guess which hands bet/call.
    """
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    dead = np.zeros(36, dtype=bool)
    dead[hole] = True
    dead[board] = True
    deck = np.flatnonzero(~dead)
    need = 5 + (5 - len(board))
    picks = np.argsort(rng.random((n, len(deck))), axis=1)[:, :need]
    drawn = deck[picks]                                 # (n, need)
    opp = drawn[:, :5]
    full_board = np.concatenate([np.broadcast_to(board, (n, len(board))), drawn[:, 5:]], axis=1)
    if len(board) == 5:
        my_final = np.full(n, best_scores(hole[None, :], board)[0])
    else:
        my_final = best_scores(np.broadcast_to(hole, (n, 5)), full_board)
    opp_final = best_scores(opp, full_board)
    opp_now = best_scores(opp, board) if len(board) >= 3 else None
    return my_final, opp_final, opp_now
