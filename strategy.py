"""
strategy.py - the brain of the bot. agent.py calls these three functions.

How a decision is made:
  1. Guess how often we win ("equity"):
       preflop  -> lookup table (data/preflop.json)
       postflop -> play out a few hundred random opponent hands and boards
  2. Narrow the opponent's possible hands using what they did this hand
     (a raise from a bot that rarely raises means a strong hand).
  3. Work out the expected chips for each legal move (fold / check-call /
     half-pot raise / pot raise), using how often THIS opponent folds.
  4. Pick the move worth the most chips.

Everything we learn about the opponent lives in `memory`, which is wiped
every half (the tournament runs each half in a fresh server anyway).
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from fasteval import sample_showdowns
from game import Action, AgentConfig, HandResult, Observation, Street
from preflop import canon_key, load_table

# ---------------------------------------------------------------------------
# Startup: load the preflop table and build helper curves
# ---------------------------------------------------------------------------

PF = load_table()                         # key -> [equity, percentile, count]

# TOP_EQ(f): average equity (vs a random hand) of the best f share of hands.
# Used to guess our equity against a narrowed preflop range:
#   eq_vs_range ~= our_eq + 0.5 - TOP_EQ(f)
_rows = sorted(PF.values(), key=lambda r: -r[0])
_w = np.array([r[2] for r in _rows], dtype=float)
_e = np.array([r[0] for r in _rows], dtype=float)
_cum_w = np.cumsum(_w) / _w.sum()
_cum_eq = np.cumsum(_w * _e) / np.cumsum(_w)


def top_eq(f: float) -> float:
    return float(np.interp(min(max(f, 0.01), 1.0), _cum_w, _cum_eq))


# Warm numpy up so the first real decision is not slow.
sample_showdowns([0, 5, 10, 20, 30], [1, 2, 3], 64, np.random.default_rng(0))

# ---------------------------------------------------------------------------
# Tunable settings
# ---------------------------------------------------------------------------

TIME_BUDGET = 5.5          # seconds of our own thinking per half (limit is 8)
MAX_SAMPLES = 500
MIN_SAMPLES = 48
SEC_PER_SAMPLE = 1.4e-6    # measured cost of one simulated showdown
DECISIONS_PER_HAND = 3.0   # rough guess, used to spread the time budget

REALIZE_IP = 0.97          # share of equity we keep when acting last
REALIZE_OOP = 0.88         # ... when acting first
LOW_WEIGHT_FLOP = 0.20     # weight of "unlikely" hands in a narrowed range
LOW_WEIGHT_TURN = 0.12     #   (they could be draws or bluffs)
LOW_WEIGHT_RIVER = 0.05
PF_K_MULT = 1.4            # opponent raise rate -> share of hands it raises with
POST_K_MULT = 1.4
K_FLOOR = 0.12
HONESTY_PRIOR = 6          # showdowns before we fully trust the honesty check

# Mixed strategy: sometimes pick a close-to-best move so we are harder to read.
MIX_TOL = 0.06             # moves within this share of the pot of the best are "close"
MIX_TEMP = 0.02            # randomness among close moves (bigger = more random)
BLUFF_MIX = 0.10           # chance to bet a weak hand when we would check
BLUFF_MAX_LOSS = 0.20      #   ... only if it costs at most this share of the pot
SLOWPLAY_MIX = 0.12        # chance to just check/call a very strong hand
SLOWPLAY_MAX_LOSS = 0.15   #   ... only if it costs at most this share of the pot


# ---------------------------------------------------------------------------
# Opponent statistics (counts with a small "prior" so early hands are sane)
# ---------------------------------------------------------------------------

def _new_stats() -> dict:
    return {
        "pf_dec": 10.0, "pf_raise": 2.5,     # opponent preflop raise rate
        "post_dec": 10.0, "post_raise": 2.5, # opponent postflop raise/bet rate
        "faced": 10.0, "folded": 3.0,        # opponent fold rate when we bet/raise
        "pf_faced": 6.0, "pf_folded": 1.5,   # same, preflop only
        "my_faced": 10.0, "my_folded": 4.0,  # how often WE fold to its postflop bets
        # Showdown check: are the hands it raises with really stronger?
        # [sum, count] of hand strength when it raised / did not raise.
        "sd_pf_r": [0.0, 0], "sd_pf_n": [0.0, 0],
        "sd_post_r": [0.0, 0], "sd_post_n": [0.0, 0],
    }


def _rate(st: dict, num: str, den: str) -> float:
    return st[num] / st[den]


# ---------------------------------------------------------------------------
# The three entry points
# ---------------------------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    memory["num_hands"] = int(config.num_hands)
    memory["t_used"] = 0.0
    memory["stats"] = _new_stats()
    memory["rng"] = random.Random(12345)
    memory["np_rng"] = np.random.default_rng(12345)
    memory["hand"] = None        # (hand_index, is_button)
    memory["hist"] = []          # last seen action history of the current hand


def choose_action(obs: Observation, memory: dict) -> Action:
    start = time.perf_counter()
    try:
        if "stats" not in memory:            # safety: new_half was never called
            new_half(AgentConfig(), memory)
        memory["hand"] = (obs.hand_index, obs.is_button)
        action = _decide(obs, memory)
        # remember the history plus our own move (used by on_hand_end)
        memory["hist"] = [(a.street, a.is_me, a.action) for a in obs.action_history]
        memory["hist"].append((obs.street, True, action))
    except Exception:
        action = None
    memory["t_used"] = memory.get("t_used", 0.0) + time.perf_counter() - start
    return _safe(action, obs)


def on_hand_end(result: HandResult, memory: dict) -> None:
    start = time.perf_counter()
    try:
        _learn(result, memory)
    except Exception:
        pass
    memory["hist"] = []
    memory["t_used"] = memory.get("t_used", 0.0) + time.perf_counter() - start


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def _safe(action, obs: Observation) -> Action:
    """Always return a legal action."""
    if action in obs.legal_actions:
        return action
    if Action.CHECK_CALL in obs.legal_actions and obs.to_call == 0:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return obs.legal_actions[0]


# ---------------------------------------------------------------------------
# Learning about the opponent
# ---------------------------------------------------------------------------

def _learn(result: HandResult, memory: dict) -> None:
    st = memory["stats"]
    hand = memory.get("hand")
    hist = list(memory.get("hist") or [])
    have_hist = hand is not None and hand[0] == result.hand_index and hist

    if not have_hist:
        # We never acted: we were big blind and the button folded at once.
        if not result.went_to_showdown and result.winner == "me":
            st["pf_dec"] += 1
        return

    is_button = hand[1]
    hist.extend(_hand_ending_opp_actions(hist, result))

    # Walk through the hand, tracking who owes chips, to see what the
    # opponent faced each time it acted.
    street = Street.PREFLOP
    owes_opp = 1 if not is_button else 0     # preflop: the button (SB) owes 1
    owes_me = 0
    first_pf_opp = True
    for s, is_me, act in hist:
        if s != street:
            street, owes_opp, owes_me = s, 0, 0
        if is_me:
            if owes_me and street != Street.PREFLOP:
                st["my_faced"] += 1
                st["my_folded"] += act == Action.FOLD
            owes_me = 0
            if act in (Action.RAISE_HALF, Action.RAISE_POT):
                owes_opp = 1
            elif act == Action.CHECK_CALL:
                owes_opp = 0
            continue
        raised = act in (Action.RAISE_HALF, Action.RAISE_POT)
        owes_me = 1 if raised else 0
        if street == Street.PREFLOP:
            st["pf_dec"] += 1
            st["pf_raise"] += raised
            # The button's first preflop action only faces the blind, not our bet.
            facing_bet = owes_opp and not (first_pf_opp and not is_button)
            first_pf_opp = False
            if facing_bet:
                st["pf_faced"] += 1
                st["pf_folded"] += act == Action.FOLD
        else:
            st["post_dec"] += 1
            st["post_raise"] += raised
            if owes_opp:
                st["faced"] += 1
                st["folded"] += act == Action.FOLD
        owes_opp = 0

    if result.went_to_showdown and result.opp_hole and len(result.board) == 5:
        _learn_showdown(result, hist, memory)


def _hand_ending_opp_actions(hist: list, result: HandResult) -> list:
    """The opponent's actions after our last decision, which we never see in
    an Observation. `hist` ends with our own last action.
      * no showdown and we won        -> it folded
      * showdown after our raise      -> it called (a re-raise would have
                                         given us another decision)
      * showdown after we checked first on a postflop street -> it checked
        behind (a bet would have given us another decision)
    Otherwise our check/call closed the betting and it did nothing more."""
    if not hist or not hist[-1][1]:
        return []
    street, _, last = hist[-1]
    if not result.went_to_showdown:
        return [(street, False, Action.FOLD)] if result.winner == "me" else []
    if last in (Action.RAISE_HALF, Action.RAISE_POT):
        return [(street, False, Action.CHECK_CALL)]
    if last == Action.CHECK_CALL and street != Street.PREFLOP:
        opp_acted_this_street = any(s == street and not me for s, me, _ in hist[:-1])
        if not opp_acted_this_street:
            return [(street, False, Action.CHECK_CALL)]
    return []


def _learn_showdown(result: HandResult, hist: list, memory: dict) -> None:
    st = memory["stats"]
    opp_raised = {s for s, is_me, a in hist
                  if not is_me and a in (Action.RAISE_HALF, Action.RAISE_POT)}
    pf_pct = PF[canon_key(result.opp_hole)][1]
    key = "sd_pf_r" if Street.PREFLOP in opp_raised else "sd_pf_n"
    st[key][0] += pf_pct
    st[key][1] += 1
    # final strength: win chance of its hand against random hands on this board
    opp, other, _ = sample_showdowns(result.opp_hole, result.board, 128, memory["np_rng"])
    strength = float(np.mean((opp > other) + 0.5 * (opp == other)))
    key = "sd_post_r" if opp_raised - {Street.PREFLOP} else "sd_post_n"
    st[key][0] += strength
    st[key][1] += 1


def _honesty(st: dict, prefix: str, honest_gap: float) -> float:
    """1.0 = its raises mean strong hands (default until we see otherwise);
    0.0 = it raises with any hand (raises tell us nothing)."""
    (sr, nr), (sn, nn) = st[prefix + "_r"], st[prefix + "_n"]
    if nr < 3 or nn < 3:
        return 1.0
    gap = sr / nr - sn / nn
    raw = min(max(gap / honest_gap, 0.0), 1.0)
    w = min(nr, nn) / (min(nr, nn) + HONESTY_PRIOR)
    return w * raw + (1.0 - w) * 1.0


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def _raise_ks(st: dict) -> tuple[float, float]:
    """Share of its hands the opponent raises with (preflop, postflop)."""
    pf_k = min(max(PF_K_MULT * _rate(st, "pf_raise", "pf_dec"), K_FLOOR), 0.97)
    post_k = min(max(POST_K_MULT * _rate(st, "post_raise", "post_dec"), K_FLOOR), 0.97)
    # If its raises are not backed by strong hands, believe them less.
    pf_k = 1.0 - (1.0 - pf_k) * _honesty(st, "sd_pf", 0.4)
    post_k = 1.0 - (1.0 - post_k) * _honesty(st, "sd_post", 0.25)
    return pf_k, post_k


def _range_fraction(obs: Observation, st: dict) -> float:
    """Roughly what share of all hands the opponent still holds, given what
    it did this hand. 1.0 = any hand; 0.2 = only its best 20%."""
    pf_k, post_k = _raise_ks(st)
    call_k = min(max(1.0 - _rate(st, "folded", "faced"), 0.4), 1.0)
    f = 1.0
    my_bet_pending = False
    for a in obs.action_history:
        if a.is_me:
            my_bet_pending = a.action in (Action.RAISE_HALF, Action.RAISE_POT)
            continue
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            if a.street == Street.PREFLOP:
                f *= pf_k
            elif a.street == obs.street:
                f *= post_k
            else:
                f *= math.sqrt(post_k)
        elif a.action == Action.CHECK_CALL and my_bet_pending and a.street != Street.PREFLOP:
            f *= call_k
        my_bet_pending = False
    return max(f, 0.05)


def _num_samples(obs: Observation, memory: dict) -> int:
    left_hands = max(memory["num_hands"] - obs.hand_index, 1)
    left_time = TIME_BUDGET - memory["t_used"]
    per_decision = left_time / (left_hands * DECISIONS_PER_HAND)
    n = int((per_decision - 1.5e-4) / SEC_PER_SAMPLE)
    if left_time < 0.3:
        return 0
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


def _invest(obs: Observation, raise_to: int) -> tuple[int, int]:
    """(chips we add, chips opponent must add to call) for a raise to raise_to,
    capped by what the opponent can actually match."""
    opp_call = min(raise_to - obs.opp_bet, obs.opp_stack)
    ours = min(raise_to - obs.my_bet, obs.to_call + opp_call)
    return ours, opp_call


def _decide(obs: Observation, memory: dict) -> Action:
    st = memory["stats"]
    f = _range_fraction(obs, st)
    if obs.street == Street.PREFLOP:
        p_all = _rate(st, "pf_folded", "pf_faced")
    else:
        p_all = _rate(st, "folded", "faced")
    # The opponent folds the weakest p_all share of ALL hands. If its range is
    # already narrowed to its best f, it folds less of what it still holds.
    keep = min(f, max(1.0 - p_all, 0.1))
    p_fold = 1.0 - keep / f
    if obs.street == Street.PREFLOP:
        eq = PF[canon_key(obs.hole_cards)][0]
        e_now, e_called = eq + 0.5 - top_eq(f), eq + 0.5 - top_eq(keep)
    else:
        n = _num_samples(obs, memory)
        if n == 0:
            return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        low_scale = min(1.0, _rate(st, "post_raise", "post_dec") / 0.3)
        eq = _postflop_equity(obs, memory, n, low_scale)
        e_now, e_called = eq(f), eq(keep)
    ev = _action_evs(obs, e_now, e_called, p_fold)
    return _mix(obs, ev, e_now, memory["rng"])


def _postflop_equity(obs, memory, n, low_scale):
    """Simulate n showdowns once; return eq(frac): our win chance
    against the opponent's best `frac` of hands (other hands weighted low)."""
    my, opp, now = sample_showdowns(obs.hole_cards, obs.board, n, memory["np_rng"])
    win = (my > opp) + 0.5 * (my == opp)
    # rank the opponent's sampled hands by how good they are right now
    pct = np.empty(n)
    pct[np.argsort(now, kind="stable")] = (np.arange(n) + 0.5) / n
    low = {Street.FLOP: LOW_WEIGHT_FLOP, Street.TURN: LOW_WEIGHT_TURN}.get(obs.street, LOW_WEIGHT_RIVER)
    low *= low_scale      # opponents that rarely bet rarely bluff

    def eq(frac: float) -> float:
        w = np.where(pct >= 1.0 - frac, 1.0, low)
        return float((win * w).sum() / w.sum())

    return eq


def _action_evs(obs: Observation, e_now: float, e_called: float, p_fold: float) -> dict:
    """Expected chips of each legal action."""
    pot = obs.pot
    # Acting last after the flop = the button. Preflop the button acts first
    # but still has position for the rest of the hand.
    realize = REALIZE_IP if obs.is_button else REALIZE_OOP
    if obs.street == Street.RIVER:
        realize = 1.0

    ev = {}
    if Action.FOLD in obs.legal_actions:
        ev[Action.FOLD] = 0.0
    if obs.to_call == 0:
        ev[Action.CHECK_CALL] = e_now * pot * realize
    else:
        call = min(obs.to_call, obs.my_stack)
        allin = call >= obs.my_stack or obs.opp_stack == 0
        r = 1.0 if allin else realize
        ev[Action.CHECK_CALL] = e_now * r * (pot + call) - call

    for a in (Action.RAISE_HALF, Action.RAISE_POT):
        if a not in obs.legal_actions:
            continue
        ours, theirs = _invest(obs, obs.raise_to[a])
        final_pot = pot + ours + theirs
        allin = ours >= obs.my_stack or theirs >= obs.opp_stack
        r = 1.0 if allin else realize
        # Opponent might re-raise instead of calling; shave a little for that.
        if_called = e_called * r * final_pot - ours
        ev[a] = p_fold * pot + (1.0 - p_fold) * if_called - 0.02 * ours

    return ev


def _mix(obs: Observation, ev: dict, e_now: float, rng: random.Random) -> Action:
    """Pick a move: usually the best one, sometimes a close one, so the
    opponent can't read our hand from our action. Every random choice is
    capped so it never costs much."""
    best = max(ev, key=ev.get)
    scale = max(obs.pot, 2 * obs.big_blind)
    raises = [a for a in (Action.RAISE_HALF, Action.RAISE_POT) if a in ev]

    # Bluff: bet a weak hand now and then instead of checking.
    if (best == Action.CHECK_CALL and obs.to_call == 0 and raises
            and e_now < 0.35 and rng.random() < BLUFF_MIX):
        if ev[Action.RAISE_HALF if Action.RAISE_HALF in ev else raises[0]] >= ev[best] - BLUFF_MAX_LOSS * scale:
            return Action.RAISE_HALF if Action.RAISE_HALF in ev else raises[0]

    # Slowplay: just check/call a very strong hand now and then (not on the
    # river, where there is no later street to win more).
    if (best in raises and obs.street != Street.RIVER and e_now > 0.75
            and rng.random() < SLOWPLAY_MIX
            and ev[Action.CHECK_CALL] >= ev[best] - SLOWPLAY_MAX_LOSS * scale):
        return Action.CHECK_CALL

    # Otherwise choose among near-best moves, weighted toward the best.
    close = [a for a in ev if ev[a] >= ev[best] - MIX_TOL * scale]
    if len(close) == 1:
        return best
    weights = [math.exp((ev[a] - ev[best]) / (MIX_TEMP * scale)) for a in close]
    return rng.choices(close, weights)[0]
