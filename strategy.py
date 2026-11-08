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
import zlib

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

# Simulated showdowns per postflop decision. Fixed (not time-based) so the
# same deal and history always give the same decision on any machine.
# More samples when facing a bet: those decisions put the most chips at risk.
SAMPLES = {Street.FLOP: (200, 300), Street.TURN: (200, 300), Street.RIVER: (150, 250)}  # (no bet, facing a bet)

# Time safety. The judge charges each request's time minus a free 1 ms.
# Measured on an idle core, that is our own timer plus about 0.03 ms per
# call; 0.05 ms per call is added as a margin. (On a contended core the
# judge also charges time spent waiting for the CPU outside our code: in one
# measurement our timer read 4.5 s while the judge charged 6.4 s.)
OVERHEAD_PER_CALL = 5e-5
PROJECTED_LIMIT = 5.0      # if this half is projected to use more, switch to cheap mode
CHEAP_SAMPLES = 48         # samples per decision in cheap mode
CLOSE_GAP = 0.03           # if our top two moves are within this share of the pot ...
CLOSE_EXTRA = 400          # ... draw this many more samples (same seeded generator) and decide again
HARD_LIMIT = 7.0           # past this, play check/fold after the flop (no simulation)
MIN_HANDS_TO_PROJECT = 50  # don't extrapolate from the first few hands

REALIZE_IP = 0.97          # share of equity we keep when acting last
REALIZE_OOP = 0.88         # ... when acting first
LOW_WEIGHT_FLOP = 0.20     # weight of "unlikely" hands in a narrowed range
LOW_WEIGHT_TURN = 0.12     #   (they could be draws or bluffs)
LOW_WEIGHT_RIVER = 0.05
PF_K_MULT = 1.4            # opponent raise rate -> share of hands it raises with
POST_K_MULT = 1.4
K_FLOOR = 0.12
HONESTY_PRIOR = 6          # showdowns before we fully trust the honesty check
# A half-pot bet after we check is mostly a cheap stab (measured against the
# actual cards: we were ahead far more often than the narrowed range said),
# so it narrows the range less than other bets: 0 = not at all, 1 = as usual.
HALF_STAB_NARROW = 0.0
# Facing a half-pot bet after our check from an opponent that bets often when
# we check to it on this street (raw rate >= the gate over >= STAB_GATE_MIN
# chances this half), decide FOLD vs CALL with our equity against its WHOLE
# range. Raise EVs keep today's narrowed equity, and if a raise is already
# the best move nothing changes: the whole-range read only turns folds into
# calls. Gate "any": all its bets count (threshold in the sim-seed gap between
# nit/tag/F1 <= 42% and lag/learner/F4/PR #2 >= 55%). Gate "half": only its
# half-pot bets count (gap: nit/tag/F1 <= 18.4%, learner/lag/F4/PR #2 >= 27.3%).
STAB_GATE_KIND = "any"
STAB_GATE_RATES = {"any": 0.50, "half": 0.23}
STAB_GATE_MIN = 20
# River, after our check, facing its half-pot bet: if _mix would fold, call
# instead when our smoothed showdown win share after calling such bets this
# half (prior SD_CALL_PRIOR worth SD_CALL_WEIGHT showdowns) beats the 25% pot
# odds. Measured on sim seeds against the real cards: our folds there were
# wrong more often than not vs jish4, PR #2, F4 and learner.
SD_CALL_PRIOR = 0.30
SD_CALL_WEIGHT = 20
SD_CALL_ODDS = 0.25
# Same idea when it LEADS the river with a half-pot bet (we are the button and
# have not acted on the river): its own counter and prior weight.
SD_LEAD_PRIOR = 0.30
SD_LEAD_WEIGHT = 10
# Turn/river, no bet facing us, and _mix chose to CHECK: bet instead if a bet
# is now the best move when valued with this opponent's MEASURED fold rate to
# our first-in bets on this street at this size (learned this half: prior
# BET_FOLD_PRIOR worth BET_FOLD_WEIGHT bets, capped at BET_FOLD_CAP) and our
# equity against only the best (1 - fold rate) share of its range (it folds
# its weakest hands). Only checks become bets: every bet, its size, and every
# decision facing a bet stay as before. The model's own fold chance for these
# bets is ~1-4%; measured rates were 30-60% (branching replay, sim0-7).
BET_FOLD_PRIOR = 0.30
BET_FOLD_WEIGHT = 20
BET_FOLD_CAP = 0.75
BET_FOLD_STREETS = (Street.TURN, Street.RIVER)

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
        # When we check to it, per street: raw [bets, half-pot bets, chances].
        "chk_bet": {s: [0, 0, 0] for s in (Street.FLOP, Street.TURN, Street.RIVER)},
        # Showdowns after we check the river, it bets half pot and we call:
        # raw [sum of our win shares (1 / 0.5 / 0), count].
        "riv_sd": [0.0, 0],
        # Same after it leads the river with a half-pot bet and we call.
        "riv_lead_sd": [0.0, 0],
        # Our first-in bets (no bet facing us) on the turn/river, per street and
        # size: raw [bets, times it folded right away].
        "bet_fold": {s: {Action.RAISE_HALF: [0, 0], Action.RAISE_POT: [0, 0]} for s in BET_FOLD_STREETS},
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
    memory["pots"] = None        # (pot after our last move, chips the opponent needs to call it)
    memory["blinds"] = int(config.small_blind) + int(config.big_blind)
    memory["last_learned"] = -1  # hand_index of the last hand learned from
    memory["calls"] = 0          # choose_action calls this half (for the overhead estimate)
    memory["mode_counts"] = {"full": 0, "cheap": 0, "none": 0}   # postflop sample modes used


def choose_action(obs: Observation, memory: dict) -> Action:
    start = time.perf_counter()
    memory["calls"] = memory.get("calls", 0) + 1
    try:
        if "stats" not in memory:            # safety: new_half was never called
            new_half(AgentConfig(), memory)
        memory["hand"] = (obs.hand_index, obs.is_button)
        action = _decide(obs, memory)
    except Exception:
        action = None
    action = _safe(action, obs)
    try:
        # Remember the history plus the move we actually return (used by on_hand_end).
        hist = [(a.street, a.is_me, a.action) for a in obs.action_history]
        hist.append((obs.street, True, action))
        memory["hand"] = (obs.hand_index, obs.is_button)
        memory["hist"], memory["pots"] = hist, _pots_after(obs, action)
    except Exception:
        memory["hist"], memory["pots"] = [], None
    memory["t_used"] = memory.get("t_used", 0.0) + time.perf_counter() - start
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    start = time.perf_counter()
    try:
        # Each hand is learned from once, even if it were delivered twice.
        if result.hand_index > memory.get("last_learned", -1):
            memory["last_learned"] = result.hand_index
            _learn(result, memory)
    except Exception:
        pass
    memory["hist"], memory["pots"] = [], None
    memory["t_used"] = memory.get("t_used", 0.0) + time.perf_counter() - start


def _pots_after(obs: Observation, action: Action) -> tuple[int, int]:
    """(pot right after our move, chips the opponent must add to call it)."""
    if action in (Action.RAISE_HALF, Action.RAISE_POT):
        raise_to = obs.raise_to[action]
        ours = raise_to - obs.my_bet
        opp_call = min(raise_to - obs.opp_bet, obs.opp_stack)
    elif action == Action.CHECK_CALL:
        ours, opp_call = min(obs.to_call, obs.my_stack), 0
    else:
        ours, opp_call = 0, 0
    return obs.pot + ours, opp_call


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
        # We never acted: we were big blind and the button folded at once
        # (the pot then holds only the blinds).
        if (not result.went_to_showdown and result.winner == "me"
                and result.pot == memory.get("blinds", 3)):
            st["pf_dec"] += 1
        return

    is_button = hand[1]
    hist.extend(_hand_ending_opp_actions(hist, result, memory.get("pots")))

    # Walk through the hand, tracking who owes chips, to see what the
    # opponent faced each time it acted.
    street = Street.PREFLOP
    owes_opp = 1 if not is_button else 0     # preflop: the button (SB) owes 1
    owes_me = 0
    first_pf_opp = True
    checked_to_opp = False                   # our last move this street was a check (nothing owed)
    first_in_bet = None                      # our turn/river bet with nothing owed, awaiting its answer
    for s, is_me, act in hist:
        if s != street:
            street, owes_opp, owes_me = s, 0, 0
            checked_to_opp = False
            first_in_bet = None
        if is_me:
            checked_to_opp = (street != Street.PREFLOP and act == Action.CHECK_CALL and not owes_me)
            first_in_bet = (act if street in BET_FOLD_STREETS and not owes_me
                            and act in (Action.RAISE_HALF, Action.RAISE_POT) else None)
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
            if checked_to_opp and not owes_opp:
                cb = st["chk_bet"][street]
                cb[0] += raised
                cb[1] += act == Action.RAISE_HALF
                cb[2] += 1
            checked_to_opp = False
            if owes_opp:
                st["faced"] += 1
                st["folded"] += act == Action.FOLD
            if first_in_bet is not None:
                bf = st["bet_fold"][street][first_in_bet]
                bf[0] += 1
                bf[1] += act == Action.FOLD
                first_in_bet = None
        owes_opp = 0

    if result.went_to_showdown and result.opp_hole and len(result.board) == 5:
        _learn_showdown(result, hist, memory)
        share = {"me": 1.0, "split": 0.5}.get(result.winner, 0.0)
        if _called_river_half_stab(hist):
            st["riv_sd"][0] += share
            st["riv_sd"][1] += 1
        if _called_river_half_lead(hist):
            st["riv_lead_sd"][0] += share
            st["riv_lead_sd"][1] += 1


def _called_river_half_stab(hist: list) -> bool:
    """On the river we checked, it bet half pot and we called (hist includes
    the actions inferred at hand end)."""
    r = [(is_me, a) for s, is_me, a in hist if s == Street.RIVER]
    for i in range(len(r) - 2):
        our_check = r[i] == (True, Action.CHECK_CALL) and (i == 0 or r[i - 1] == (False, Action.CHECK_CALL))
        if our_check and r[i + 1] == (False, Action.RAISE_HALF) and r[i + 2] == (True, Action.CHECK_CALL):
            return True
    return False


def _called_river_half_lead(hist: list) -> bool:
    """On the river it bet half pot before we acted and we called."""
    r = [(is_me, a) for s, is_me, a in hist if s == Street.RIVER]
    return len(r) >= 2 and r[0] == (False, Action.RAISE_HALF) and r[1] == (True, Action.CHECK_CALL)


def _sd_call_share(st: dict) -> float:
    """Smoothed showdown win share after calling river half-pot stabs this half."""
    won, n = st["riv_sd"]
    return (won + SD_CALL_PRIOR * SD_CALL_WEIGHT) / (n + SD_CALL_WEIGHT)


def _sd_lead_share(st: dict) -> float:
    """Smoothed showdown win share after calling its river half-pot leads this half."""
    won, n = st["riv_lead_sd"]
    return (won + SD_LEAD_PRIOR * SD_LEAD_WEIGHT) / (n + SD_LEAD_WEIGHT)


def _bet_fold_rate(st: dict, street: Street, action: Action) -> float:
    """Smoothed, capped fold rate to our first-in bet of this size on this street."""
    bets, folds = st["bet_fold"][street][action]
    return min((folds + BET_FOLD_PRIOR * BET_FOLD_WEIGHT) / (bets + BET_FOLD_WEIGHT), BET_FOLD_CAP)


def _facing_half_lead(obs: Observation) -> bool:
    """It bet half pot on the river before we acted on the river: we face it now."""
    h = obs.action_history
    return (obs.street == Street.RIVER and obs.to_call > 0 and len(h) >= 1
            and not h[-1].is_me and h[-1].action == Action.RAISE_HALF and h[-1].street == Street.RIVER
            and not any(a.is_me and a.street == Street.RIVER for a in h))


def _hand_ending_opp_actions(hist: list, result: HandResult, pots) -> list:
    """The opponent's actions after our last decision, which no Observation
    shows us. `hist` ends with our own last action. In heads-up the opponent
    can't act twice in a row without giving us a decision (game.py), so:
      * no showdown and we won          -> it folded
      * showdown after our raise        -> it called (a re-raise would have
                                           given us another decision)
      * showdown after we checked first on the river -> it checked behind
        (a bet would have given us another decision; on earlier streets a
        check-check leads to another decision for us, not a showdown)
    Otherwise our check/call closed the betting and it did nothing more.

    Each inference must also match the final pot exactly (`pots` comes from
    _pots_after). If it doesn't, our record is incomplete (e.g. the judge
    played a move for us), so nothing is inferred rather than guessing."""
    if not hist or not hist[-1][1] or pots is None:
        return []
    street, _, last = hist[-1]
    pot_after, opp_call = pots
    if not result.went_to_showdown:
        if result.winner == "me" and result.pot == pot_after:
            return [(street, False, Action.FOLD)]
        return []
    if last in (Action.RAISE_HALF, Action.RAISE_POT):
        if opp_call > 0 and result.pot == pot_after + opp_call:
            return [(street, False, Action.CHECK_CALL)]
        return []
    if last == Action.CHECK_CALL and street == Street.RIVER and result.pot == pot_after:
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
    we_checked = False          # our last move was a check after the flop (not a call)
    opp_bet_pending = False
    street = None
    for a in obs.action_history:
        if a.street != street:
            street, we_checked, opp_bet_pending = a.street, False, False
        if a.is_me:
            my_bet_pending = a.action in (Action.RAISE_HALF, Action.RAISE_POT)
            we_checked = (a.street != Street.PREFLOP and a.action == Action.CHECK_CALL
                          and not opp_bet_pending)
            opp_bet_pending = False
            continue
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            opp_bet_pending = True
            if a.street == Street.PREFLOP:
                f *= pf_k
            else:
                k = post_k
                if we_checked and a.action == Action.RAISE_HALF:
                    k = 1.0 - HALF_STAB_NARROW * (1.0 - post_k)   # half-pot stab after our check
                f *= k if a.street == obs.street else math.sqrt(k)
        elif a.action == Action.CHECK_CALL and my_bet_pending and a.street != Street.PREFLOP:
            f *= call_k
        my_bet_pending = we_checked = False
    return max(f, 0.05)


def _facing_half_stab(obs: Observation) -> bool:
    """We checked on this street after the flop and it bet half pot: we face it now."""
    h = obs.action_history
    return (obs.street != Street.PREFLOP and obs.to_call > 0 and len(h) >= 2
            and not h[-1].is_me and h[-1].action == Action.RAISE_HALF and h[-1].street == obs.street
            and h[-2].is_me and h[-2].action == Action.CHECK_CALL and h[-2].street == obs.street)


def _frequent_stabber(st: dict, street: Street) -> bool:
    """Does it bet (gate "any") or bet half pot (gate "half") at least the gate
    share of the times we check to it on `street` (raw counts, at least
    STAB_GATE_MIN chances this half)?"""
    bets, half, chances = st["chk_bet"][street]
    n = bets if STAB_GATE_KIND == "any" else half
    return chances >= STAB_GATE_MIN and n >= STAB_GATE_RATES[STAB_GATE_KIND] * chances


def _num_samples(obs: Observation, memory: dict) -> int:
    """Fixed sample count for this street and situation. Only if this half is
    on course to use too much of the time bank (a slow machine) do we drop to
    cheap mode, and past HARD_LIMIT to no simulation at all."""
    used = _charged_estimate(memory)
    counts = memory.setdefault("mode_counts", {"full": 0, "cheap": 0, "none": 0})
    if used > HARD_LIMIT:
        counts["none"] += 1
        return 0
    done = obs.hand_index + 1
    projected = used * memory.get("num_hands", 1000) / done if done >= MIN_HANDS_TO_PROJECT else used
    if max(used, projected) > PROJECTED_LIMIT:
        counts["cheap"] += 1
        return CHEAP_SAMPLES
    counts["full"] += 1
    no_bet, facing = SAMPLES[obs.street]
    return facing if obs.to_call > 0 else no_bet


def _charged_estimate(memory: dict) -> float:
    """Our estimate of the time bank the judge has charged so far."""
    return memory.get("t_used", 0.0) + memory.get("calls", 0) * OVERHEAD_PER_CALL


def _state_rng(obs: Observation) -> np.random.Generator:
    """A random generator seeded from the cards and the betting so far, so the
    same deal and history always give the same simulation (and decision)."""
    key = (tuple(obs.hole_cards), tuple(obs.board), int(obs.street), bool(obs.is_button),
           tuple((int(a.street), bool(a.is_me), int(a.action), int(a.amount)) for a in obs.action_history))
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


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
        rng = _state_rng(obs)
        sims = sample_showdowns(obs.hole_cards, obs.board, n, rng)
        eq = _postflop_equity(obs, sims, low_scale)
        e_now, e_called = eq(f), eq(keep)
        # Close call between our top two moves: sampling noise could flip it,
        # so draw more samples from the same generator (still repeatable) and
        # decide again on all of them. Full mode only (never when short on time).
        if n > CHEAP_SAMPLES and _top2_gap(_action_evs(obs, e_now, e_called, p_fold)) < CLOSE_GAP * max(obs.pot, 1):
            more = sample_showdowns(obs.hole_cards, obs.board, CLOSE_EXTRA, rng)
            sims = tuple(np.concatenate(pair) for pair in zip(sims, more))
            eq = _postflop_equity(obs, sims, low_scale)
            e_now, e_called = eq(f), eq(keep)
            counts = memory.setdefault("mode_counts", {})
            counts["resampled"] = counts.get("resampled", 0) + 1
    ev = _action_evs(obs, e_now, e_called, p_fold)
    if obs.street != Street.PREFLOP and _facing_half_stab(obs) and _frequent_stabber(st, obs.street):
        ev = _wide_call(obs, ev, eq(1.0), e_called, p_fold, memory)
    action = _mix(obs, ev, e_now, memory["rng"])
    if action == Action.CHECK_CALL and obs.to_call == 0 and obs.street in BET_FOLD_STREETS:
        action = _measured_bet(obs, ev, e_now, sims, f, st, memory)
    if action == Action.FOLD and obs.street == Street.RIVER:
        key = None
        if _facing_half_stab(obs) and _sd_call_share(st) > SD_CALL_ODDS:
            key = "sd_call"
        elif _facing_half_lead(obs) and _sd_lead_share(st) > SD_CALL_ODDS:
            key = "sd_lead_call"
        if key is not None:
            action = Action.CHECK_CALL
            counts = memory.setdefault("mode_counts", {})
            counts[key] = counts.get(key, 0) + 1
    return action


def _wide_call(obs, ev, e_wide, e_called, p_fold, memory) -> dict:
    """Facing a frequent stabber's half-pot bet: value the CALL with our equity
    against its whole range. Raise EVs stay narrowed, and if a raise is already
    best nothing changes (this only turns folds into calls)."""
    best = max(ev, key=ev.get)
    if best in (Action.RAISE_HALF, Action.RAISE_POT) or Action.CHECK_CALL not in ev:
        return ev
    ev = dict(ev)
    ev[Action.CHECK_CALL] = max(ev[Action.CHECK_CALL],
                                _action_evs(obs, e_wide, e_called, p_fold)[Action.CHECK_CALL])
    counts = memory.setdefault("mode_counts", {})
    counts["wide_call"] = counts.get("wide_call", 0) + 1
    return ev


def _measured_bet(obs, ev, e_now, sims, f, st, memory) -> Action:
    """We are about to check (turn/river, nothing to call). Re-value each bet
    with the measured fold rate and our equity vs only the hands that would
    continue; bet if that makes a bet the best move. Otherwise keep the check."""
    raises = [a for a in (Action.RAISE_HALF, Action.RAISE_POT) if a in ev]
    if not raises:
        return Action.CHECK_CALL
    new = dict(ev)
    for a in raises:
        pf = _bet_fold_rate(st, obs.street, a)
        e_called = _top_equity(sims, max(f * (1.0 - pf), 0.02))
        new[a] = _action_evs(obs, e_now, e_called, pf)[a]
    best_old, best_new = max(ev, key=ev.get), max(new, key=new.get)
    if best_new in raises and best_new != best_old:
        counts = memory.setdefault("mode_counts", {})
        counts["fb_bet"] = counts.get("fb_bet", 0) + 1
        return best_new
    return Action.CHECK_CALL


def _top_equity(sims, frac: float) -> float:
    """Our win share vs ONLY the opponent's best `frac` of sampled hands (ranked
    by strength on this board, as in _postflop_equity); weaker hands weigh 0."""
    my, opp, now = sims
    n = len(my)
    win = (my > opp) + 0.5 * (my == opp)
    pct = np.empty(n)
    pct[np.argsort(now, kind="stable")] = (np.arange(n) + 0.5) / n
    keep = pct >= 1.0 - frac
    if not keep.any():
        keep = pct >= pct.max()
    return float(win[keep].mean())


def _top2_gap(ev: dict) -> float:
    """Chips between the best and second-best move (inf if only one move)."""
    vals = sorted(ev.values(), reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else math.inf


def _postflop_equity(obs, sims, low_scale):
    """From simulated showdowns (my, opp, now) return eq(frac): our win chance
    against the opponent's best `frac` of hands (other hands weighted low)."""
    my, opp, now = sims
    n = len(my)
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
