"""
sparring.py - extra practice opponents for tools/sim.py (not used in matches).

Each bot plays from simple win-chance rules, with different personalities:
  checkfold  the starter bot: checks when free, otherwise folds
  nit        only bets/calls with strong hands, folds a lot
  tag        "tight aggressive": plays decent hands and bets them hard
  lag        "loose aggressive": bets and raises a lot, bluffs often
  sticky     calls too much, rarely raises (a smarter calling station)
  learner    watches US: bluffs more if we fold a lot, and folds to our
             raises if our raises always turn out to be strong hands
  self       a second copy of our own bot
"""
import importlib.util
import os
import random
import types

import numpy as np

from fasteval import sample_showdowns
from game import Action, Street
from preflop import canon_key, load_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF = load_table()

# (play threshold preflop percentile, value-bet equity, call margin, bluff chance)
STYLES = {
    "nit":    (0.55, 0.72, 0.12, 0.00),
    "tag":    (0.35, 0.62, 0.04, 0.10),
    "lag":    (0.10, 0.52, -0.02, 0.35),
    "sticky": (0.05, 0.80, -0.10, 0.00),
}
BOTS = ["checkfold", *STYLES, "learner", "self"]


def _learner_bot():
    """Exploits a predictable opponent. Starts like 'tag', then adapts."""
    RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

    def new_half(config, memory):
        memory.update(rng=random.Random(9), np=np.random.default_rng(9),
                      faced=4.0, folded=1.0,          # how often WE fold to its bets
                      sd_r=[0.6, 1], sd_n=[0.5, 1],    # our showdown strength: raised / not
                      we_raised=False)

    def choose_action(obs, memory):
        rng = memory["rng"]
        eq, pct = _equity(obs, memory["np"])
        we_raised_now = any(not a.is_me and a.action in RAISES and a.street == obs.street
                            for a in obs.action_history)
        memory["we_raised"] = memory["we_raised"] or we_raised_now
        can_raise = Action.RAISE_POT in obs.legal_actions
        odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        our_fold = memory["folded"] / memory["faced"]
        honest = memory["sd_r"][0] / memory["sd_r"][1] - memory["sd_n"][0] / memory["sd_n"][1]
        if obs.street == Street.PREFLOP:
            if pct < 0.3:
                return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
            if can_raise and (pct > 0.8 or rng.random() < our_fold * 0.5):
                return Action.RAISE_POT
            return Action.CHECK_CALL if eq >= odds or obs.to_call <= 1 else Action.FOLD
        if can_raise and eq >= 0.62:
            return Action.RAISE_POT
        if can_raise and obs.to_call == 0 and rng.random() < our_fold:
            return Action.RAISE_HALF                      # bluff as often as we fold
        if obs.to_call and we_raised_now:
            # Facing our raise: the more our raises mean strength, the tighter it calls.
            need = odds + max(0.0, honest) * 0.6
            return Action.CHECK_CALL if eq >= need else Action.FOLD
        if obs.to_call == 0 or eq >= odds:
            return Action.CHECK_CALL
        return Action.FOLD

    def on_hand_end(result, memory):
        # Did WE fold to its bet? (We folded = it won without showdown.)
        if not result.went_to_showdown:
            memory["faced"] += 0.5
            memory["folded"] += 0.5 * (result.winner == "me")
        elif result.opp_hole and len(result.board) == 5:
            me, other, _ = sample_showdowns(result.opp_hole, result.board, 128, memory["np"])
            strength = float(np.mean((me > other) + 0.5 * (me == other)))
            key = "sd_r" if memory["we_raised"] else "sd_n"
            memory[key][0] += strength
            memory[key][1] += 1
        memory["we_raised"] = False

    return types.SimpleNamespace(new_half=new_half, choose_action=choose_action,
                                 on_hand_end=on_hand_end)


def _equity(obs, rng):
    if obs.street == Street.PREFLOP:
        return PF[canon_key(obs.hole_cards)][0], PF[canon_key(obs.hole_cards)][1]
    my, opp, _ = sample_showdowns(obs.hole_cards, obs.board, 200, rng)
    e = float(np.mean((my > opp) + 0.5 * (my == opp)))
    return e, e


def _style_bot(style):
    play, value, margin, bluff = STYLES[style]

    def new_half(config, memory):
        memory["rng"] = random.Random(7)
        memory["np"] = np.random.default_rng(7)

    def choose_action(obs, memory):
        rng = memory["rng"]
        eq, pct = _equity(obs, memory["np"])
        can_raise = Action.RAISE_POT in obs.legal_actions
        odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        if obs.street == Street.PREFLOP:
            if pct < play:
                return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
            if can_raise and pct > 1 - (1 - play) / 3:
                return Action.RAISE_POT
            return Action.CHECK_CALL if eq >= odds + margin or obs.to_call <= 1 else Action.FOLD
        if can_raise and eq >= value:
            return Action.RAISE_POT if eq > value + 0.1 else Action.RAISE_HALF
        if can_raise and obs.to_call == 0 and rng.random() < bluff:
            return Action.RAISE_HALF
        if obs.to_call == 0 or eq >= odds + margin:
            return Action.CHECK_CALL
        return Action.FOLD

    return types.SimpleNamespace(new_half=new_half, choose_action=choose_action,
                                 on_hand_end=lambda r, m: None)


def make(name):
    if name == "checkfold":
        return types.SimpleNamespace(
            new_half=lambda c, m: None,
            choose_action=lambda o, m: Action.CHECK_CALL if o.to_call == 0 else Action.FOLD,
            on_hand_end=lambda r, m: None)
    if name == "learner":
        return _learner_bot()
    if name == "self":
        spec = importlib.util.spec_from_file_location("self_copy", os.path.join(ROOT, "agent.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _style_bot(name)
