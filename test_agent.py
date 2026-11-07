"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import math
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Our own tests (strategy.py, fasteval.py, preflop.py)
# ---------------------------------------------------------------------------

def test_fast_evaluator_matches_reference():
    import numpy as np
    from evaluator import omaha_score
    from fasteval import best_scores
    rng = random.Random(3)
    holes, boards = [], []
    for _ in range(2000):
        cards = rng.sample(range(36), 10)
        holes.append(cards[:5])
        boards.append(cards[5:])
    fast = best_scores(np.array(holes), np.array(boards))
    for h, b, s in zip(holes, boards, fast):
        assert s == omaha_score(h, b)


def test_preflop_table_covers_every_hand():
    from preflop import canon_key, load_table
    table = load_table()
    rng = random.Random(4)
    for _ in range(3000):
        assert canon_key(sorted(rng.sample(range(36), 5))) in table


def test_out_of_time_still_legal(bot):
    # With the time bank used up the bot must still answer legally.
    module, client = bot
    start_half(client)
    module.MEMORY["t_used"] = 999.0
    play_hands(client, 30, seed="late", raise_rate=0.5)


def test_strategy_errors_fall_back_to_legal_action(bot, monkeypatch):
    import strategy
    module, client = bot
    start_half(client)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(strategy, "_decide", boom)
    monkeypatch.setattr(strategy, "_learn", boom)
    play_hands(client, 20, seed="boom")


def test_full_half_fits_time_budget():
    # One 1000-hand half against an aggressive random opponent, timed.
    import time
    import strategy
    from game import HandResult  # noqa: F401
    memory = {}
    strategy.new_half(AgentConfig(), memory)
    rng = random.Random(5)
    start = time.perf_counter()
    for hand in range(1000):
        game = Game(make_deal("budget", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                action = strategy.choose_action(obs, memory)
            else:
                action = rng.choice(obs.legal_actions)
            game.apply(action)
        strategy.on_hand_end(game.hand_result(0), memory)
    assert time.perf_counter() - start < 6.0


def test_opponent_stats_count_hand_ending_actions():
    # The opponent's last action (a river call, a check behind, a fold) never
    # reaches choose_action. The postflop stats must still count every
    # opponent decision and every time it faced our bet, exactly.
    import strategy
    memory = {}
    strategy.new_half(AgentConfig(), memory)
    start = dict(memory["stats"])
    rng = random.Random(1)
    decisions = faced = folded = 0
    for hand in range(400):
        game = Game(make_deal("stats", hand), button_seat=hand % 2, hand_index=hand)
        we_bet = False
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                action = strategy.choose_action(obs, memory)
                we_bet = action in (Action.RAISE_HALF, Action.RAISE_POT)
            else:
                # folds 40% to our bets, otherwise calls or checks (never raises)
                action = Action.FOLD if we_bet and rng.random() < 0.4 else Action.CHECK_CALL
                if obs.street != 0:
                    decisions += 1
                    faced += we_bet
                    folded += action == Action.FOLD
                we_bet = False
            game.apply(action)
        strategy.on_hand_end(game.hand_result(0), memory)
    st = memory["stats"]
    assert st["post_dec"] - start["post_dec"] == decisions
    assert st["faced"] - start["faced"] == faced
    assert st["folded"] - start["folded"] == folded


# ---------------------------------------------------------------------------
# Opponent statistics vs the engine's own action log (ground truth)
# ---------------------------------------------------------------------------

STAT_KEYS = ("pf_dec", "pf_raise", "pf_faced", "pf_folded", "post_dec", "post_raise",
             "faced", "folded", "my_faced", "my_folded")
C, F, H, P = Action.CHECK_CALL, Action.FOLD, Action.RAISE_HALF, Action.RAISE_POT


def _true_counts(history, me):
    """Count the stats strategy.py tracks, straight from the engine's full,
    seat-based action log (game.result().history). Written independently of
    strategy._learn: an action "faces a bet" when the previous action on the
    same street was a raise by the other seat."""
    c = dict.fromkeys(STAT_KEYS, 0)
    for i, rec in enumerate(history):
        prev = history[i - 1] if i and history[i - 1].street == rec.street else None
        facing = prev is not None and prev.seat != rec.seat and prev.action in (H, P)
        if rec.seat == me:
            if facing and rec.street != 0:
                c["my_faced"] += 1
                c["my_folded"] += rec.action == F
            continue
        pre = "pf_" if rec.street == 0 else "post_"
        c[pre + "dec"] += 1
        c[pre + "raise"] += rec.action in (H, P)
        if facing:
            c["pf_faced" if rec.street == 0 else "faced"] += 1
            c["pf_folded" if rec.street == 0 else "folded"] += rec.action == F
    return c


def _counted(memory, start):
    return {k: memory["stats"][k] - start[k] for k in STAT_KEYS}


def _sparring(name):
    sys.path.insert(0, os.path.join(BUNDLE, "tools"))
    import sparring
    return sparring.make(name)


def _random_opp(raise_rate, seed):
    import types
    rng = random.Random(seed)

    def act(obs, memory):
        raises = [a for a in obs.legal_actions if a in (H, P)]
        if raises and rng.random() < raise_rate:
            return rng.choice(raises)
        if F in obs.legal_actions and rng.random() < 0.25:
            return F
        return C
    return types.SimpleNamespace(new_half=lambda c, m: None, choose_action=act,
                                 on_hand_end=lambda r, m: None)


def _play_and_compare(opp, hands, seed, skip_rate=0.0):
    """Our strategy vs `opp`, alternating seats. After every hand, compare the
    stats strategy.py counted with the engine's ground truth. With
    skip_rate > 0 some of our moves are played by the judge's default move
    instead (out of time / an error), so our record of the hand is partial.
    Returns (exact_hands, partial_hands)."""
    import strategy
    from game import default_action
    memory, opp_mem = {}, {}
    strategy.new_half(AgentConfig(), memory)
    opp.new_half(AgentConfig(), opp_mem)
    rng = random.Random(seed)
    start = dict(memory["stats"])
    exact = partial = 0
    for hand in range(hands):
        me = (hand // 2) % 2      # the button alternates every hand, our seat every 2
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        skipped = False
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == me:
                if rng.random() < skip_rate:
                    action, skipped = default_action(obs.to_call), True
                else:
                    action = strategy.choose_action(obs, memory)
            else:
                action = opp.choose_action(obs, opp_mem)
            assert action in obs.legal_actions
            game.apply(action)
        before = _counted(memory, start)
        strategy.on_hand_end(game.hand_result(me), memory)
        opp.on_hand_end(game.hand_result(1 - me), opp_mem)
        got = {k: v - before[k] for k, v in _counted(memory, start).items()}
        want = _true_counts(game.result().history, me)
        if skipped:
            partial += 1
            for k in STAT_KEYS:    # may miss actions, must never invent one
                assert got[k] <= want[k], (hand, k, got, want, game.result().history)
        else:
            exact += 1
            assert got == want, (hand, got, want, game.result().history)
    return exact, partial


@pytest.mark.parametrize("opp_name", ["random_aggro", "random_passive", "learner", "tag", "lag", "nit"])
def test_opponent_stats_match_engine_log(opp_name):
    if opp_name.startswith("random"):
        opp = _random_opp(0.5 if opp_name == "random_aggro" else 0.1, opp_name)
    else:
        opp = _sparring(opp_name)
    exact, partial = _play_and_compare(opp, 300, "truth-" + opp_name)
    assert exact == 300 and partial == 0


def test_always_pot_raise_rate_matches_engine_log():
    # always_pot raises whenever it can, so its hand-ending actions are mostly
    # calls (raise cap reached, or we are all-in). The old code missed those
    # and saw a postflop raise rate of 1.000 instead of about 0.85.
    import strategy
    spec = importlib.util.spec_from_file_location(
        "baseline_always_pot", os.path.join(BUNDLE, "baseline-bots", "baseline_always_pot.py"))
    opp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opp)
    exact, partial = _play_and_compare(opp, 60, "always-pot")    # exact per hand
    assert exact == 60 and partial == 0
    # And the raise rate over the whole run equals the engine log's.
    memory, opp_mem = {}, {}
    strategy.new_half(AgentConfig(), memory)
    opp.new_half(AgentConfig(), opp_mem)
    start = dict(memory["stats"])
    true_dec = true_raise = 0
    for hand in range(60):
        game = Game(make_deal("always-pot", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            game.apply(strategy.choose_action(obs, memory) if game.to_act == 0
                       else opp.choose_action(obs, opp_mem))
        strategy.on_hand_end(game.hand_result(0), memory)
        want = _true_counts(game.result().history, 0)
        true_dec += want["post_dec"]
        true_raise += want["post_raise"]
    got = _counted(memory, start)
    assert true_dec >= 6 and 0 < true_raise < true_dec      # it doesn't always raise
    assert (got["post_dec"], got["post_raise"]) == (true_dec, true_raise)


def test_partial_records_never_invent_actions():
    # The judge sometimes plays our move for us (out of time, an error).
    # Then we can't see the whole hand: stats may miss actions but must
    # never count one that didn't happen.
    exact, partial = _play_and_compare(_random_opp(0.4, "skip"), 400, "skip", skip_rate=0.15)
    assert partial > 50 and exact > 50


def _scripted(script, me_button, monkeypatch):
    """Play one hand where both sides follow `script` (actions in order).
    Our moves go through strategy.choose_action with _decide forced.
    Checks the counts against the engine log; returns (delta, memory, game)."""
    import strategy
    memory = {}
    strategy.new_half(AgentConfig(), memory)
    start = dict(memory["stats"])
    moves = iter(script)
    me = 0 if me_button else 1
    game = Game(make_deal("script", 0), button_seat=0, hand_index=0)
    while not game.is_terminal:
        want = next(moves)
        obs = game.observation(game.to_act)
        if game.to_act == me:
            monkeypatch.setattr(strategy, "_decide", lambda o, m, a=want: a)
            assert strategy.choose_action(obs, memory) == want
        game.apply(want)
    assert next(moves, None) is None, "script is longer than the hand"
    strategy.on_hand_end(game.hand_result(me), memory)
    delta = _counted(memory, start)
    assert delta == _true_counts(game.result().history, me)
    return delta, memory, game


def test_terminal_river_call_is_counted(monkeypatch):
    # We are BB (first to act after the flop) and bet the river; the button calls.
    d, _, g = _scripted([C, C, C, C, C, C, H, C], me_button=False, monkeypatch=monkeypatch)
    assert g.result().went_to_showdown
    assert (d["post_dec"], d["faced"], d["folded"], d["post_raise"]) == (3, 1, 0, 0)


def test_terminal_check_behind_is_counted(monkeypatch):
    # We are BB and check the river; the button checks behind.
    d, _, _ = _scripted([C, C, C, C, C, C, C, C], me_button=False, monkeypatch=monkeypatch)
    assert (d["post_dec"], d["faced"]) == (3, 0)


def test_terminal_fold_is_counted_once(monkeypatch):
    # We are the button: the BB checks the flop, we bet, it folds.
    d, _, _ = _scripted([C, C, C, H, F], me_button=True, monkeypatch=monkeypatch)
    assert (d["post_dec"], d["faced"], d["folded"]) == (2, 1, 1)


def test_all_in_call_is_counted(monkeypatch):
    # We complete, then pot raises go back and forth until we shove; the
    # opponent calls the shove and the board is run out.
    d, _, g = _scripted([C, P, P, P, P, C], me_button=True, monkeypatch=monkeypatch)
    assert g.result().went_to_showdown and g.result().pot == 160
    assert (d["pf_dec"], d["pf_raise"], d["pf_faced"], d["pf_folded"]) == (3, 2, 2, 0)


def test_our_closing_call_adds_nothing(monkeypatch):
    # The button bets the river and we call: the hand ends on OUR action,
    # so no opponent action may be added after it.
    d, _, _ = _scripted([C, C, C, C, C, C, C, H, C], me_button=False, monkeypatch=monkeypatch)
    assert (d["post_dec"], d["post_raise"], d["faced"], d["my_faced"]) == (3, 1, 0, 1)


def test_hand_result_delivered_twice_counts_once(monkeypatch):
    import strategy
    _, memory, g = _scripted([C, C, C, H, F], me_button=True, monkeypatch=monkeypatch)
    snapshot = {k: memory["stats"][k] for k in STAT_KEYS}
    strategy.on_hand_end(g.hand_result(0), memory)
    assert {k: memory["stats"][k] for k in STAT_KEYS} == snapshot


def test_new_half_resets_opponent_memory(monkeypatch):
    import strategy
    _, memory, _ = _scripted([C, C, C, H, F], me_button=True, monkeypatch=monkeypatch)
    assert memory["stats"] != strategy._new_stats()
    strategy.new_half(AgentConfig(), memory)
    assert memory["stats"] == strategy._new_stats()
    assert memory["hist"] == [] and memory["pots"] is None and memory["last_learned"] == -1


def test_stats_exact_through_http_with_batched_results(bot):
    # The same ground-truth check, but through the server, with finished hands
    # delivered late and in batches exactly like judge_engine.py does.
    module, client = bot
    start_half(client)
    start = dict(module.MEMORY["stats"])
    opp = _random_opp(0.35, "http")
    truth = dict.fromkeys(STAT_KEYS, 0)
    pending = []
    for hand in range(150):
        game = Game(make_deal("http", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200 and r.json()["hand_end_errors"] == []
                action = coerce_action(r.json()["action"])
                assert action in obs.legal_actions
            else:
                action = opp.choose_action(obs, {})
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
        for k, v in _true_counts(game.result().history, 0).items():
            truth[k] += v
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    assert _counted(module.MEMORY, start) == truth


# ---------------------------------------------------------------------------
# Repeatability and time safety
# ---------------------------------------------------------------------------

def _play_record(hands, seed, slow_clock=0.0):
    """Our strategy vs a seeded random opponent; returns every action we chose.
    slow_clock adds fake seconds to our timer each decision (a busy machine)."""
    import strategy
    memory = {}
    strategy.new_half(AgentConfig(), memory)
    opp = _random_opp(0.35, seed)
    actions = []
    for hand in range(hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                memory["t_used"] += slow_clock
                action = strategy.choose_action(obs, memory)
                actions.append(action)
            else:
                action = opp.choose_action(obs, {})
            game.apply(action)
        strategy.on_hand_end(game.hand_result(0), memory)
    return actions, memory


def test_same_deals_give_identical_decisions():
    # Same deals and history -> same decisions, even if the machine is slower
    # (as long as we stay under the time limits).
    a, _ = _play_record(120, "repeat")
    b, _ = _play_record(120, "repeat")
    c, mem = _play_record(120, "repeat", slow_clock=0.0002)
    assert a == b == c
    assert mem["mode_counts"]["cheap"] == mem["mode_counts"]["none"] == 0


def test_state_seeded_simulation_is_repeatable():
    import strategy
    obs = Game(make_deal("rng", 1), button_seat=0).observation(0)
    x = strategy._state_rng(obs).random(5)
    y = strategy._state_rng(obs).random(5)
    assert (x == y).all()


def test_projected_overrun_switches_to_cheap_then_none():
    import strategy
    game = Game(make_deal("cheap", 0), button_seat=0, hand_index=99)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)                     # now on the flop
    obs = game.observation(game.to_act)
    memory = {}
    strategy.new_half(AgentConfig(), memory)
    memory["t_used"] = 0.30                            # 100 hands: on pace for 3.0 s
    assert strategy._num_samples(obs, memory) == strategy.SAMPLES[obs.street][0]
    memory["t_used"] = 0.55                            # on pace for 5.5 s > 5.0
    assert strategy._num_samples(obs, memory) == strategy.CHEAP_SAMPLES
    memory["t_used"] = strategy.HARD_LIMIT + 0.1
    assert strategy._num_samples(obs, memory) == 0
    assert strategy.choose_action(obs, memory) in obs.legal_actions
    assert memory["mode_counts"] == {"full": 1, "cheap": 1, "none": 2}


def test_charged_estimate_adds_per_call_overhead():
    import strategy
    memory = {"t_used": 2.0, "calls": 1000}
    assert strategy._charged_estimate(memory) == pytest.approx(2.0 + 1000 * strategy.OVERHEAD_PER_CALL)


# ---------------------------------------------------------------------------
# Field clones (tools/field/): practice opponents must be legal and fast too
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bot_file", ["f1_equity.py", "f2_rules.py", "f3_aggro.py", "f4_exploit.py"])
def test_field_bot_legal_and_fast(bot_file):
    import time
    spec = importlib.util.spec_from_file_location(
        "field_" + bot_file[:-3], os.path.join(BUNDLE, "tools", "field", bot_file))
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)
    opp = _random_opp(0.4, bot_file)
    memory = {}
    bot.new_half(AgentConfig(), memory)
    think, slowest = 0.0, 0.0
    for hand in range(300):
        seat = hand // 2 % 2
        game = Game(make_deal("field-" + bot_file, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == seat:
                t = time.perf_counter()
                action = bot.choose_action(obs, memory)
                dt = time.perf_counter() - t
                think, slowest = think + dt, max(slowest, dt)
                assert action in obs.legal_actions, (bot_file, action, obs.legal_actions)
            else:
                action = opp.choose_action(obs, {})
            game.apply(action)
        t = time.perf_counter()
        bot.on_hand_end(game.hand_result(seat), memory)
        think += time.perf_counter() - t
    assert think / 300 * 1000 < 6.0, f"{bot_file}: {think / 300 * 1000:.2f} s per 1000 hands"
    assert slowest < 0.25, f"{bot_file}: slowest decision {slowest * 1000:.1f} ms"


# ---------------------------------------------------------------------------
# How often the opponent bets when we check to it (per street)
# ---------------------------------------------------------------------------

def _true_checkbets(history, me):
    """From the engine log: per postflop street, [bets, chances] where a
    chance is an opponent action right after OUR check (a check that did not
    face a bet) on the same street. Written independently of strategy._learn."""
    out = {s: [0, 0] for s in (1, 2, 3)}
    for i, rec in enumerate(history):
        if rec.seat == me or rec.street == 0 or not i:
            continue
        prev = history[i - 1]
        if prev.street != rec.street or prev.seat != me or prev.action != C:
            continue
        before = history[i - 2] if i >= 2 and history[i - 2].street == rec.street else None
        if before is not None and before.action in (H, P):
            continue                    # our CHECK_CALL was a call, not a check
        out[int(rec.street)][0] += rec.action in (H, P)
        out[int(rec.street)][1] += 1
    return out


def _checkbet_counts(memory):
    return {int(s): list(v) for s, v in memory["stats"]["chk_bet"].items()}


@pytest.mark.parametrize("opp_name,skip", [("random_aggro", 0.0), ("learner", 0.0), ("lag", 0.0),
                                           ("nit", 0.0), ("random_skip", 0.15)])
def test_checkbet_counts_match_engine_log(opp_name, skip):
    import strategy
    from game import default_action
    opp = _random_opp(0.5, opp_name) if opp_name.startswith("random") else _sparring(opp_name)
    memory, opp_mem = {}, {}
    strategy.new_half(AgentConfig(), memory)
    opp.new_half(AgentConfig(), opp_mem)
    rng = random.Random(opp_name)
    chances = 0
    for hand in range(300):
        me = (hand // 2) % 2
        game = Game(make_deal("chk-" + opp_name, hand), button_seat=hand % 2, hand_index=hand)
        skipped = False
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == me:
                if rng.random() < skip:
                    action, skipped = default_action(obs.to_call), True
                else:
                    action = strategy.choose_action(obs, memory)
            else:
                action = opp.choose_action(obs, opp_mem)
            game.apply(action)
        before = _checkbet_counts(memory)
        strategy.on_hand_end(game.hand_result(me), memory)
        opp.on_hand_end(game.hand_result(1 - me), opp_mem)
        after = _checkbet_counts(memory)
        got = {s: [after[s][0] - before[s][0], after[s][1] - before[s][1]] for s in after}
        want = _true_checkbets(game.result().history, me)
        chances += sum(v[1] for v in want.values())
        if skipped:                     # partial record: may miss, never invent
            assert all(got[s][0] <= want[s][0] and got[s][1] <= want[s][1] for s in got), (hand, got, want)
        else:
            assert got == want, (hand, got, want, game.result().history)
    assert chances > 30


def test_checkbet_rate_starts_neutral_and_is_smoothed():
    import strategy
    from game import Street
    st = strategy._new_stats()
    for s in (Street.FLOP, Street.TURN, Street.RIVER):
        assert strategy._checkbet_rate(st, s) == pytest.approx(strategy.CHECKBET_PRIOR)
    # 30 checks in 30 chances only moves it halfway from the prior (worth 30 chances)
    assert strategy.CHECKBET_PRIOR_N == 30
    st["chk_bet"][Street.TURN][1] += 30
    assert strategy._checkbet_rate(st, Street.TURN) == pytest.approx(strategy.CHECKBET_PRIOR / 2)
    assert strategy._checkbet_rate(st, Street.FLOP) == pytest.approx(strategy.CHECKBET_PRIOR)  # per street
    # never bets: floored, so a bet still leaves it some hands
    st["chk_bet"][Street.RIVER] = [0.0, 1000.0]
    assert strategy._checkbet_rate(st, Street.RIVER) == strategy.CHECKBET_MIN


def test_checkbet_counted_in_scripted_hands(monkeypatch):
    import strategy
    from game import Street
    b0, n0 = strategy.CHECKBET_PRIOR * strategy.CHECKBET_PRIOR_N, float(strategy.CHECKBET_PRIOR_N)
    # We are BB: check the flop, the button bets, we fold.
    _, mem, _ = _scripted([C, C, C, H, F], me_button=False, monkeypatch=monkeypatch)
    assert mem["stats"]["chk_bet"][Street.FLOP] == [b0 + 1, n0 + 1]
    # We are BB: check every street; the button checks behind each time
    # (the river check is inferred from the showdown).
    _, mem, _ = _scripted([C] * 8, me_button=False, monkeypatch=monkeypatch)
    for s in (Street.FLOP, Street.TURN, Street.RIVER):
        assert mem["stats"]["chk_bet"][s] == [b0, n0 + 1]
    # The button checks first and we bet: not a chance for it after our check.
    _, mem, _ = _scripted([C, C, C, H, F], me_button=True, monkeypatch=monkeypatch)
    assert mem["stats"]["chk_bet"][Street.FLOP] == [b0, n0]


def _obs_after(script):
    """Observation for whoever acts after `script` (button = seat 0)."""
    game = Game(make_deal("range", 0), button_seat=0, hand_index=0)
    for a in script:
        game.apply(a)
    return game.observation(game.to_act)


@pytest.mark.parametrize("bets,chances", [(28.5, 30.0), (60.0, 100.0), (6.0, 40.0)])
def test_checkbet_range_width_equals_bet_rate(bets, chances):
    import strategy
    from game import Street
    st = strategy._new_stats()
    st["chk_bet"][Street.FLOP] = [bets, chances]
    # BB (seat 1) checks the flop, the button bets: it keeps its best `rate` share.
    obs = _obs_after([C, C, C, H])
    assert obs.street == Street.FLOP and obs.to_call > 0
    assert strategy._range_fraction(obs, st) == pytest.approx(max(bets / chances, 0.05))
    # Next street after we called it: sqrt decay, as for other postflop bets.
    obs = _obs_after([C, C, C, H, C])
    assert obs.street == Street.TURN
    assert strategy._range_fraction(obs, st) == pytest.approx(max(math.sqrt(bets / chances), 0.05))


def test_lead_bet_still_uses_overall_raise_rate():
    import strategy
    st = strategy._new_stats()
    st["chk_bet"] = {s: [29.0, 30.0] for s in st["chk_bet"]}
    # The BB leads into us (we never checked): unchanged rule, post_k.
    obs = _obs_after([C, C, H])
    assert obs.to_call > 0
    assert strategy._range_fraction(obs, st) == pytest.approx(max(strategy._raise_ks(st)[1], 0.05))
