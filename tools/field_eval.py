"""
field_eval.py - our bot vs the field clones on HELD-OUT seeds, with leak attribution.

    python tools/field_eval.py                                # f1-f4 + learner, seeds hold0-hold7
    python tools/field_eval.py --opp f4 --seeds 2 --deals 500

Seeds are named hold0, hold1, ... and are for measurement only: never tune
on them (use tools/sim.py's sim0, sim1, ... for development).

For each opponent it prints big blinds won per deal (duplicate: both seats on
the same cards), the 95% interval and verdict, the per-deal standard
deviation, the smallest edge that would count as a significant win in one
2000-hand match, and where the chips go: every hand's result is attributed
to OUR LAST DECISION in that hand (street, what we did, what we faced).
"""
import argparse
import collections
import importlib.util
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "field")]

from game import BIG_BLIND, AgentConfig, Game, make_deal  # noqa: E402

FIELD = {"f1": "f1_equity.py", "f2": "f2_rules.py", "f3": "f3_aggro.py", "f4": "f4_exploit.py"}
RAISES = ("RAISE_HALF", "RAISE_POT")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_opponent(name, tag):
    if name in FIELD:
        return load(os.path.join(ROOT, "tools", "field", FIELD[name]), f"field_{name}_{tag}")
    base = os.path.join(ROOT, "baseline-bots", f"baseline_{name}.py")
    if os.path.exists(base):
        return load(base, f"base_{name}_{tag}")
    import sparring
    return sparring.make(name)


def situation(history, me):
    """Label OUR last decision in a finished hand: 'STREET what [facing ...]'."""
    mine = [i for i, r in enumerate(history) if r.seat == me]
    if not mine:
        return "PREFLOP no decision (opp folded)"
    i = mine[-1]
    rec = history[i]
    prev = history[i - 1] if i and history[i - 1].street == rec.street else None
    facing = prev.action.name if prev is not None and prev.seat != me and prev.action.name in RAISES else None
    size = {"RAISE_HALF": "half-pot", "RAISE_POT": "pot"}.get(facing)
    if rec.action.name == "FOLD":
        what = f"fold to {size} bet"
    elif rec.action.name == "CHECK_CALL":
        what = f"call {size} bet" if facing else ("complete/check" if rec.street.name == "PREFLOP" else "check")
    else:
        kind = "raise" if facing else "bet"
        what = f"{kind} {'half' if rec.action.name == 'RAISE_HALF' else 'pot'}"
        if facing:
            what += f" vs {size}"
    return f"{rec.street.name:7s} {what}"


def play_half(me_mod, opp_mod, seed, deals, me_seat, leaks):
    me_mem, opp_mem = {}, {}
    me_mod.new_half(AgentConfig(), me_mem)
    opp_mod.new_half(AgentConfig(), opp_mem)
    nets = []
    for d in range(deals):
        g = Game(make_deal(seed, d), button_seat=d % 2, hand_index=d)
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            bot, mem = (me_mod, me_mem) if g.to_act == me_seat else (opp_mod, opp_mem)
            action = bot.choose_action(obs, mem)
            if action not in obs.legal_actions:
                raise RuntimeError(f"illegal action {action!r} by {'us' if bot is me_mod else 'opponent'}")
            g.apply(action)
        me_mod.on_hand_end(g.hand_result(me_seat), me_mem)
        opp_mod.on_hand_end(g.hand_result(1 - me_seat), opp_mem)
        res = g.result()
        net = res.nets[me_seat]
        nets.append(net)
        key = situation(res.history, me_seat)
        leaks[key][0] += 1
        leaks[key][1] += net
    return nets


def evaluate(name, seeds, deals):
    me = load(os.path.join(ROOT, "agent.py"), f"me_vs_{name}")
    leaks = collections.defaultdict(lambda: [0, 0])
    paired = []
    for s in seeds:
        opp = make_opponent(name, s)
        n1 = play_half(me, opp, s, deals, 0, leaks)
        n2 = play_half(me, opp, s, deals, 1, leaks)
        paired += [(x + y) / BIG_BLIND for x, y in zip(n1, n2)]
    mean = statistics.mean(paired)
    sd = statistics.stdev(paired)
    se = sd / math.sqrt(len(paired))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    verdict = "WIN" if lo > 0 else "LOSE" if hi < 0 else "CHOP"
    margin = 1.96 * sd / math.sqrt(deals)     # one match = `deals` duplicate deals
    print(f"\n{name}: {mean:+.3f} bb/deal  [{lo:+.3f}, {hi:+.3f}]  {verdict}   "
          f"(n={len(paired)} deals, seeds {seeds[0]}..{seeds[-1]})")
    print(f"  per-deal sd {sd:.2f} bb -> a single {deals}-deal match needs a true edge of about "
          f"{margin:.2f} bb/deal to be a significant win")
    hands = 2 * len(paired)
    rows = sorted(leaks.items(), key=lambda kv: kv[1][1])
    print(f"  where the chips go (all {hands} hands, by our last decision), bb per 1000 hands:")
    for key, (count, chips) in rows[:6]:
        print(f"    {key:38s} {count:6d} hands ({100 * count / hands:4.1f}%)  "
              f"{1000 * chips / BIG_BLIND / hands:+8.1f}  ({chips / BIG_BLIND / count:+.2f} bb/hand)")
    print("    ... biggest winners:")
    for key, (count, chips) in rows[-3:][::-1]:
        print(f"    {key:38s} {count:6d} hands ({100 * count / hands:4.1f}%)  "
              f"{1000 * chips / BIG_BLIND / hands:+8.1f}  ({chips / BIG_BLIND / count:+.2f} bb/hand)")
    by_street = collections.defaultdict(lambda: [0, 0])
    for key, (count, chips) in leaks.items():
        st = key.split()[0]
        by_street[st][0] += count
        by_street[st][1] += chips
    print("  by street of our last decision (bb per 1000 hands): " + ", ".join(
        f"{st} {1000 * c / BIG_BLIND / hands:+.0f}" for st, (n, c) in sorted(by_street.items())))
    return mean, lo, hi, verdict, sd, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="f1,f2,f3,f4,learner")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--deals", type=int, default=1000)
    args = ap.parse_args()
    seeds = [f"hold{i}" for i in range(args.seeds)]
    for name in args.opp.split(","):
        evaluate(name, seeds, args.deals)


if __name__ == "__main__":
    main()
