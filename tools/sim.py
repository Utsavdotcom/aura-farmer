"""
sim.py - fast local matches without web servers (same rules and scoring as the judge).

    python tools/sim.py                      # our bot vs every practice opponent
    python tools/sim.py --opp station --deals 1000 --seeds 3

Each match is duplicate: half 1 our bot sits in seat 0, half 2 in seat 1 with
the same cards. Prints big blinds won per deal, the 95% range and the verdict,
plus how much thinking time our bot used per half (limit: 8 s).
"""
import argparse
import importlib.util
import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from game import BIG_BLIND, AgentConfig, Game, make_deal  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Bot:
    """Wraps a module with new_half/choose_action/on_hand_end."""

    def __init__(self, mod):
        self.mod, self.memory, self.time = mod, {}, 0.0

    def new_half(self):
        self.memory = {}
        self.time = 0.0
        self.mod.new_half(AgentConfig(), self.memory)

    def act(self, obs):
        t = time.perf_counter()
        a = self.mod.choose_action(obs, self.memory)
        self.time += time.perf_counter() - t
        if a not in obs.legal_actions:
            raise RuntimeError(f"illegal action {a!r}")
        return a

    def end(self, res):
        t = time.perf_counter()
        self.mod.on_hand_end(res, self.memory)
        self.time += time.perf_counter() - t


def opponents():
    import sparring
    bots = {}
    for f in sorted(os.listdir(os.path.join(ROOT, "baseline-bots"))):
        if f.endswith(".py"):
            bots[f[len("baseline_"):-3]] = lambda f=f: load(os.path.join(ROOT, "baseline-bots", f), f[:-3])
    for name in sparring.BOTS:
        bots[name] = lambda name=name: sparring.make(name)
    return bots


def play_half(a, b, seed, deals, a_seat):
    a.new_half()
    b.new_half()
    seats = {a_seat: a, 1 - a_seat: b}
    nets = []
    for d in range(deals):
        g = Game(make_deal(seed, d), button_seat=d % 2, hand_index=d)
        while not g.is_terminal:
            g.apply(seats[g.to_act].act(g.observation(g.to_act)))
        a.end(g.hand_result(a_seat))
        b.end(g.hand_result(1 - a_seat))
        nets.append(g.result().nets[a_seat])
    return nets, a.time


def match(a_mod, b_mod, seed, deals):
    a, b = Bot(a_mod), Bot(b_mod)
    n1, t1 = play_half(a, b, seed, deals, 0)
    n2, t2 = play_half(a, b, seed, deals, 1)
    paired = [(x + y) / BIG_BLIND for x, y in zip(n1, n2)]
    mean = statistics.mean(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired))
    return mean, se, max(t1, t2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="all", help="opponent name, comma list, or 'all'")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--agent", default=os.path.join(ROOT, "agent.py"))
    args = ap.parse_args()

    opps = opponents()
    names = list(opps) if args.opp == "all" else args.opp.split(",")
    for name in names:
        res = []
        for s in range(args.seeds):
            me = load(args.agent, f"me_{name}_{s}")
            res.append(match(me, opps[name](), f"sim{s}", args.deals))
        mean = statistics.mean(r[0] for r in res)
        se = math.sqrt(sum(r[1] ** 2 for r in res)) / len(res)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        verdict = "WIN" if lo > 0 else "LOSE" if hi < 0 else "CHOP"
        print(f"{name:16s} {mean:+7.3f} bb/deal  [{lo:+.3f}, {hi:+.3f}]  {verdict:5s}"
              f"  max think {max(r[2] for r in res):.2f}s/half", flush=True)


if __name__ == "__main__":
    main()
