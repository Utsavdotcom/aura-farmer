# Team notes: start here

Read `README.md` for the tournament rules. This file covers what we built and how to work on it.

## Setup (5 minutes)
```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt numpy
.venv/bin/python -m pytest -q          # should say: 17 passed
```

## Where things are
| File | What it does |
|---|---|
| `agent.py` | The web server the tournament talks to. Only hands over to `strategy.py`. Don't change the server code. |
| `strategy.py` | **The brain.** All decisions are made here. Settings are at the top. |
| `fasteval.py` | Fast hand scorer (numpy). Gives the same scores as `evaluator.py` (tested). |
| `preflop.py` + `data/preflop.json` | Starting-hand strength table for all 20k kinds of starting hands. |
| `tools/build_preflop.py` | Rebuilds that table (about 20 s). Only needed if the rules change. |
| `tools/sim.py` | Fast test matches without servers. **Use this to test every change.** |
| `tools/sparring.py` | Extra practice bots: checkfold, nit, tag, lag, sticky, learner, self. |
| `test_agent.py` | Tests. Run them before every push. |

## How the bot decides (strategy.py)
1. **Win chance ("equity"):** a table lookup before the flop. After the flop, it plays out a fixed 150-300 random endings (more when facing a bet), seeded from the cards and betting so the same situation always gives the same decision.
2. **Opponent's likely hands:** when the opponent raises, it narrows them down, more so if that opponent rarely raises.
3. **Truth check:** at showdown it checks whether the opponent's raises really came with strong hands. If not, it trusts them less.
4. **Fold rate:** it tracks how often this opponent folds to our bets.
5. **Chips per move:** it works out the expected chips for fold, call, half-pot raise and pot raise.
6. **Mixing:** it picks the best move, or sometimes a close one, so we're harder to read (`_mix`).

Everything it learns resets every half (the tournament rule).

## How to test a change
```
.venv/bin/python tools/sim.py --deals 1000 --seeds 3             # vs every bot (~1-2 min)
.venv/bin/python tools/sim.py --opp learner,tag --deals 1000 --seeds 3
```
- Results are in big blinds per deal. WIN means the whole 95% range is above 0.
- Results move around by about ±0.3, so use `--seeds 3` or more before believing a small change.

## Current results (1000 deals x 3 runs)
- Clear wins vs all 4 baselines (+7 to +12), sticky (+7), lag (+3), tag (+1.5) and nit (+1.0).
- **We lose to `learner` (-0.75).** This is the main open problem.
- Thinking time (judge-charged, laptop): at most about 3.1 s of the 8 s per half, 4.8 s with a 1.5x slower CPU. 0 errors. See bench/phase1_timing.txt.

## Open work (in order)
1. **Beat `learner`.** Already tried with no gain: calling more vs frequent bettors, counting bluffs from shown hands, and making checks cost more. Our river folds look correct (we're usually behind when we fold).
2. **Make the tag/nit wins bigger.** They're small and could become draws vs strong teams.
3. **Run `./tools/check_build.sh`** (needs Docker running). It must say `ALL CHECKS PASSED` before we submit.
4. **Set the team name:** `PARTICIPANT` / `AGENT_NAME` in `agent.py`.

## Rules for working together
- Run `pytest` and `tools/sim.py` before pushing.
- Only keep a change if it doesn't make any opponent noticeably worse.
- Never let the bot crash. 5 errors in a half = forfeit = we lose the match.
- Submitting uses an exact commit ID. Agree on which commit before filling in the form.
