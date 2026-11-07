# Team notes: start here

Read `README.md` for the tournament rules. This file covers what we built and how to work on it.

## Handoff for a fresh session (2026-11-07)

**Submitted commit:** `070ce721b1a512f6fc949d1c72536dd7ab237624` on `aayush` (remote `aura` = github.com/Utsavdotcom/aura-farmer). Upstream `origin` = tamu-datathon-org/heads-up-bundle; its `game.py` / `evaluator.py` / `judge_engine.py` are byte-identical to ours (checked 2026-11-07). Never edit those, `docker/` or `baseline-bots/`.

| Branch | Tip | What |
|---|---|---|
| `aayush` | this commit (submitted code = `070ce72`) | main line |
| `opp-history-fix` | `3585a7d` | opponent-stats fix (merged) |
| `time-safety` | `070ce72` | fixed sample counts, field clones (merged) |

Not yet done: `tools/check_build.sh` (no Docker on the dev laptop). This is Windows: use `.venv/Scripts/python.exe`, and `http://127.0.0.1:PORT` for the judge (`localhost` fails here).

**Seeds:** tune on `sim0`-`sim7` (`tools/sim.py`); confirm only on `hold0`-`hold7` (`tools/field_eval.py`). Never tune on hold seeds.

**Commands**
```
.venv/Scripts/python.exe -m pytest -q                                   # 42 tests
.venv/Scripts/python.exe tools/sim.py --opp learner,tag,nit,lag,always_pot --deals 1000 --seeds 8
.venv/Scripts/python.exe tools/field_eval.py                            # f1-f4 + learner on hold0-hold7, with leak table
# judge timing (fresh servers per half, like the tournament):
PORT=5008 .venv/Scripts/python.exe agent.py   &   PORT=5009 .venv/Scripts/python.exe baseline-bots/baseline_always_pot.py &
.venv/Scripts/python.exe judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009 --half 1 --seed 7
```
Paired before/after comparisons and calibrated timing used scratch scripts (session-temp folder, may be gone):
`C:\Users\rajes\AppData\Local\Temp\claude\a32a1c84fb274f89\3640397d-4643-470b-b94e-c4a1259c8aa6\scratchpad\`
- `dump.py <root> <out.json> <opps> <seeds> <deals>` then `compare.py <old.json> <new.json>`: per-deal paired diff with 95% CI. Run the old version from a `git archive <sha>` export.
- `calib.py <root> <label> <opp> <seed> <half>`: judge-charged vs internal time (`SLOW_FACTOR=1.5` emulates a slower CPU).

**Competitor bots (upstream open PRs #1 and #2), for testing only, never copy or commit their code:**
`...\scratchpad\competitors\pr1`, `...\pr2` (exports), `...\venv_comp` (throwaway venv with numba 0.68.1 for PR #2), `...\comp_eval.py`, results in `...\comp_results.txt`. Run:
`venv_comp\Scripts\python.exe comp_eval.py <070ce72 export dir> pr2 PR2 8 1000`. PR #2's `agent.py` hides code in an `exec(zlib/base64)` blob; it was decoded and reviewed (no network/file/process calls) before running.

**Key results so far** (bb/deal, duplicate, 8000 deals each)
- Timing: judge-charged at most 3.1 s per half (4.8 s at 1.5x slower CPU), 0 errors. Decisions are fully repeatable. Open risk: on a contended CPU the judge charged 6.4 s while our timer read 4.5 s.
- `sim` seeds vs 3585a7d: no opponent significantly worse after the time-safety change (`bench/phase1_paired.txt`).
- Held-out: F1 +4.00, F2 +1.63, F3 +5.79 (wins); **F4 -0.97, learner -0.66 (losses)**; **PR #2 -2.17 (loss)**; PR #1 -0.18 (chop).
- Main leak: we fold too often after checking. F4 and PR #2 bet ~90-97% of the time when we check and our turn/river folds to half-pot bets lose the most. Next phase: defend checks (minimum-defence calling), use our own tracked fold rate (`my_folded`), add deliberate check-raises/bluffs.

## Setup (5 minutes)
```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt numpy
.venv/bin/python -m pytest -q          # should say: 42 passed
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
