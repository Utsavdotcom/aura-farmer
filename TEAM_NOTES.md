# Team notes: start here

Read `README.md` for the tournament rules. This file covers what we built and how to work on it.

## Handoff for a fresh session (updated 2026-11-07, evening)

**Submitted version:** `8f2681a2a069c78614b2c631c282571ac4110193` on `aayush` (remote `aura` = github.com/Utsavdotcom/aura-farmer; this notes commit sits on top of it and changes no code). Upstream `origin` = tamu-datathon-org/heads-up-bundle. Never edit `game.py`, `evaluator.py`, `judge_engine.py`, `docker/` or `baseline-bots/`. Windows laptop: use `.venv/Scripts/python.exe` and `http://127.0.0.1:PORT` (`localhost` fails). 73 tests.

**What each merged change does** (since `070ce72`, all in `strategy.py` + tests)
| Merge | Change |
|---|---|
| `e09a3a0` (branch `defense`) | **Close-decision resampling** (`0a9870f`): if our top two moves are within 0.03 pots (`CLOSE_GAP`), draw 400 more samples (`CLOSE_EXTRA`) from the same seeded generator and decide again. Full mode only; still repeatable. The check-bet range reading in the same branch (`97cf51f`) was reverted (`b7199c1`). |
| `2f5030c` (branch `defense3`) | **Half-pot stab = no narrowing** (`7dc4064`): when the opponent bets HALF pot right after we check (any street), that bet does not narrow its range (`HALF_STAB_NARROW = 0`). Pot bets, leads and raises of our bets narrow as before. The whole-range variant `8d23ddc` was reverted (`d9ff978`, nit got worse). |
| `8f2681a` (branch `defense6`) | **Whole-range CALL vs frequent stabbers** (`4cd19f2` + `f4d0778`): per street we count, per half, how often it bets (and bets half pot) when we check to it (`stats["chk_bet"] = [bets, half-pot bets, chances]`). Facing its half-pot bet after our check, if it bets >= 50% of those chances with >= 20 chances (`STAB_GATE_KIND = "any"`, `STAB_GATE_RATES`, `STAB_GATE_MIN`), the CALL is valued with our equity against its whole range (`_wide_call`). Raise EVs keep the narrowed equity and a spot where raising is already best is untouched, so this only turns folds into calls. `"half"` (half-pot >= 0.23) is the tested alternative (lost on sim, F1 worse). |

**Branches and worktrees**
| Branch | Tip | Where | What |
|---|---|---|---|
| `aayush` | `8f2681a` + this notes commit | `heads-up-bundle` (this folder), pushed | main line, submitted |
| `defense` | `e09a3a0` | local only | resampling (merged) |
| `defense2` | `81bbdc6` | local only | half-pot fix + its revert (rejected under the old rule; superseded by defense3) |
| `defense3` | `d9ff978` | local only | version A + B + revert of B (merged) |
| `defense4` | `b68d9a8` | local only | whole-range read gated on any-bet >= 0.50 + revert (R1 significantly worse on hold24-31) |
| `defense5` | `d487d1a` | local only, not merged | same, gated on half-pot >= 0.23 (F5 and R1 significantly worse on hold40-47) |
| `defense6` | `f4d0778` | local only | call/fold-only version (merged as `8f2681a`) |
| `f5-reader` | `071fee4` | pushed | `tools/field/f5_reader.py` (F5) and `tools/points.py` (expected tournament points) |
| `redteam` | `86bc74d` | pushed | held-out red-team bots R1-R6 (`tools/field/r*.py`, `rt_common.py`) and an updated F5 (cold-start probe, time guard) |
| `final-kit` | `a6be03f` | pushed; checked out in worktree `C:/Users/rajes/OneDrive/Desktop/aura-f5` (clean) | `tools/final_check.py` (final field run, hold32-39), `tools/stress_judge.py`, `FINAL_CHECKLIST.md`, jish4 support |
| `adaptive` | - | **not found** locally, on `aura` or on `origin` (checked 2026-11-07) | - |

**Test opponents (other teams' code is for testing only: never copy or commit it)**
- F1-F4: `tools/field/` on `aayush`. Learner/tag/nit/lag: `tools/sparring.py`. always_pot etc.: `baseline-bots/`.
- F5: `tools/field/f5_reader.py` on `f5-reader` (`071fee4`) and the newer one on `redteam`/`final-kit` (`86bc74d`). The 86bc74d F5 drops to 300 samples when its own thinking time runs high, so it varies with CPU load.
- R1-R6: `redteam` `86bc74d`. **Held out: never tune or choose on them.** R1 bets pot as a bluff and half pot only with strong hands; R2 flips sizes, R3 jams, R4 traps, R5 switches styles, R6 barrels.
- PR #1 / PR #2 (upstream open PRs): exports in `C:\Users\rajes\AppData\Local\Temp\claude\a32a1c84fb274f89\3640397d-4643-470b-b94e-c4a1259c8aa6\scratchpad\competitors\pr1` and `...\pr2`, run with that folder's `venv_comp\Scripts\python.exe` (numba 0.68.1). PR #2's code was decoded and reviewed before running.
- jish4 (team quad-j): clone of github.com/650JoshLee/quad-j-jish @ `9e062bd` at `C:\Users\rajes\AppData\Local\Temp\claude\a32a1c84fb274f89\5ff8e6df-1406-4a4e-ae06-57c05c2ab89d\scratchpad\jish\quad-j-jish` (root `agent.py` = `jish_4/agent.py`, uses `data/preflop.npy`), run under `venv_comp`.
- Easiest full run: in the `aura-f5` worktree, `.venv/Scripts/python.exe tools/final_check.py <sha> --opp f1,pr2,jish4 --seed-start 56 --seeds 8` (defaults: every opponent, hold32-39, `--comp-dir` and `--jish-dir` as above, PR/jish under venv_comp). It runs our real time guard, so keep `--jobs 1` unless you watch the `cheap` column.

**Scratch harness** (session temp folder, may be deleted): `S = C:\Users\rajes\AppData\Local\Temp\claude\bf70ebd202af69d9\cca85991-78f1-47b0-8db1-d90f27cd292e\scratchpad`. Export each version first: `git archive <sha> | tar -x -C S/<name>`. Copies of F5/R1-R6 from `86bc74d` are in `S/rt86/tools/field`, F5 from `071fee4` in `S/f5tools`; jish4 and PR bots are loaded from the paths above.
- **Paired comparison:** `python S/pdump.py <export dir> <out.json> <opp> <sim|hold> <nseeds> <deals> [first seed]` plays duplicate halves and saves per-deal results, the leak table (by our last decision) and the opponent's bet rate after our checks. Opp names: f1-f4, f5 (86bc74d), f5_071, r1-r6, learner/tag/nit/lag, always_pot, pr1, pr2, jish4 (use venv_comp python for pr1/pr2/jish4). It switches OUR time fallback off (CPU contention), so judge timing is checked separately. Batch runner: `bash S/runjobs.sh <jobfile> <parallel> <sim|hold> <first seed>` (job line: `python export out.json opp label`; run old and new version of each opponent side by side for the load-sensitive bots). Then `python S/pcompare.py <old dir> <new dir> [opp,opp leak tables]` (paired diff, 95% CI, BETTER/WORSE) and `python S/points_cmp.py <old dir> <new dir>` (expected points via tools/points.py).
- **field_eval:** `tools/field_eval.py` in the repo (f1-f4 + learner on hold0-7 with a leak table; f5/r1-r6 on `final-kit`).
- **Phase 1 scorer:** `python S/phase1_g1.py <export> <opp> <out.json> [nseeds] [deals]` (sim seeds) scores every decision facing a bet right after our check against the opponent's real cards (exact equity), with model equity, narrowed and whole-range; `python S/p1_report.py S` prints the tables (reads `S/p1/<opp>.json`). `S/lead.py` + `S/lead_report.py` do the same for half-pot LEADS.
- **Fold-rate replay:** `python S/foldrates.py <export> <out.json> sim 1 <opp,...>` records, for each of our bets/raises, the fold chance the model used and whether the opponent folded; `python S/foldrates_report.py <out.json>`. `python S/foldgain2.py <export> <rates.json> <out.json> sim1 <opp,...>` estimates (with the model's own equities) what measured fold rates would change. Both run at below-normal priority, one process.
- **Judge timing:** `python <old scratch>\calib.py <export> <label> <lag|always_pot> 7 <1|2>` (old scratch = `...\3640397d-4643-470b-b94e-c4a1259c8aa6\scratchpad`, with `debug_server.py`): judge-charged vs internal time per half. Limit 4.0 s.

**Keep rule (current):** keep a change if total expected tournament points (`tools/points.py`, all opponents) rise by >= 0.2, no opponent we beat drops below P(win) 0.5, and none of PR #1, F5, learner or jish4 is significantly worse. Choose variants on sim seeds only; confirm ONCE on fresh hold seeds; never adjust after seeing hold results. If kept: judge timing <= 4.0 s, determinism, full tests, then merge (merge commit) and push.

**Seeds:** tune on `sim0`-`sim7`. Used for confirmations: hold0-31 and hold40-55. **hold32-39 are reserved for the final check** (`final_check.py` default). Next fresh seeds: **hold56+**.

**Known issues**
- `test_same_deals_give_identical_decisions` is flaky on slow or cold runs (early decisions fall into cheap mode). It fails on the old `aayush` too. Rerun on an idle machine, and check determinism with `pdump.py` twice (time fallback off) and compare the outputs deal by deal.
- PR #1 samples until a wall-clock deadline, and PR #2, F5 (86bc74d) and jish4 cut their samples by time spent. Their results vary with CPU load: run old and new side by side, and expect noise even when our decisions do not change (G1 never fired against PR #1, yet PR #1 moved -0.22).
- Git care: revert with `git revert --no-edit <sha>` and run `git log --oneline -5` after every git step. No rebase, no force push. Never chain `git push` after a test command without checking its result.

**Latest results: `8f2681a` on hold48-55** (19 opponents, 8000 duplicate deals each, run side by side with `2f5030c`)
| Opponent | bb/deal | P(win) | Points | Change vs 2f5030c |
|---|---|---|---|---|
| PR #2 | -0.74 | 0.00 | 0.27 | +1.02 (better) |
| jish4 | -0.88 | 0.00 | 0.19 | +1.17 (better) |
| F4 | -0.85 | 0.00 | 0.30 | +0.12 |
| learner | +0.03 | 0.03 | 0.50 | +0.52 (better) |
| PR #1 | +0.00 | 0.03 | 0.50 | -0.22 (noise, see above) |
| F5 | +0.23 | 0.07 | 0.53 | -0.10 |
| F2 | +1.82 | 0.92 | 0.96 | -0.29 (worse) |
| R1 | +1.56 | 0.93 | 0.96 | -0.38 (worse) |
| R2 | +2.09 | 0.99 | 1.00 | -0.25 (worse) |
| R5 / R6 / nit | +2.24 / +1.62 / +1.18 | 0.95 / 0.80 / 0.95 | 0.98 / 0.90 / 0.98 | ~0 |
| F1, F3, tag, lag, always_pot, R3, R4 | +2.0 to +8.0 | >= 0.998 | 1.00 | ~0 |
| **Total** | | | **15.06 of 19** (was 14.58) | |

Where we still lose: vs jish4 and PR #2, river folds to half-pot bets (-1.2 to -1.4 bb/deal) and turn/river folds to pot bets; vs F4, flop folds to pot bets and turn/river play vs half-pot bets; vs PR #1, folds to its pot bets (flop -1.04, turn -0.92 bb/deal). R1's loss comes from our river calls of its (strong) half-pot bets.

**Fold-rate finding:** our model's fold chance for our turn/river POT bets is ~1-4% (`p_fold = 1 - keep/f` goes to 0 once the opponent's range is narrowed), but real fold rates are 31-88% against every opponent (sim0: PR #1 86-88%, jish4 66-70%, PR #2 59-69%, learner 56-61%, F5 62-67%, lag 31-44%). Flop half-pot bets are over-estimated instead vs F1, lag, F5, tag (real 2-9% vs model 17-26%). Preflop and raise fold rates match. A model-only estimate says measured rates would add +1.6 to +6.3 bb/deal vs PR #1/#2, jish4, F5, learner, but that assumes the bots do not adapt: unverified.

**Ranked next fixes (not built)**
1. **Measured fold rate per street and bet size** as `p_fold` (learned per half from hand records, smoothed toward 0.3 worth ~30 bets, capped at 0.75, >= 20 bets per spot), with the calling range made consistent (`e_called` = equity vs its best `1 - p_fold` share). Expected +0.5 to +1.5 points (very uncertain); ~50 min to build; medium-high risk (adaptive bots F5/jish4/PR #2 and R2/R4/R6). **Measure first** with a branching replay on sim seeds: at each spot where the measured rate would change our move, copy the game and the opponent's memory, apply the bet, record its real response and score against real cards. Build only if the true gain is >= +0.3 bb/deal vs PR #1, PR #2, jish4 and learner and no field opponent goes negative. Then sim0-7 vs aayush (10 field opponents) and ONE hold run on hold56+ (19 opponents).
2. **Showdown-calibrated river calls** vs half-pot stabs: call when our smoothed showdown win share after calling its river half-pot bets beats the pot odds (25%). Targets the remaining river-fold leak vs PR #2/jish4 and stops paying off R1. +0.2 to +0.35 points; ~40 min; low-medium risk.
3. **Floor `p_fold` for turn/river pot bets at 0.35** (no per-opponent counters; below every measured rate). +0.1 to +0.3 points; ~25 min; low-medium risk.

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
