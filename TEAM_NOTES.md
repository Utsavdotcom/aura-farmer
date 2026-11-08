# Team notes: start here

Read `README.md` for the tournament rules. This file covers what we built and how to work on it.

## Handoff (branch `aayush`)

- Never edit `game.py`, `evaluator.py`, `judge_engine.py`, `docker/` or `baseline-bots/`.
- On Windows use `.venv/Scripts/python.exe` and `http://127.0.0.1:PORT` (`localhost` does not resolve for the local judge).
- Tune on seeds `sim0`-`sim7`. Confirm once on fresh `hold` seeds and never adjust after seeing a hold result. `hold32`-`hold39` are reserved for the final check.

## Where things are
| File | What it does |
|---|---|
| `agent.py` | The web server the tournament talks to. Only hands over to `strategy.py`. Don't change the server code. |
| `strategy.py` | **The brain.** All decisions are made here. Settings are at the top. |
| `fasteval.py` | Fast hand scorer (numpy). Gives the same scores as `evaluator.py` (tested). |
| `preflop.py` + `data/preflop.json` | Starting-hand strength table for all 20k kinds of starting hands. |
| `tools/build_preflop.py` | Rebuilds that table (about 20 s). Only needed if the rules change. |
| `tools/sim.py` | Fast test matches without servers. **Use this to test every change.** |
| `tools/sparring.py` | Practice bots: checkfold, nit, tag, lag, sticky, learner, self. |
| `tools/field/` | Our own field opponents F1-F4 (equity, rules, aggro, exploit). |
| `tools/field_eval.py` | Runs the field opponents and learner on hold seeds with a leak table. |
| `test_agent.py` | Tests. Run them before every push. |

## How the bot decides (strategy.py)
1. **Win chance ("equity"):** a table lookup before the flop. After the flop it plays out a fixed 150-300 random endings (more when facing a bet), seeded from the cards and betting so the same situation always gives the same decision. If the top two moves are within `CLOSE_GAP` of each other it draws `CLOSE_EXTRA` more endings and decides again.
2. **Opponent's likely hands:** when the opponent raises, it narrows them down, more so if that opponent rarely raises. A half-pot bet right after our check does not narrow (`HALF_STAB_NARROW`).
3. **Truth check:** at showdown it checks whether the opponent's raises really came with strong hands. If not, it trusts them less.
4. **Fold rate:** it tracks how often this opponent folds to our bets, overall and (turn/river, by bet size) for our first-in bets (`bet_fold`). A check on the turn/river becomes a bet when the measured fold rate makes the bet the best move (`_measured_bet`).
5. **Chips per move:** it works out the expected chips for fold, call, half-pot raise and pot raise.
6. **Reads that only turn folds into calls** (each per half, from hand records):
   - facing a half-pot bet after our check from an opponent that bets at least `STAB_GATE_RATES` of the times we check to it, the call is valued against its whole range (`_wide_call`);
   - on the river, facing a half-pot bet after our check or a half-pot lead, we call when our smoothed showdown win share after such calls beats the 25% pot odds (`riv_sd`, `riv_lead_sd`).
7. **Mixing:** it picks the best move, or sometimes a close one, so we're harder to read (`_mix`).

Everything it learns resets every half (the tournament rule). The time guard switches to cheap mode (`CHEAP_SAMPLES`) if a half is projected to exceed `PROJECTED_LIMIT` seconds, and to check/fold past `HARD_LIMIT`.

## Our test opponents
- **Baselines:** `baseline-bots/` (always_pot and friends).
- **Sparring:** `tools/sparring.py` (nit, tag, lag, learner, ...).
- **Field F1-F4:** `tools/field/` on this branch. **F5** (a range reader that drops its sample count under time pressure, so it varies with CPU load) lives on the `f5-reader` branch together with `tools/points.py` (expected tournament points).
- **Red team R1-R6** (`redteam` branch): pot-bluffer, size-flipper, jammer, trapper, style-switcher, barreler. **Held out: never tune or choose on them.**
- **Adaptive A1-A4** (`adaptive` branch): re-raiser, bluff-catcher, balanced, bandit (three styles, picks the most profitable). Held out as well.
- `final-kit` branch: `tools/final_check.py` (full field run on hold32-39), `tools/stress_judge.py`, `FINAL_CHECKLIST.md`.

Load-sensitive opponents (F5, A4) give noisy results: always run the old and the new version side by side under the same load.

## Keep rule
Keep a change if total expected tournament points over all our opponents rise by at least 0.2, no opponent we beat drops below P(win) 0.5, and no opponent we care about is significantly worse (whole 95% interval below zero). Choose variants on sim seeds only; confirm once on fresh hold seeds. If kept: judge-charged thinking time at most 4.0 s per half, determinism check, full tests, then merge with a merge commit and push.

## How to test a change
```
.venv/Scripts/python.exe -m pytest -q                              # unit tests
.venv/Scripts/python.exe tools/sim.py --deals 1000 --seeds 3        # vs every bot (~1-2 min)
.venv/Scripts/python.exe tools/sim.py --opp learner,tag --deals 1000 --seeds 3
.venv/Scripts/python.exe tools/field_eval.py                        # field opponents + learner, leak table
```
- Results are in big blinds per deal. WIN means the whole 95% range is above 0.
- Results move around by about +-0.3, so use `--seeds 3` or more before believing a small change.
- Thinking time (judge-charged, laptop): about 3-3.5 s of the 8 s per half in full mode. 0 errors.

## Known issues
- `test_same_deals_give_identical_decisions` is flaky on slow or cold runs (early decisions fall into cheap mode). Rerun it alone on an idle machine; it is not a regression unless it also fails there.
- The determinism check for a change is best done by playing the same seeds twice with the time fallback off and comparing decisions deal by deal.

## Git workflow
- Branch per change off `aayush`, one commit per change, tests green before merging.
- Merge with `git merge --no-ff <branch>`; run `git log --oneline -5` after every git step.
- Revert with `git revert --no-edit <sha>`. No rebase, no force push, no amend of pushed commits.
- Never chain `git push` after a test command without checking the test result first. Push `aayush` to the `aura` remote only.
- Submitting uses an exact commit ID. Agree on which commit before filling in the form.

## Rules for working together
- Run `pytest` and `tools/sim.py` before pushing.
- Only keep a change if it doesn't make any opponent noticeably worse.
- Never let the bot crash. 5 errors in a half = forfeit = we lose the match.
- Set the team name: `PARTICIPANT` / `AGENT_NAME` in `agent.py`.
