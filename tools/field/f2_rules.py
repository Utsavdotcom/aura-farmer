"""
f2_rules.py - field practice opponent F2: hand-rules bot.

Tight preflop points chart (high pairs, double-suited, connected, high cards),
then postflop rules by made-hand category and draws (flush draw, straight outs).
    python tools/field/f2_rules.py      (PORT env var, default 5009)
"""

import os
import sys
import traceback

# This file lives in tools/field/: put the repo root (game.py, evaluator.py)
# and this folder (common.py) on the path, however the file is loaded.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from game import Action, AgentConfig, HandResult, Observation, Street

import common as C  # noqa: E402  (builds the numpy score table at import, < 1 s)

PARTICIPANT = "field"
AGENT_NAME = "F2-Rules"


# ===== YOUR CODE HERE =====

from collections import Counter


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands"] = 0


def preflop_points(hole) -> float:
    ranks = [c >> 2 for c in hole]
    suits = Counter(c & 3 for c in hole)
    rc = Counter(ranks)
    pts = 0.0
    pairs = [r for r, n in rc.items() if n >= 2]
    if pairs:
        top = max(pairs)
        pts += 3 if top >= 6 else (2 if top >= 4 else 1)   # QQ+ / TT-JJ / lower
    if max(rc.values()) >= 3:
        pts -= 2                                             # trips in hand are bad in Omaha
    two_plus = [s for s, n in suits.items() if n >= 2]
    if len(two_plus) >= 2:
        pts += 2                                             # double-suited
    elif two_plus:
        s = two_plus[0]
        pts += 1 if any((c >> 2) == 8 and (c & 3) == s for c in hole) else 0.5
    if max(suits.values()) >= 4:
        pts -= 1                                             # too many of one suit
    distinct = set(ranks)
    window = max(len(distinct & set(range(lo, lo + 5))) for lo in range(5))
    window = max(window, len(distinct & {8, 0, 1, 2, 3}))
    pts += 2 if window >= 4 else (1 if window == 3 else 0)
    if sum(1 for r in ranks if r >= 4) >= 3:
        pts += 1                                             # three or more T+ cards
    return pts


def _preflop(obs) -> Action:
    pts = preflop_points(obs.hole_cards)
    if pts >= 5:
        if obs.raises_this_street < 3:
            return C.legalize(obs, Action.RAISE_POT)
        return Action.CHECK_CALL
    if pts >= 3.5:
        if obs.to_call <= 1 and obs.raises_this_street == 0:
            return C.legalize(obs, Action.RAISE_HALF)
        if obs.to_call <= 12:
            return Action.CHECK_CALL
        return C.legalize(obs, Action.FOLD)
    if pts >= 2:
        if obs.to_call <= 1:                                 # complete the SB / check the BB
            return Action.CHECK_CALL
        if obs.to_call <= 4 and obs.raises_this_street <= 1:
            return Action.CHECK_CALL
        return C.legalize(obs, Action.FOLD)
    return C.legalize(obs, Action.FOLD)


def _classify(obs):
    """Return (strength, draw): strength in nuts/strong/medium/weak;
    draw 0 (none), 1 (draw), 2 (strong draw)."""
    hole, board = obs.hole_cards, obs.board
    score = C.best_score(hole, board)
    cat = C.category_of(score)
    branks = [c >> 2 for c in board]
    bsuits = Counter(c & 3 for c in board)
    paired = len(set(branks)) < len(branks)
    flushy = max(bsuits.values()) >= 3
    if cat >= C.FOUR_OF_A_KIND or (cat == C.FLUSH and not paired):
        strength = "nuts"
    elif cat >= C.FULL_HOUSE:
        strength = "strong"
    elif cat == C.THREE_OF_A_KIND:
        strength = "medium" if (paired or flushy) else "strong"
    elif cat == C.STRAIGHT:
        strength = "medium" if (flushy or paired) else "strong"
    elif cat == C.TWO_PAIR:
        strength = "weak" if paired else "medium"
    elif cat == C.ONE_PAIR:
        pair_rank = (score >> 16) & 15                       # top pair or overpair
        strength = "medium" if (pair_rank >= max(branks) and not paired) else "weak"
    else:
        strength = "weak"
    fd = C.flush_draw(hole, board)
    so = C.straight_outs(hole, board)
    draw = 0
    if len(board) < 5:
        if fd == 2 or (fd and so >= 2) or so >= 4:
            draw = 2
        elif fd or so >= 2:
            draw = 1
    return strength, draw


def _postflop(obs) -> Action:
    strength, draw = _classify(obs)
    frac = C.opp_bet_fraction(obs)
    odds = C.pot_odds(obs)
    river = obs.street == Street.RIVER
    if strength in ("nuts", "strong"):
        if obs.raises_this_street < 3 or strength == "nuts":
            return C.legalize(obs, Action.RAISE_POT)
        return Action.CHECK_CALL
    if strength == "medium":
        if obs.to_call == 0:
            return Action.CHECK_CALL if river else C.legalize(obs, Action.RAISE_HALF)
        if frac <= 0.6 or (draw and frac <= 1.0):
            return Action.CHECK_CALL
        if obs.raises_this_street <= 1 and frac <= 1.0 and not river:
            return Action.CHECK_CALL
        return C.legalize(obs, Action.FOLD)
    # weak made hand
    if draw == 2:
        if obs.to_call == 0:
            return C.legalize(obs, Action.RAISE_HALF) if obs.is_button else Action.CHECK_CALL
        return Action.CHECK_CALL if odds <= 0.36 else C.legalize(obs, Action.FOLD)
    if draw == 1:
        if obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.CHECK_CALL if odds <= 0.25 else C.legalize(obs, Action.FOLD)
    return C.legalize(obs, Action.FOLD)


def _decide(obs, memory) -> Action:
    if obs.street == Street.PREFLOP:
        return _preflop(obs)
    return _postflop(obs)


def choose_action(obs: Observation, memory: dict) -> Action:
    try:
        action = C.legalize(obs, _decide(obs, memory))
        if action in obs.legal_actions:
            return action
    except Exception:
        pass
    return C.safe_default(obs)


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        memory["hands"] = memory.get("hands", 0) + 1
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
# Keep your per-half state in `memory`: other global variables are NOT reset
# between halves.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5009 for your own testing convenience
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
