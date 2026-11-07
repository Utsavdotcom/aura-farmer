"""
f3_aggro.py - field practice opponent F3: aggressive equity bot.

Equity like F1 but aggressive: pot-raises at equity >= 0.55 and with strong
draws (semi-bluff), bluffs ~35% when checked to with some equity, calls wider
than pot odds, re-raises often.
    python tools/field/f3_aggro.py      (PORT env var, default 5009)
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
AGENT_NAME = "F3-AggroEquity"


# ===== YOUR CODE HERE =====

import numpy as np

N_PRE = 300
N_POST = 200


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = np.random.default_rng(3)
    memory["hands"] = 0


def _equity(obs, memory) -> float:
    key = (obs.hand_index, int(obs.street))
    cache = memory.get("eq")
    if cache and cache[0] == key:
        return cache[1]
    rng = memory.setdefault("rng", np.random.default_rng(3))
    n = N_PRE if obs.street == Street.PREFLOP else N_POST
    eq = C.equity_vs_random(obs.hole_cards, obs.board, n, rng)
    memory["eq"] = (key, eq)
    return eq


def _decide(obs, memory) -> Action:
    eq = _equity(obs, memory)
    rng = memory["rng"]
    odds = C.pot_odds(obs)
    raises = obs.raises_this_street
    can_raise = C.can_raise(obs)

    if obs.street == Street.PREFLOP:
        if can_raise and (eq >= 0.55 or (eq >= 0.50 and raises < 2)):
            return C.legalize(obs, Action.RAISE_POT)
        if can_raise and raises == 0 and eq >= 0.44 and rng.random() < 0.5:
            return C.legalize(obs, Action.RAISE_HALF)
        if obs.to_call == 0 or eq >= odds - 0.08:
            return Action.CHECK_CALL
        return Action.FOLD

    strong_draw = obs.street != Street.RIVER and (
        C.flush_draw(obs.hole_cards, obs.board) >= 1 or C.straight_outs(obs.hole_cards, obs.board) >= 3)

    if can_raise:
        if eq >= 0.65:
            return C.legalize(obs, Action.RAISE_POT)
        if eq >= 0.55 and raises < 3:
            return C.legalize(obs, Action.RAISE_POT)
        if strong_draw and raises < 2:
            return C.legalize(obs, Action.RAISE_POT)
        if obs.to_call == 0 and eq >= 0.30 and rng.random() < 0.35:
            return C.legalize(obs, Action.RAISE_POT if rng.random() < 0.5 else Action.RAISE_HALF)
        if obs.to_call > 0 and eq >= 0.40 and raises < 2 and rng.random() < 0.12:
            return C.legalize(obs, Action.RAISE_POT)
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if eq >= odds - 0.06 or (strong_draw and odds <= 0.4):
        return Action.CHECK_CALL
    return Action.FOLD


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
