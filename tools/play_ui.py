"""
Local human-vs-bot practice table (plays against agent.py).

Not imported by the agent, so it doesn't change tournament behaviour (the
Dockerfile copies the whole bundle, so the file is present but unused).
Run from the bundle root with the project venv:

    .venv/Scripts/python tools/play_ui.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BUNDLE = Path(__file__).resolve().parent.parent
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from evaluator import card_rank, card_suit, card_to_str, category_name, omaha_score  # noqa: E402
from game import (  # noqa: E402
    Action,
    AgentConfig,
    Game,
    STARTING_STACK,
    Street,
    make_deal,
)

# Import after path setup. agent.warmup() runs on import.
import agent as bot_agent  # noqa: E402

SUIT_GLYPH = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
SUIT_COLOR = {0: "black", 1: "red", 2: "red", 3: "black"}
RANK_LABEL = "6789TJQKA"

app = FastAPI(title="Practice Table")


class TableState:
    def __init__(self) -> None:
        self.seed: str = "practice"
        self.hand_index: int = 0
        self.human_seat: int = 0
        self.game: Optional[Game] = None
        self.memory: dict[str, Any] = {}
        self.log: list[str] = []
        self.session_nets = {"you": 0, "bot": 0}
        self.hands_finished = 0
        bot_agent.new_half(AgentConfig(), self.memory)

    @property
    def bot_seat(self) -> int:
        return 1 - self.human_seat


STATE = TableState()


def _card_view(card: int, face_down: bool = False) -> dict[str, Any]:
    if face_down:
        return {"face_down": True, "label": "?", "suit": "", "color": "back", "text": "??"}
    rank = card_rank(card)
    suit = card_suit(card)
    return {
        "face_down": False,
        "label": RANK_LABEL[rank],
        "suit": SUIT_GLYPH[suit],
        "color": SUIT_COLOR[suit],
        "text": card_to_str(card),
        "id": card,
    }


def _action_label(action: Action, obs) -> str:
    if action == Action.FOLD:
        return "Fold"
    if action == Action.CHECK_CALL:
        return "Check" if obs.to_call == 0 else f"Call {obs.to_call}"
    if action == Action.RAISE_HALF:
        return f"Raise ½ pot → {obs.raise_to[action]}"
    if action == Action.RAISE_POT:
        return f"Raise pot → {obs.raise_to[action]}"
    return action.name


def _bot_act_until_human() -> None:
    game = STATE.game
    assert game is not None
    while not game.is_terminal and game.to_act == STATE.bot_seat:
        obs = game.observation(STATE.bot_seat)
        try:
            action = bot_agent.choose_action(obs, STATE.memory)
            if action not in obs.legal_actions:
                action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
                if action not in obs.legal_actions:
                    action = obs.legal_actions[0]
        except Exception as exc:
            STATE.log.append(f"Bot error ({type(exc).__name__}); using fallback.")
            action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
            if action not in obs.legal_actions:
                action = obs.legal_actions[0]
        game.apply(action)
        STATE.log.append(f"Bot: {_action_label(action, obs)}")


def _finish_if_needed() -> None:
    game = STATE.game
    assert game is not None
    if not game.is_terminal:
        return
    result = game.hand_result(STATE.human_seat)
    try:
        # The bot learns from its own point of view, not the human's.
        bot_agent.on_hand_end(game.hand_result(STATE.bot_seat), STATE.memory)
    except Exception:
        pass
    STATE.session_nets["you"] += result.my_net
    STATE.session_nets["bot"] -= result.my_net
    STATE.hands_finished += 1
    if result.winner == "me":
        outcome = "You win"
    elif result.winner == "opp":
        outcome = "Bot wins"
    else:
        outcome = "Split"
    STATE.log.append(f"Hand over — {outcome} ({result.my_net:+d} chips).")


def _snapshot() -> dict[str, Any]:
    game = STATE.game
    if game is None:
        return {
            "ready": False,
            "brand": bot_agent.AGENT_NAME,
            "participant": bot_agent.PARTICIPANT,
            "session": STATE.session_nets,
            "hands_finished": STATE.hands_finished,
            "log": STATE.log[-12:],
        }

    human = STATE.human_seat
    bot = STATE.bot_seat
    show_bot = game.is_terminal and bool(game.result().went_to_showdown)
    if game.is_terminal:
        board_cards = list(game.result().board_revealed)
    else:
        board_cards = list(game.observation(human).board)

    your_obs = game.observation(human) if not game.is_terminal and game.to_act == human else None
    legal = []
    if your_obs is not None:
        legal = [
            {"action": a.name, "label": _action_label(a, your_obs)}
            for a in your_obs.legal_actions
        ]

    hand_score = None
    bot_score = None
    if game.is_terminal and game.result().went_to_showdown:
        hand_score = category_name(omaha_score(game.deal.hole_cards[human], game.deal.board))
        bot_score = category_name(omaha_score(game.deal.hole_cards[bot], game.deal.board))

    return {
        "ready": True,
        "brand": bot_agent.AGENT_NAME,
        "participant": bot_agent.PARTICIPANT,
        "hand_index": STATE.hand_index,
        "seed": STATE.seed,
        "street": "SHOWDOWN" if game.is_terminal else game.street.name,
        "pot": game.pot,
        "to_call": 0 if your_obs is None else your_obs.to_call,
        "your_stack": game.stacks[human] if not game.is_terminal else STARTING_STACK + (
            game.hand_result(human).my_net
        ),
        "bot_stack": game.stacks[bot] if not game.is_terminal else STARTING_STACK - (
            game.hand_result(human).my_net
        ),
        "your_bet": game.bets[human],
        "bot_bet": game.bets[bot],
        "is_button": human == game.button_seat,
        "your_cards": [_card_view(c) for c in game.deal.hole_cards[human]],
        "bot_cards": [
            _card_view(c, face_down=not show_bot) for c in game.deal.hole_cards[bot]
        ],
        "board": [_card_view(c) for c in board_cards],
        "board_slots": max(0, 5 - len(board_cards)),
        "legal_actions": legal,
        "terminal": game.is_terminal,
        "your_hand": hand_score,
        "bot_hand": bot_score,
        "winner": None if not game.is_terminal else game.hand_result(human).winner,
        "your_net": 0 if not game.is_terminal else game.hand_result(human).my_net,
        "session": STATE.session_nets,
        "hands_finished": STATE.hands_finished,
        "log": STATE.log[-12:],
        "waiting_for_you": your_obs is not None,
    }


class NewHandBody(BaseModel):
    seed: str = Field(default="practice")
    human_seat: int = Field(default=0, ge=0, le=1)


class ActBody(BaseModel):
    action: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return _snapshot()


@app.post("/api/new-hand")
def api_new_hand(body: NewHandBody) -> dict[str, Any]:
    if STATE.game is not None:
        STATE.hand_index += 1
    STATE.seed = body.seed or "practice"
    STATE.human_seat = body.human_seat
    deal = make_deal(STATE.seed, STATE.hand_index)
    button = STATE.hand_index % 2
    STATE.game = Game(deal, button_seat=button, hand_index=STATE.hand_index)
    STATE.log = [
        f"Hand #{STATE.hand_index + 1} · seed {STATE.seed!r} · "
        f"you are seat {STATE.human_seat} "
        f"({'button / SB' if STATE.human_seat == button else 'BB'})"
    ]
    _bot_act_until_human()
    _finish_if_needed()
    return _snapshot()


@app.post("/api/act")
def api_act(body: ActBody) -> dict[str, Any]:
    game = STATE.game
    if game is None:
        raise HTTPException(400, "Start a hand first.")
    if game.is_terminal:
        raise HTTPException(400, "Hand is over. Deal a new hand.")
    if game.to_act != STATE.human_seat:
        raise HTTPException(400, "Not your turn.")
    try:
        action = Action[body.action]
    except KeyError as exc:
        raise HTTPException(400, f"Unknown action {body.action!r}") from exc
    obs = game.observation(STATE.human_seat)
    if action not in obs.legal_actions:
        raise HTTPException(400, f"{action.name} is not legal right now.")
    game.apply(action)
    STATE.log.append(f"You: {_action_label(action, obs)}")
    _bot_act_until_human()
    _finish_if_needed()
    return _snapshot()


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Practice Table</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --felt: #0f3d2e;
      --felt-deep: #0a2a20;
      --rail: #1a120c;
      --cream: #f3ebe0;
      --ink: #140f0c;
      --gold: #d4a85a;
      --mist: rgba(243, 235, 224, 0.72);
      --danger: #c45c4a;
      --ok: #7cb389;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; min-height: 100%;
      background:
        radial-gradient(1200px 700px at 50% -10%, #1d4d3a 0%, transparent 55%),
        linear-gradient(165deg, #071910 0%, #10281f 45%, #07140f 100%);
      color: var(--cream);
      font-family: "Manrope", system-ui, sans-serif;
    }
    body {
      display: grid;
      place-items: center;
      padding: 1.25rem;
    }
    .shell {
      width: min(980px, 100%);
      display: grid;
      gap: 1rem;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .brand {
      font-family: "Instrument Serif", Georgia, serif;
      font-size: clamp(2.4rem, 5vw, 3.4rem);
      line-height: 0.9;
      letter-spacing: 0.02em;
      margin: 0;
    }
    .brand em {
      font-style: italic;
      color: var(--gold);
    }
    .sub {
      margin: 0.35rem 0 0;
      color: var(--mist);
      font-size: 0.95rem;
      max-width: 36rem;
    }
    .session {
      text-align: right;
      font-size: 0.9rem;
      color: var(--mist);
    }
    .session strong { color: var(--cream); font-weight: 600; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      align-items: center;
      background: rgba(0,0,0,0.22);
      border: 1px solid rgba(243,235,224,0.08);
      padding: 0.75rem;
    }
    label { font-size: 0.8rem; color: var(--mist); display: grid; gap: 0.25rem; }
    input, select {
      background: #120e0b;
      color: var(--cream);
      border: 1px solid rgba(243,235,224,0.18);
      padding: 0.45rem 0.6rem;
      font: inherit;
      min-width: 8rem;
    }
    button {
      font: inherit;
      cursor: pointer;
      border: 0;
      padding: 0.65rem 1rem;
      background: var(--gold);
      color: var(--ink);
      font-weight: 700;
      letter-spacing: 0.02em;
      transition: transform 120ms ease, filter 120ms ease;
    }
    button:hover { filter: brightness(1.05); transform: translateY(-1px); }
    button:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
    button.ghost {
      background: transparent;
      color: var(--cream);
      border: 1px solid rgba(243,235,224,0.28);
      font-weight: 600;
    }
    button.fold { background: var(--danger); color: #fff; }
    button.call { background: var(--ok); color: #0b1a12; }
    .table {
      position: relative;
      min-height: 520px;
      border-radius: 999px / 180px;
      background:
        radial-gradient(ellipse at center, #176048 0%, var(--felt) 55%, var(--felt-deep) 100%);
      box-shadow:
        0 0 0 14px #2a1c14,
        0 0 0 18px #4a3424,
        0 30px 80px rgba(0,0,0,0.45);
      padding: 2rem 2.5rem 1.5rem;
      display: grid;
      grid-template-rows: auto 1fr auto auto;
      gap: 1rem;
      align-items: center;
      justify-items: center;
      overflow: hidden;
    }
    .table::before {
      content: "";
      position: absolute; inset: 18px;
      border: 1px solid rgba(243,235,224,0.12);
      border-radius: 999px / 160px;
      pointer-events: none;
    }
    .seat {
      display: grid;
      gap: 0.45rem;
      justify-items: center;
      z-index: 1;
    }
    .seat-meta {
      font-size: 0.82rem;
      color: var(--mist);
      text-align: center;
    }
    .seat-meta b { color: var(--cream); }
    .cards {
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
      justify-content: center;
    }
    .card {
      width: 52px;
      height: 74px;
      border-radius: 6px;
      background: #f7f1e8;
      color: #1a120c;
      display: grid;
      place-content: center;
      font-weight: 700;
      box-shadow: 0 8px 16px rgba(0,0,0,0.25);
      animation: deal 280ms ease both;
    }
    .card .rank { font-size: 1.15rem; line-height: 1; text-align: center; }
    .card .suit { font-size: 1.05rem; text-align: center; }
    .card.red { color: #b42318; }
    .card.black { color: #171310; }
    .card.back {
      background:
        repeating-linear-gradient(135deg, #2c1810 0 6px, #3a2218 6px 12px);
      color: transparent;
      border: 1px solid rgba(243,235,224,0.25);
    }
    .card.slot {
      background: transparent;
      border: 1px dashed rgba(243,235,224,0.22);
      box-shadow: none;
    }
    .board-wrap {
      display: grid;
      gap: 0.5rem;
      justify-items: center;
      z-index: 1;
    }
    .pot {
      font-family: "Instrument Serif", Georgia, serif;
      font-size: 1.55rem;
      color: var(--gold);
    }
    .street {
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.72rem;
      color: var(--mist);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      justify-content: center;
      min-height: 2.6rem;
      z-index: 1;
    }
    .banner {
      z-index: 1;
      min-height: 1.4rem;
      font-size: 0.95rem;
      color: var(--gold);
      text-align: center;
    }
    .log {
      background: rgba(0,0,0,0.28);
      border: 1px solid rgba(243,235,224,0.08);
      padding: 0.85rem 1rem;
      font-size: 0.86rem;
      color: var(--mist);
      max-height: 9rem;
      overflow: auto;
    }
    .log div { margin: 0.15rem 0; }
    @keyframes deal {
      from { opacity: 0; transform: translateY(-10px) scale(0.96); }
      to { opacity: 1; transform: none; }
    }
    @media (max-width: 720px) {
      .table {
        border-radius: 28px;
        min-height: 560px;
        padding: 1.25rem;
      }
      .table::before { border-radius: 22px; inset: 10px; }
      .card { width: 44px; height: 64px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1 class="brand" id="brand">Practice <em>Table</em></h1>
        <p class="sub">Practice table — you vs your bot on Short Deck PLO5. Local only; not part of the tournament image.</p>
      </div>
      <div class="session" id="session">Session: —</div>
    </header>

    <div class="controls">
      <label>Seed
        <input id="seed" value="practice" />
      </label>
      <label>Your seat
        <select id="seat">
          <option value="0">Seat 0</option>
          <option value="1">Seat 1</option>
        </select>
      </label>
      <button id="deal" type="button">Deal hand</button>
      <button class="ghost" id="refresh" type="button">Refresh</button>
    </div>

    <section class="table" aria-label="Poker table">
      <div class="seat" id="bot-seat">
        <div class="seat-meta"><b id="bot-name">Bot</b> · stack <span id="bot-stack">80</span> · bet <span id="bot-bet">0</span></div>
        <div class="cards" id="bot-cards"></div>
        <div class="seat-meta" id="bot-hand"></div>
      </div>

      <div class="board-wrap">
        <div class="street" id="street">PREFLOP</div>
        <div class="pot" id="pot">Pot 0</div>
        <div class="cards" id="board"></div>
      </div>

      <div class="banner" id="banner"></div>
      <div class="actions" id="actions"></div>

      <div class="seat" id="you-seat">
        <div class="cards" id="your-cards"></div>
        <div class="seat-meta"><b>You</b> · stack <span id="your-stack">80</span> · bet <span id="your-bet">0</span> · <span id="button-tag"></span></div>
        <div class="seat-meta" id="your-hand"></div>
      </div>
    </section>

    <div class="log" id="log"></div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);

    function cardEl(c) {
      const d = document.createElement("div");
      if (c.slot) {
        d.className = "card slot";
        return d;
      }
      d.className = "card " + (c.face_down ? "back" : c.color);
      if (c.face_down) {
        d.innerHTML = "<div class='rank'>?</div>";
        return d;
      }
      d.innerHTML = `<div class="rank">${c.label}</div><div class="suit">${c.suit}</div>`;
      d.title = c.text;
      return d;
    }

    function renderCards(el, cards, emptySlots = 0) {
      el.innerHTML = "";
      for (const c of cards) el.appendChild(cardEl(c));
      for (let i = 0; i < emptySlots; i++) el.appendChild(cardEl({ slot: true }));
    }

    function render(state) {
      if (state.brand) {
        $("bot-name").textContent = state.brand;
        $("brand").innerHTML = `${state.brand} <em>Practice</em>`;
        document.title = `${state.brand} Practice Table`;
      }
      $("session").innerHTML =
        `Hands <strong>${state.hands_finished}</strong> · You <strong>${state.session.you}</strong> · Bot <strong>${state.session.bot}</strong>`;
      $("log").innerHTML = (state.log || []).map(line => `<div>${line}</div>`).join("") || "<div>Deal a hand to begin.</div>";

      if (!state.ready) {
        $("banner").textContent = "Deal a hand to sit at the table.";
        $("actions").innerHTML = "";
        return;
      }

      $("street").textContent = state.street;
      $("pot").textContent = `Pot ${state.pot}`;
      $("bot-stack").textContent = state.bot_stack;
      $("your-stack").textContent = state.your_stack;
      $("bot-bet").textContent = state.bot_bet;
      $("your-bet").textContent = state.your_bet;
      $("button-tag").textContent = state.is_button ? "Button / SB" : "Big Blind";
      $("bot-hand").textContent = state.bot_hand ? state.bot_hand : "";
      $("your-hand").textContent = state.your_hand ? state.your_hand : "";

      renderCards($("bot-cards"), state.bot_cards);
      renderCards($("your-cards"), state.your_cards);
      renderCards($("board"), state.board, state.board_slots);

      if (state.terminal) {
        const map = { me: "You win this hand", opp: "Bot wins this hand", split: "Split pot" };
        $("banner").textContent = `${map[state.winner] || "Hand over"} (${state.your_net >= 0 ? "+" : ""}${state.your_net})`;
      } else if (state.waiting_for_you) {
        $("banner").textContent = state.to_call ? `To call: ${state.to_call}` : "Your action";
      } else {
        $("banner").textContent = "Bot acting…";
      }

      const actions = $("actions");
      actions.innerHTML = "";
      if (state.waiting_for_you) {
        for (const a of state.legal_actions) {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = a.label;
          if (a.action === "FOLD") b.className = "fold";
          else if (a.action === "CHECK_CALL") b.className = "call";
          b.onclick = () => act(a.action);
          actions.appendChild(b);
        }
      } else if (state.terminal) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = "Next hand";
        b.onclick = () => newHand();
        actions.appendChild(b);
      }
    }

    async function api(path, body) {
      const res = await fetch(path, {
        method: body ? "POST" : "GET",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || res.statusText);
        return null;
      }
      return res.json();
    }

    async function refresh() {
      const state = await api("/api/state");
      if (state) render(state);
    }

    async function newHand() {
      const state = await api("/api/new-hand", {
        seed: $("seed").value || "practice",
        human_seat: Number($("seat").value),
      });
      if (state) render(state);
    }

    async function act(action) {
      const state = await api("/api/act", { action });
      if (state) render(state);
    }

    $("deal").onclick = newHand;
    $("refresh").onclick = refresh;
    refresh();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PLAY_UI_PORT", "8765"))
    print(f"{bot_agent.AGENT_NAME} practice table → http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
