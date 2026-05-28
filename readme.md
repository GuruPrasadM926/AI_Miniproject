# 🐍 Autonomous AI Snake — A* Pathfinding Engine

Classic Nokia Snake × Explainable AI. One Python file. Zero human input.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **A\* Pathfinding** with Manhattan & Euclidean heuristics
- **6 AI Strategies**: Safe Path, Tail Following, Dead End Avoidance, Open Space Priority, Survival Mode, Longest Safe Route
- **Real-time AI Decision Log** — every move explained
- **Risk Analyzer** — danger score, survival probability, trapping detection
- **Live analytics** — strategy usage charts, survival probability graph
- **Auto-restart** — snake restarts automatically on death
- **Downloadable CSV logs**
- Retro Nokia-style pixel grid with neon glow aesthetic

## Controls

| Control | Action |
|---------|--------|
| ▶ START | Launch new game |
| ⏸ PAUSE | Pause / Resume |
| ↺ RESTART | New game (keeps high score) |
| Grid Size | 10×10 to 30×30 |
| Speed | 1–20 steps/second |
| Obstacle Density | 0–20% |
| Heuristic | Manhattan / Euclidean |

## AI Strategy Logic

```
1. SAFE_PATH           → Direct path to food if post-move space is safe
2. DEAD_END_AVOIDANCE  → Path to food with partial space check
3. TAIL_FOLLOWING      → Chase own tail when trapped
4. OPEN_SPACE_PRIORITY → Move toward largest open area
5. SURVIVAL_MODE       → Last-resort: best available neighbor
```

## Architecture

All logic lives in `app.py`:

- `astar()` — pure A* with heapq
- `flood_fill()` — BFS space counter
- `analyze_risk()` → `RiskReport` dataclass
- `decide_move()` → `MoveDecision` dataclass
- `GameState` — full game state + analytics
- `render_grid_html()` — SVG grid renderer
- `main()` — Streamlit UI