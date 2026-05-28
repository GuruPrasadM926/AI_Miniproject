"""
╔══════════════════════════════════════════════════════════════════╗
║         AUTONOMOUS AI SNAKE — A* PATHFINDING SYSTEM             ║
║         Classic Nokia Snake × Explainable AI Engine             ║
╚══════════════════════════════════════════════════════════════════╝

Single-file Streamlit application.
Run with:  streamlit run app.py
"""

# ── Standard Library ──────────────────────────────────────────────
import heapq
import math
import random
import time
import json
import io
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Set

# ── Third-party ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# CONSTANTS & ENUMERATIONS
# ─────────────────────────────────────────────────────────────────

class Strategy(Enum):
    SAFE_PATH           = "SAFE_PATH"
    TAIL_FOLLOWING      = "TAIL_FOLLOWING"
    LONGEST_SAFE_ROUTE  = "LONGEST_SAFE_ROUTE"
    SURVIVAL_MODE       = "SURVIVAL_MODE"
    OPEN_SPACE_PRIORITY = "OPEN_SPACE_PRIORITY"
    DEAD_END_AVOIDANCE  = "DEAD_END_AVOIDANCE"

class Heuristic(Enum):
    MANHATTAN = "Manhattan"
    EUCLIDEAN = "Euclidean"

class Cell(Enum):
    EMPTY    = 0
    SNAKE    = 1
    HEAD     = 2
    FOOD     = 3
    OBSTACLE = 4
    PATH     = 5
    EXPLORED = 6

# Retro colour palette
COLORS = {
    Cell.EMPTY:    "#0a0f0a",
    Cell.SNAKE:    "#00cc44",
    Cell.HEAD:     "#ffffff",
    Cell.FOOD:     "#ffdd00",
    Cell.OBSTACLE: "#442200",
    Cell.PATH:     "#005500",
    Cell.EXPLORED: "#001a33",
    "bg":          "#050a05",
    "grid":        "#0d1a0d",
    "neon_green":  "#00ff66",
    "neon_yellow": "#ffee00",
    "neon_red":    "#ff3300",
    "neon_blue":   "#0088ff",
    "panel":       "#090e09",
}

# ─────────────────────────────────────────────────────────────────
# A* PATHFINDER
# ─────────────────────────────────────────────────────────────────

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def astar(
    grid: np.ndarray,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    heuristic: Heuristic = Heuristic.MANHATTAN,
    blocked: Optional[Set[Tuple[int,int]]] = None,
) -> Tuple[Optional[List[Tuple[int,int]]], Set[Tuple[int,int]], int]:
    """
    A* search returning (path, explored_nodes, cost).
    path is None if unreachable.
    """
    rows, cols = grid.shape
    h_fn = manhattan if heuristic == Heuristic.MANHATTAN else euclidean
    blocked = blocked or set()

    open_set: List[Tuple[float,int,Tuple[int,int]]] = []
    counter = 0
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    g_score: Dict[Tuple[int,int], float] = defaultdict(lambda: float('inf'))
    g_score[start] = 0.0
    explored: Set[Tuple[int,int]] = set()

    DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            # Reconstruct path
            path: List[Tuple[int,int]] = []
            node: Optional[Tuple[int,int]] = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, explored, int(g_score[goal])

        r, c = current
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            neighbor = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == Cell.OBSTACLE.value:
                continue
            if grid[nr, nc] == Cell.SNAKE.value and neighbor not in blocked:
                continue
            if neighbor in explored or neighbor in blocked:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + h_fn(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    return None, explored, -1

# ─────────────────────────────────────────────────────────────────
# FLOOD FILL — space counter
# ─────────────────────────────────────────────────────────────────

def flood_fill(grid: np.ndarray, start: Tuple[int,int]) -> int:
    """Count reachable empty cells from start using BFS."""
    rows, cols = grid.shape
    visited = {start}
    queue = deque([start])
    DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
    count = 0
    while queue:
        r, c = queue.popleft()
        count += 1
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            nb = (nr, nc)
            if (0 <= nr < rows and 0 <= nc < cols
                    and nb not in visited
                    and grid[nr, nc] not in (Cell.OBSTACLE.value, Cell.SNAKE.value, Cell.HEAD.value)):
                visited.add(nb)
                queue.append(nb)
    return count

# ─────────────────────────────────────────────────────────────────
# RISK ANALYZER
# ─────────────────────────────────────────────────────────────────

@dataclass
class RiskReport:
    danger_score: float          # 0-100
    survival_prob: float         # 0-100
    food_reachable: bool
    space_after_food: int
    trapping_risk: bool
    open_space: int

def analyze_risk(
    grid: np.ndarray,
    snake: deque,
    food: Tuple[int,int],
    heuristic: Heuristic,
) -> RiskReport:
    head = snake[0]
    rows, cols = grid.shape
    total_cells = rows * cols

    # Reachable space from head
    open_space = flood_fill(grid, head)

    # Can we reach food?
    path, _, _ = astar(grid, head, food, heuristic)
    food_reachable = path is not None

    # Space after eating food (simulate)
    space_after_food = 0
    trapping_risk = False
    if food_reachable and path:
        # Simulate snake after eating
        sim_grid = grid.copy()
        # Remove tail temporarily
        tail = snake[-1]
        sim_grid[tail[0], tail[1]] = Cell.EMPTY.value
        sim_grid[food[0], food[1]] = Cell.HEAD.value
        space_after_food = flood_fill(sim_grid, food)
        trapping_risk = space_after_food < len(snake) + 2

    # Danger score
    danger_score = 0.0
    if not food_reachable:
        danger_score += 50
    if trapping_risk:
        danger_score += 30
    neighbor_walls = sum(
        1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
        if not (0 <= head[0]+dr < rows and 0 <= head[1]+dc < cols)
           or grid[head[0]+dr, head[1]+dc] in (Cell.OBSTACLE.value, Cell.SNAKE.value, Cell.HEAD.value)
    )
    danger_score += neighbor_walls * 5
    danger_score = min(100, danger_score)

    survival_prob = max(0, 100 - danger_score)
    if open_space < len(snake):
        survival_prob *= 0.5

    return RiskReport(
        danger_score=round(danger_score, 1),
        survival_prob=round(survival_prob, 1),
        food_reachable=food_reachable,
        space_after_food=space_after_food,
        trapping_risk=trapping_risk,
        open_space=open_space,
    )

# ─────────────────────────────────────────────────────────────────
# AI BRAIN — Strategy Selector
# ─────────────────────────────────────────────────────────────────

@dataclass
class MoveDecision:
    next_pos: Tuple[int,int]
    strategy: Strategy
    rejected: List[Strategy]
    reason: str
    path: List[Tuple[int,int]]
    explored: Set[Tuple[int,int]]
    path_length: int
    heuristic: Heuristic
    risk: RiskReport

def get_neighbors(pos: Tuple[int,int], grid: np.ndarray) -> List[Tuple[int,int]]:
    rows, cols = grid.shape
    r, c = pos
    return [
        (r+dr, c+dc)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
        if 0 <= r+dr < rows and 0 <= c+dc < cols
        and grid[r+dr, c+dc] not in (Cell.OBSTACLE.value, Cell.SNAKE.value, Cell.HEAD.value)
    ]

def best_neighbor_by_space(pos: Tuple[int,int], grid: np.ndarray) -> Optional[Tuple[int,int]]:
    """Pick the neighbor giving the most flood-fill space."""
    neighbors = get_neighbors(pos, grid)
    if not neighbors:
        return None
    scores = []
    for nb in neighbors:
        sim = grid.copy()
        sim[nb[0], nb[1]] = Cell.SNAKE.value
        scores.append((flood_fill(sim, nb), nb))
    scores.sort(reverse=True)
    return scores[0][1] if scores else None

def decide_move(
    grid: np.ndarray,
    snake: deque,
    food: Tuple[int,int],
    heuristic: Heuristic,
) -> Optional[MoveDecision]:
    """
    Core AI decision engine.
    Tries strategies in priority order and returns best move.
    """
    head = snake[0]
    tail = snake[-1]
    risk = analyze_risk(grid, snake, food, heuristic)
    rejected: List[Strategy] = []

    # ── Strategy 1: SAFE_PATH ──────────────────────────────────
    if risk.food_reachable and not risk.trapping_risk:
        path, explored, cost = astar(grid, head, food, heuristic)
        if path and len(path) > 1:
            next_pos = path[1]
            # Verify we won't dead-end after move
            sim = grid.copy()
            sim[next_pos[0], next_pos[1]] = Cell.HEAD.value
            if flood_fill(sim, next_pos) >= len(snake):
                return MoveDecision(
                    next_pos=next_pos,
                    strategy=Strategy.SAFE_PATH,
                    rejected=rejected,
                    reason="Direct path to food is safe. Post-move space is sufficient.",
                    path=path,
                    explored=explored,
                    path_length=cost,
                    heuristic=heuristic,
                    risk=risk,
                )
        rejected.append(Strategy.SAFE_PATH)

    else:
        rejected.append(Strategy.SAFE_PATH)

    # ── Strategy 2: DEAD_END_AVOIDANCE ─────────────────────────
    if risk.food_reachable:
        path, explored, cost = astar(grid, head, food, heuristic)
        if path and len(path) > 1:
            next_pos = path[1]
            sim = grid.copy()
            sim[next_pos[0], next_pos[1]] = Cell.HEAD.value
            space = flood_fill(sim, next_pos)
            if space >= max(5, len(snake) // 2):
                return MoveDecision(
                    next_pos=next_pos,
                    strategy=Strategy.DEAD_END_AVOIDANCE,
                    rejected=rejected,
                    reason="Path to food accepted with partial space check. Safe corridor detected.",
                    path=path,
                    explored=explored,
                    path_length=cost,
                    heuristic=heuristic,
                    risk=risk,
                )
        rejected.append(Strategy.DEAD_END_AVOIDANCE)
    else:
        rejected.append(Strategy.DEAD_END_AVOIDANCE)

    # ── Strategy 3: TAIL_FOLLOWING ─────────────────────────────
    # Temporarily treat tail as reachable target
    tail_path, tail_explored, tail_cost = astar(
        grid, head, tail, heuristic,
        blocked={tail}  # allow tail square
    )
    if tail_path and len(tail_path) > 1:
        next_pos = tail_path[1]
        return MoveDecision(
            next_pos=next_pos,
            strategy=Strategy.TAIL_FOLLOWING,
            rejected=rejected,
            reason="Food unreachable or risky. Following own tail to stay mobile.",
            path=tail_path,
            explored=tail_explored,
            path_length=tail_cost,
            heuristic=heuristic,
            risk=risk,
        )
    rejected.append(Strategy.TAIL_FOLLOWING)

    # ── Strategy 4: OPEN_SPACE_PRIORITY ────────────────────────
    best = best_neighbor_by_space(head, grid)
    if best:
        return MoveDecision(
            next_pos=best,
            strategy=Strategy.OPEN_SPACE_PRIORITY,
            rejected=rejected,
            reason="No safe path to food or tail. Moving to largest open area.",
            path=[head, best],
            explored=set(),
            path_length=1,
            heuristic=heuristic,
            risk=risk,
        )
    rejected.append(Strategy.OPEN_SPACE_PRIORITY)

    # ── Strategy 5: SURVIVAL_MODE ──────────────────────────────
    neighbors = get_neighbors(head, grid)
    if neighbors:
        # Pick neighbor that keeps most options open
        nb = max(neighbors, key=lambda n: flood_fill(grid, n))
        return MoveDecision(
            next_pos=nb,
            strategy=Strategy.SURVIVAL_MODE,
            rejected=rejected,
            reason="Critical survival mode. Picking last viable move.",
            path=[head, nb],
            explored=set(),
            path_length=1,
            heuristic=heuristic,
            risk=risk,
        )

    return None  # Game over

# ─────────────────────────────────────────────────────────────────
# GAME STATE
# ─────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    step: int
    head: Tuple[int,int]
    food: Tuple[int,int]
    strategy: str
    rejected: List[str]
    reason: str
    nodes_explored: int
    path_length: int
    heuristic: str
    danger_score: float
    survival_prob: float
    score: int

@dataclass
class GameState:
    grid_size: int
    obstacle_density: float
    heuristic: Heuristic

    # Mutable game fields
    grid: np.ndarray            = field(init=False)
    snake: deque                = field(init=False)
    food: Tuple[int,int]        = field(init=False)
    score: int                  = field(default=0, init=False)
    high_score: int             = field(default=0, init=False)
    step: int                   = field(default=0, init=False)
    alive: bool                 = field(default=True, init=False)
    start_time: float           = field(init=False)

    # Analytics
    strategy_counts: Dict[str,int]  = field(default_factory=dict, init=False)
    logs: List[StepLog]             = field(default_factory=list, init=False)
    path_costs: List[int]           = field(default_factory=list, init=False)
    nodes_per_step: List[int]       = field(default_factory=list, init=False)
    collision_avoided: int          = field(default=0, init=False)
    recalculations: int             = field(default=0, init=False)

    # Visualisation state
    current_path: List[Tuple[int,int]]     = field(default_factory=list, init=False)
    current_explored: Set[Tuple[int,int]]  = field(default_factory=set, init=False)

    def __post_init__(self):
        self.start_time = time.time()
        self._init_grid()
        for s in Strategy:
            self.strategy_counts[s.value] = 0

    def _init_grid(self):
        n = self.grid_size
        self.grid = np.zeros((n, n), dtype=np.int32)

        # Place obstacles
        obstacle_count = int(n * n * self.obstacle_density)
        placed = 0
        attempts = 0
        while placed < obstacle_count and attempts < obstacle_count * 10:
            r, c = random.randint(0, n-1), random.randint(0, n-1)
            if (r, c) not in {(n//2, n//2), (n//2+1, n//2), (n//2, n//2+1)}:
                self.grid[r, c] = Cell.OBSTACLE.value
                placed += 1
            attempts += 1

        # Place snake (3 cells, centre)
        mid = n // 2
        self.snake = deque([(mid, mid), (mid, mid+1), (mid, mid+2)])
        for r, c in self.snake:
            self.grid[r, c] = Cell.SNAKE.value
        self.grid[mid, mid] = Cell.HEAD.value

        # Spawn food
        self.food = self._spawn_food()
        self.grid[self.food[0], self.food[1]] = Cell.FOOD.value

    def _spawn_food(self) -> Tuple[int,int]:
        n = self.grid_size
        empties = list(zip(*np.where(self.grid == Cell.EMPTY.value)))
        if empties:
            chosen = random.choice(empties)
            return (int(chosen[0]), int(chosen[1]))
        # Fallback
        for _ in range(1000):
            r, c = random.randint(0, n-1), random.randint(0, n-1)
            if self.grid[r, c] == Cell.EMPTY.value:
                return (r, c)
        return (0, 0)

    def step_game(self) -> bool:
        """Advance one step. Returns True if alive."""
        if not self.alive:
            return False

        decision = decide_move(self.grid, self.snake, self.food, self.heuristic)
        if decision is None:
            self.alive = False
            return False

        self.step += 1
        self.recalculations += 1
        next_pos = decision.next_pos
        self.current_path = decision.path
        self.current_explored = decision.explored
        self.strategy_counts[decision.strategy.value] += 1
        self.nodes_per_step.append(len(decision.explored))
        if decision.path_length > 0:
            self.path_costs.append(decision.path_length)

        # Move snake
        old_tail = self.snake[-1]
        self.snake.appendleft(next_pos)

        ate = (next_pos == self.food)
        if ate:
            self.score += 10
            if self.score > self.high_score:
                self.high_score = self.score
            # Don't remove tail (snake grows)
        else:
            # Remove tail
            self.snake.pop()
            self.grid[old_tail[0], old_tail[1]] = Cell.EMPTY.value

        # Collision check
        r, c = next_pos
        if (not (0 <= r < self.grid_size and 0 <= c < self.grid_size)
                or self.grid[r, c] in (Cell.OBSTACLE.value, Cell.SNAKE.value)):
            self.alive = False
            return False

        # Update grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # Re-draw obstacles (re-scan original isn't stored, so track them)
        # Rebuild from snake + food; obstacles must be tracked separately
        # (they are already stamped in grid from __post_init__, but grid was reset above)
        # Fix: store obstacle positions
        for pos in self.snake:
            self.grid[pos[0], pos[1]] = Cell.SNAKE.value
        self.grid[next_pos[0], next_pos[1]] = Cell.HEAD.value

        if ate:
            self.food = self._spawn_food()
            self.grid[self.food[0], self.food[1]] = Cell.FOOD.value

        self.grid[self.food[0], self.food[1]] = Cell.FOOD.value

        # Collision avoided counter
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = next_pos[0]+dr, next_pos[1]+dc
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                    and self.grid[nr, nc] == Cell.SNAKE.value):
                self.collision_avoided += 1
                break

        # Log
        log = StepLog(
            step=self.step,
            head=next_pos,
            food=self.food,
            strategy=decision.strategy.value,
            rejected=[s.value for s in decision.rejected],
            reason=decision.reason,
            nodes_explored=len(decision.explored),
            path_length=decision.path_length,
            heuristic=decision.heuristic.value,
            danger_score=decision.risk.danger_score,
            survival_prob=decision.risk.survival_prob,
            score=self.score,
        )
        self.logs.append(log)
        if len(self.logs) > 200:
            self.logs.pop(0)

        return True

    def get_display_grid(self) -> np.ndarray:
        """Overlay explored nodes and path onto a copy of the grid."""
        display = self.grid.copy()
        for pos in self.current_explored:
            r, c = pos
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if display[r, c] == Cell.EMPTY.value:
                    display[r, c] = Cell.EXPLORED.value
        for pos in self.current_path:
            r, c = pos
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if display[r, c] == Cell.EMPTY.value:
                    display[r, c] = Cell.PATH.value
        return display

    @property
    def survival_time(self) -> float:
        return round(time.time() - self.start_time, 1)

    @property
    def avg_path_cost(self) -> float:
        return round(np.mean(self.path_costs), 2) if self.path_costs else 0.0

    @property
    def nodes_per_sec(self) -> float:
        t = self.survival_time
        total = sum(self.nodes_per_step)
        return round(total / t, 1) if t > 0 else 0.0

    def export_logs(self) -> str:
        rows = []
        for log in self.logs:
            rows.append({
                "Step": log.step,
                "Head": str(log.head),
                "Food": str(log.food),
                "Strategy": log.strategy,
                "Rejected": ", ".join(log.rejected),
                "Reason": log.reason,
                "Nodes Explored": log.nodes_explored,
                "Path Length": log.path_length,
                "Heuristic": log.heuristic,
                "Danger Score": log.danger_score,
                "Survival %": log.survival_prob,
                "Score": log.score,
            })
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

# ─────────────────────────────────────────────────────────────────
# GRID RENDERER  (HTML table approach for retro pixel look)
# ─────────────────────────────────────────────────────────────────

def render_grid_html(display_grid: np.ndarray, cell_px: int = 18) -> str:
    """Render grid as a compact HTML table with retro styling."""
    rows, cols = display_grid.shape
    color_map = {
        Cell.EMPTY.value:    COLORS[Cell.EMPTY],
        Cell.SNAKE.value:    COLORS[Cell.SNAKE],
        Cell.HEAD.value:     COLORS[Cell.HEAD],
        Cell.FOOD.value:     COLORS[Cell.FOOD],
        Cell.OBSTACLE.value: COLORS[Cell.OBSTACLE],
        Cell.PATH.value:     COLORS[Cell.PATH],
        Cell.EXPLORED.value: COLORS[Cell.EXPLORED],
    }
    # Build SVG for speed
    cell = cell_px
    total_w = cols * cell
    total_h = rows * cell

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'style="display:block;background:{COLORS["bg"]};border:2px solid {COLORS["neon_green"]}33;">'
    ]

    # Grid lines (subtle)
    for r in range(rows+1):
        y = r * cell
        svg_parts.append(
            f'<line x1="0" y1="{y}" x2="{total_w}" y2="{y}" '
            f'stroke="{COLORS["grid"]}" stroke-width="0.5"/>'
        )
    for c in range(cols+1):
        x = c * cell
        svg_parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{total_h}" '
            f'stroke="{COLORS["grid"]}" stroke-width="0.5"/>'
        )

    # Cells
    for r in range(rows):
        for c in range(cols):
            val = display_grid[r, c]
            color = color_map.get(val, COLORS[Cell.EMPTY])
            if val == Cell.EMPTY.value:
                continue  # background already correct
            x = c * cell
            y = r * cell

            if val == Cell.HEAD.value:
                # Draw head as special white square with inner dot
                svg_parts.append(
                    f'<rect x="{x+1}" y="{y+1}" width="{cell-2}" height="{cell-2}" '
                    f'rx="3" fill="{color}"/>'
                )
                cx_, cy_ = x + cell//2, y + cell//2
                svg_parts.append(
                    f'<circle cx="{cx_}" cy="{cy_}" r="{cell//5}" fill="{COLORS["bg"]}"/>'
                )
            elif val == Cell.FOOD.value:
                cx_ = x + cell//2
                cy_ = y + cell//2
                r_ = cell//2 - 2
                svg_parts.append(
                    f'<circle cx="{cx_}" cy="{cy_}" r="{r_}" fill="{color}"/>'
                )
                # Glow
                svg_parts.append(
                    f'<circle cx="{cx_}" cy="{cy_}" r="{r_+3}" fill="{color}" opacity="0.2"/>'
                )
            elif val == Cell.SNAKE.value:
                svg_parts.append(
                    f'<rect x="{x+1}" y="{y+1}" width="{cell-2}" height="{cell-2}" '
                    f'rx="2" fill="{color}"/>'
                )
            elif val == Cell.OBSTACLE.value:
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{color}"/>'
                    f'<line x1="{x}" y1="{y}" x2="{x+cell}" y2="{y+cell}" '
                    f'stroke="#663300" stroke-width="1"/>'
                )
            elif val == Cell.PATH.value:
                cx_ = x + cell//2
                cy_ = y + cell//2
                svg_parts.append(
                    f'<rect x="{x+3}" y="{y+3}" width="{cell-6}" height="{cell-6}" '
                    f'rx="1" fill="{color}" opacity="0.9"/>'
                )
            elif val == Cell.EXPLORED.value:
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{color}" opacity="0.6"/>'
                )

    svg_parts.append('</svg>')
    return ''.join(svg_parts)

# ─────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────

def init_session():
    """Initialize all session state variables."""
    if "game" not in st.session_state:
        st.session_state.game = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "speed" not in st.session_state:
        st.session_state.speed = 5
    if "grid_size" not in st.session_state:
        st.session_state.grid_size = 20
    if "obstacle_density" not in st.session_state:
        st.session_state.obstacle_density = 0.05
    if "heuristic" not in st.session_state:
        st.session_state.heuristic = Heuristic.MANHATTAN
    if "high_score" not in st.session_state:
        st.session_state.high_score = 0
    if "total_games" not in st.session_state:
        st.session_state.total_games = 0
    if "step_counter" not in st.session_state:
        st.session_state.step_counter = 0

def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    /* Dark retro base */
    html, body, [data-testid="stAppViewContainer"] {
        background: #050a05 !important;
        color: #00cc44 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    [data-testid="stSidebar"] {
        background: #020602 !important;
        border-right: 1px solid #00cc4422 !important;
    }

    [data-testid="stSidebar"] * {
        color: #00cc44 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0a1a0a !important;
        border: 1px solid #00cc4433 !important;
        border-radius: 6px !important;
        padding: 8px !important;
    }

    [data-testid="stMetricValue"] {
        color: #00ff66 !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 1.1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #007722 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    /* Buttons */
    .stButton > button {
        background: #001a00 !important;
        color: #00ff66 !important;
        border: 1px solid #00ff66 !important;
        border-radius: 4px !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        transition: all 0.2s !important;
        padding: 0.4rem 1rem !important;
    }
    .stButton > button:hover {
        background: #00ff66 !important;
        color: #001a00 !important;
        box-shadow: 0 0 16px #00ff6688 !important;
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: #00cc44 !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        color: #00ff66 !important;
        letter-spacing: 3px !important;
    }

    /* Tabs */
    [data-testid="stHorizontalBlock"] {}

    /* Select boxes */
    .stSelectbox > div > div {
        background: #0a1a0a !important;
        border: 1px solid #00cc4433 !important;
        color: #00cc44 !important;
    }

    /* Divider */
    hr {
        border-color: #00cc4422 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #050a05; }
    ::-webkit-scrollbar-thumb { background: #00cc44; border-radius: 2px; }

    /* Code / log blocks */
    .log-block {
        background: #050a05;
        border: 1px solid #00cc4433;
        border-left: 3px solid #00ff66;
        border-radius: 4px;
        padding: 10px 14px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: #00cc44;
        margin-bottom: 8px;
        line-height: 1.6;
    }
    .log-block .step-header {
        color: #ffdd00;
        font-size: 0.9rem;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        margin-bottom: 6px;
    }
    .log-block .strategy-name {
        color: #00ff66;
    }
    .log-block .rejected {
        color: #ff4400;
    }
    .log-block .metric-val {
        color: #ffffff;
    }
    .danger-low  { color: #00ff66 !important; }
    .danger-mid  { color: #ffdd00 !important; }
    .danger-high { color: #ff3300 !important; }

    /* Title area */
    .title-block {
        text-align: center;
        padding: 8px 0 4px 0;
    }
    .title-main {
        font-family: 'Orbitron', monospace;
        font-size: 1.6rem;
        font-weight: 900;
        color: #00ff66;
        letter-spacing: 6px;
        text-shadow: 0 0 20px #00ff6688;
    }
    .title-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #007722;
        letter-spacing: 4px;
        margin-top: 2px;
    }
    /* Legend */
    .legend-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.68rem;
        color: #007722;
        margin: 4px 0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .legend-dot {
        width: 10px; height: 10px;
        border-radius: 2px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)


def render_log_entry(log: StepLog) -> str:
    """Format a single log entry as styled HTML."""
    if log.danger_score < 30:
        danger_class = "danger-low"
    elif log.danger_score < 60:
        danger_class = "danger-mid"
    else:
        danger_class = "danger-high"

    rejected_str = ", ".join(log.rejected) if log.rejected else "None"
    return f"""
    <div class="log-block">
        <div class="step-header">▶ STEP {log.step:04d} &nbsp;|&nbsp; SCORE: {log.score}</div>
        <div>POS: <span class="metric-val">{log.head}</span> &nbsp;→&nbsp; FOOD: <span class="metric-val">{log.food}</span></div>
        <div>STRATEGY: <span class="strategy-name">{log.strategy}</span></div>
        <div>REJECTED: <span class="rejected">{rejected_str}</span></div>
        <div>REASON: {log.reason}</div>
        <div>NODES: <span class="metric-val">{log.nodes_explored}</span> &nbsp;|&nbsp; PATH: <span class="metric-val">{log.path_length}</span> &nbsp;|&nbsp; H: <span class="metric-val">{log.heuristic}</span></div>
        <div>DANGER: <span class="{danger_class}">{log.danger_score}%</span> &nbsp;|&nbsp; SURVIVAL: <span class="metric-val">{log.survival_prob}%</span></div>
    </div>
    """


def strategy_bar_chart(counts: Dict[str,int]) -> str:
    """Tiny inline SVG bar chart for strategy usage."""
    strategies = list(counts.keys())
    values = [counts[s] for s in strategies]
    total = max(sum(values), 1)
    bar_colors = {
        Strategy.SAFE_PATH.value:           "#00ff66",
        Strategy.TAIL_FOLLOWING.value:      "#0088ff",
        Strategy.LONGEST_SAFE_ROUTE.value:  "#aa44ff",
        Strategy.SURVIVAL_MODE.value:       "#ff3300",
        Strategy.OPEN_SPACE_PRIORITY.value: "#ffaa00",
        Strategy.DEAD_END_AVOIDANCE.value:  "#ffdd00",
    }

    w = 300
    bar_h = 18
    gap = 4
    label_w = 170
    svg_h = len(strategies) * (bar_h + gap) + 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{svg_h}" style="font-family:\'Share Tech Mono\',monospace;background:#0a1a0a;border-radius:6px;padding:4px;">']
    for i, (s, v) in enumerate(zip(strategies, values)):
        y = i * (bar_h + gap) + 5
        pct = v / total
        bar_w = int((w - label_w - 10) * pct)
        color = bar_colors.get(s, "#00cc44")
        short = s.replace("_", " ")
        parts.append(
            f'<text x="4" y="{y + bar_h - 4}" font-size="9" fill="#007722">{short}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h}" rx="2" fill="{color}" opacity="0.85"/>'
            f'<text x="{label_w + bar_w + 4}" y="{y + bar_h - 4}" font-size="9" fill="#aaa">{v}</text>'
        )
    parts.append('</svg>')
    return ''.join(parts)


def main():
    st.set_page_config(
        page_title="AI Snake",
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()
    init_session()

    # ── SIDEBAR CONTROLS ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0;">
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#00ff66;
                        letter-spacing:4px;text-shadow:0 0 10px #00ff6666;">
                🐍 AI SNAKE
            </div>
            <div style="font-size:0.6rem;color:#004422;letter-spacing:3px;margin-top:2px;">
                A* PATHFINDER ENGINE
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("##### ⚙ CONFIGURATION")
        grid_size = st.slider("Grid Size", 10, 30, st.session_state.grid_size, 2)
        speed = st.slider("Speed (steps/sec)", 1, 20, st.session_state.speed)
        obstacle_pct = st.slider("Obstacle Density %", 0, 20, int(st.session_state.obstacle_density*100))
        heuristic_choice = st.selectbox("Heuristic", ["Manhattan", "Euclidean"])

        st.markdown("---")
        st.markdown("##### 🎮 CONTROLS")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ START", use_container_width=True):
                st.session_state.heuristic = (
                    Heuristic.MANHATTAN if heuristic_choice == "Manhattan" else Heuristic.EUCLIDEAN
                )
                old_hs = st.session_state.high_score
                if st.session_state.game:
                    old_hs = max(old_hs, st.session_state.game.high_score)
                st.session_state.game = GameState(
                    grid_size=grid_size,
                    obstacle_density=obstacle_pct/100,
                    heuristic=st.session_state.heuristic,
                )
                st.session_state.game.high_score = old_hs
                st.session_state.grid_size = grid_size
                st.session_state.obstacle_density = obstacle_pct/100
                st.session_state.speed = speed
                st.session_state.running = True
                st.session_state.total_games += 1
                st.rerun()

        with col_b:
            if st.button("⏸ PAUSE", use_container_width=True):
                st.session_state.running = not st.session_state.running
                st.rerun()

        if st.button("↺ RESTART", use_container_width=True):
            old_hs = st.session_state.high_score
            if st.session_state.game:
                old_hs = max(old_hs, st.session_state.game.high_score)
            st.session_state.heuristic = (
                Heuristic.MANHATTAN if heuristic_choice == "Manhattan" else Heuristic.EUCLIDEAN
            )
            st.session_state.game = GameState(
                grid_size=grid_size,
                obstacle_density=obstacle_pct/100,
                heuristic=st.session_state.heuristic,
            )
            st.session_state.game.high_score = old_hs
            st.session_state.grid_size = grid_size
            st.session_state.running = True
            st.session_state.total_games += 1
            st.rerun()

        st.markdown("---")
        st.markdown("##### 🗺 LEGEND")
        st.markdown("""
        <div class="legend-row">
            <span class="legend-item"><span class="legend-dot" style="background:#ffffff;"></span>HEAD</span>
            <span class="legend-item"><span class="legend-dot" style="background:#00cc44;"></span>BODY</span>
            <span class="legend-item"><span class="legend-dot" style="background:#ffdd00;border-radius:50%;"></span>FOOD</span>
        </div>
        <div class="legend-row">
            <span class="legend-item"><span class="legend-dot" style="background:#005500;"></span>PATH</span>
            <span class="legend-item"><span class="legend-dot" style="background:#001a33;"></span>EXPLORED</span>
            <span class="legend-item"><span class="legend-dot" style="background:#442200;"></span>OBSTACLE</span>
        </div>
        """, unsafe_allow_html=True)

        # Download logs
        if st.session_state.game and st.session_state.game.logs:
            csv_data = st.session_state.game.export_logs()
            st.download_button(
                "📥 DOWNLOAD LOGS",
                data=csv_data,
                file_name="ai_snake_logs.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── MAIN AREA ────────────────────────────────────────────────
    # Title
    st.markdown("""
    <div class="title-block">
        <div class="title-main">AUTONOMOUS AI SNAKE</div>
        <div class="title-sub">A* PATHFINDING · EXPLAINABLE AI · REAL-TIME STRATEGY ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.game is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#007722;font-family:'Share Tech Mono',monospace;">
            <div style="font-size:3rem;margin-bottom:16px;">🐍</div>
            <div style="font-size:1.2rem;letter-spacing:4px;color:#00cc44;">PRESS START TO LAUNCH</div>
            <div style="font-size:0.75rem;margin-top:12px;color:#004422;letter-spacing:2px;">
                Configure settings in the sidebar, then click ▶ START
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    game: GameState = st.session_state.game

    # ── STEP GAME ─────────────────────────────────────────────────
    if st.session_state.running and game.alive:
        delay = 1.0 / max(1, st.session_state.speed)
        game.step_game()
        time.sleep(delay)
        st.session_state.step_counter += 1

        if not game.alive:
            st.session_state.running = False
            st.session_state.high_score = max(
                st.session_state.high_score, game.high_score
            )

    # ── LAYOUT: 3 columns ────────────────────────────────────────
    col_grid, col_log = st.columns([3, 2], gap="medium")

    with col_grid:
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SCORE", game.score)
        m2.metric("HIGH", game.high_score)
        m3.metric("STEP", game.step)
        m4.metric("LENGTH", len(game.snake))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("TIME", f"{game.survival_time}s")
        m6.metric("NODES/s", game.nodes_per_sec)
        m7.metric("AVG PATH", game.avg_path_cost)
        m8.metric("AVOIDED", game.collision_avoided)

        # Grid
        if not game.alive:
            st.markdown(
                f'<div style="text-align:center;font-family:\'Orbitron\',monospace;'
                f'color:#ff3300;font-size:1.4rem;letter-spacing:4px;padding:12px 0;'
                f'text-shadow:0 0 16px #ff330088;">⚠ GAME OVER — SCORE: {game.score}</div>',
                unsafe_allow_html=True,
            )

        cell_px = max(10, min(22, 440 // game.grid_size))
        display_grid = game.get_display_grid()
        svg_html = render_grid_html(display_grid, cell_px)
        st.markdown(
            f'<div style="display:flex;justify-content:center;margin:8px 0;">{svg_html}</div>',
            unsafe_allow_html=True,
        )

        # Strategy usage chart
        st.markdown("##### 📊 STRATEGY USAGE")
        chart_svg = strategy_bar_chart(game.strategy_counts)
        st.markdown(
            f'<div style="display:flex;justify-content:center;">{chart_svg}</div>',
            unsafe_allow_html=True,
        )

        # Survival analytics
        if len(game.logs) > 2:
            st.markdown("##### 📈 SURVIVAL ANALYTICS")
            recent = game.logs[-50:]
            df_plot = pd.DataFrame({
                "Step": [l.step for l in recent],
                "Survival %": [l.survival_prob for l in recent],
                "Danger %": [l.danger_score for l in recent],
            }).set_index("Step")
            st.line_chart(df_plot, use_container_width=True, height=150)

    with col_log:
        st.markdown("##### 🧠 AI DECISION LOG")

        status_color = "#00ff66" if game.alive else "#ff3300"
        status_text = "ALIVE" if game.alive else "DEAD"
        paused_text = "" if st.session_state.running else " | ⏸ PAUSED"
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;'
            f'color:{status_color};margin-bottom:8px;letter-spacing:2px;">'
            f'● STATUS: {status_text}{paused_text}</div>',
            unsafe_allow_html=True,
        )

        # Show last N logs, newest first
        log_html = ""
        for log in reversed(game.logs[-12:]):
            log_html += render_log_entry(log)
        st.markdown(
            f'<div style="max-height:520px;overflow-y:auto;">{log_html}</div>',
            unsafe_allow_html=True,
        )

        # Per-strategy breakdown table
        if any(v > 0 for v in game.strategy_counts.values()):
            st.markdown("##### 📋 STRATEGY BREAKDOWN")
            total_moves = max(game.step, 1)
            rows = []
            for s, count in game.strategy_counts.items():
                rows.append({
                    "Strategy": s,
                    "Uses": count,
                    "Pct": f"{100*count/total_moves:.1f}%"
                })
            df_strat = pd.DataFrame(rows)
            st.dataframe(
                df_strat,
                hide_index=True,
                use_container_width=True,
                height=220,
            )

    # ── AUTO-RERUN ────────────────────────────────────────────────
    if st.session_state.running and game.alive:
        st.rerun()
    elif not game.alive and st.session_state.running:
        # Auto-restart after a brief pause
        time.sleep(1.5)
        old_hs = max(st.session_state.high_score, game.high_score)
        st.session_state.game = GameState(
            grid_size=st.session_state.grid_size,
            obstacle_density=st.session_state.obstacle_density,
            heuristic=st.session_state.heuristic,
        )
        st.session_state.game.high_score = old_hs
        st.session_state.high_score = old_hs
        st.session_state.running = True
        st.session_state.total_games += 1
        st.rerun()


if __name__ == "__main__":
    main()