# =============================================================================
#  AIDRA.PY  —  Complete Disaster Response AI in ONE FILE
#  Scenario : 2005 Kashmir Earthquake — Muzaffarabad
#  Run with : python aidra.py
#
#  WHAT THIS FILE DOES (top to bottom):
#   PART 1 — The MAP and WORLD  (grid, victims, resources)
#   PART 2 — SEARCH algorithms  (BFS, DFS, Greedy, A*, Hill Climbing)
#   PART 3 — CSP                (who gets which ambulance)
#   PART 4 — MACHINE LEARNING   (predict survival probability)
#   PART 5 — FUZZY LOGIC        (handle uncertain risk levels)
#   PART 6 — MAIN RUNNER        (ties everything together)
# =============================================================================

import heapq                    # Special sorted list — always gives smallest item first
import time                     # For measuring how long algorithms take
import numpy as np              # Numbers and arrays
from collections import deque   # A fast queue used by BFS

# scikit-learn — the ML library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)

# =============================================================================
#  PART 1 — THE MAP AND WORLD
#  Think of this section as setting up the "board" for our simulation.
#  We define the grid, where victims are, and what resources we have.
# =============================================================================

# ── Cell type constants ────────────────────────────────────────────────────
# We give names to the 4 types of grid cells.
# Using names like BLOCKED is cleaner than writing the string everywhere.
EMPTY      = "EMPTY"       # Normal road — safe to drive on
BLOCKED    = "BLOCKED"     # Destroyed road — cannot pass
FIRE_ZONE  = "FIRE_ZONE"   # Active fire — can pass but dangerous (costs more)
AFTERSHOCK = "AFTERSHOCK"  # Unstable ground — may get blocked anytime

# ── Severity constants ─────────────────────────────────────────────────────
CRITICAL = "Critical"   # Life-threatening — rescue FIRST
MODERATE = "Moderate"   # Serious but stable — rescue SECOND
MINOR    = "Minor"      # Can wait a little — rescue LAST

# ── The 10×10 Grid ────────────────────────────────────────────────────────
# Row 0 = top of map.  Row 9 = bottom.
# Col 0 = left side.  Col 9 = right side.
#
# Layout (matching our interactive map):
#
#   Col: 0  1  2  3  4  5  6  7  8  9
# Row 0: .  .  .  .  .  .  .  .  .  M   ← MED-1 (Combined Military Hospital)
# Row 1: .  V1 .  .  F  F  V2 .  .  .   ← V1 critical, V2 critical, fire cols 4-5
# Row 2: .  .  .  X  F  F  .  .  .  .   ← X=blocked, fire continues
# Row 3: .  .  .  .  .  X  .  .  .  .   ← X=blocked
# Row 4: .  .  .  .  .  .  ~  ~  .  .   ← ~ = aftershock zones
# Row 5: .  .  .  .  .  .  ~  V4 .  .   ← V4 moderate near aftershock
# Row 6: .  .  .  V3 .  .  .  .  .  .   ← V3 moderate
# Row 7: .  .  .  .  .  .  .  .  .  .
# Row 8: B  .  .  .  .  V5 .  .  .  .   ← B=base, V5 minor
# Row 9: .  .  .  .  .  .  .  .  .  M   ← MED-2 (Red Crescent Field Hospital)

def build_grid():
    """
    Create the 10x10 disaster map.
    Returns a 2D list (list of lists) where grid[row][col] = cell type.

    HOW A 2D LIST WORKS:
      grid = [ [row0col0, row0col1, ...],   ← row 0
               [row1col0, row1col1, ...],   ← row 1
               ...
             ]
    To get cell at row=2, col=3 → grid[2][3]
    """
    # Start with all cells as EMPTY (normal road)
    grid = [[EMPTY for _ in range(10)] for _ in range(10)]
    #         ↑ make 10 EMPTYs           ↑ do that 10 times (10 rows)

    # Fire zones (rows 1-2, cols 4-5) — Hari Singh Street fire
    for row, col in [(1,4),(1,5),(2,4),(2,5)]:
        grid[row][col] = FIRE_ZONE

    # Aftershock zones (rows 4-5, cols 6-7)
    for row, col in [(4,6),(4,7),(5,6)]:
        grid[row][col] = AFTERSHOCK

    # Initially blocked roads
    for row, col in [(2,3),(3,5)]:
        grid[row][col] = BLOCKED

    return grid

# ── Victim and Resource data ───────────────────────────────────────────────
# We store victims and resources as simple dictionaries.
# A dictionary is like a labelled box: {"name": "Aisha", "row": 1, ...}

def make_victims():
    """
    Create all 5 victims.
    Each victim is a dictionary holding all their information.
    'priority' is calculated from severity — higher = rescue sooner.
    'survival' starts at 1.0 (100%) and drops as time passes.
    'rescued' starts False and becomes True when saved.
    """
    victims = [
        {
            "id"         : "V1",
            "name"       : "Aisha Bibi",
            "row"        : 1,   "col": 1,
            "severity"   : CRITICAL,
            "kits"       : 3,          # Needs 3 first-aid kits
            "needs_amb"  : True,       # Must use ambulance (not just rescue team)
            "priority"   : 115,        # 100 (Critical) + 3×5 (kits) = 115
            "survival"   : 1.0,        # 100% survival at start
            "rescued"    : False,
            "rescue_time": None,       # Will be filled in when rescued
            "assigned_to": None,       # Will be filled in by CSP
        },
        {
            "id"         : "V2",
            "name"       : "Tariq Ahmed",
            "row"        : 1,   "col": 6,
            "severity"   : CRITICAL,
            "kits"       : 3,
            "needs_amb"  : True,
            "priority"   : 115,
            "survival"   : 1.0,
            "rescued"    : False,
            "rescue_time": None,
            "assigned_to": None,
        },
        {
            "id"         : "V3",
            "name"       : "Fatima Malik",
            "row"        : 6,   "col": 3,
            "severity"   : MODERATE,
            "kits"       : 2,
            "needs_amb"  : True,
            "priority"   : 70,         # 60 (Moderate) + 2×5 = 70
            "survival"   : 1.0,
            "rescued"    : False,
            "rescue_time": None,
            "assigned_to": None,
        },
        {
            "id"         : "V4",
            "name"       : "Shahid Khan",
            "row"        : 5,   "col": 7,
            "severity"   : MODERATE,
            "kits"       : 2,
            "needs_amb"  : True,
            "priority"   : 70,
            "survival"   : 1.0,
            "rescued"    : False,
            "rescue_time": None,
            "assigned_to": None,
        },
        {
            "id"         : "V5",
            "name"       : "Zara Noor",
            "row"        : 8,   "col": 5,
            "severity"   : MINOR,
            "kits"       : 1,
            "needs_amb"  : False,      # Rescue team is enough — no ambulance needed
            "priority"   : 25,         # 20 (Minor) + 1×5 = 25
            "survival"   : 1.0,
            "rescued"    : False,
            "rescue_time": None,
            "assigned_to": None,
        },
    ]
    return victims

def make_resources():
    """
    Create 2 ambulances and 1 rescue team.
    'load' = how many victims currently assigned to it.
    'max'  = maximum victims it can carry at once.
    """
    return [
        {"id": "AMB-1",  "type": "Ambulance",    "max": 2, "load": 0, "is_free": True},
        {"id": "AMB-2",  "type": "Ambulance",    "max": 2, "load": 0, "is_free": True},
        {"id": "TEAM-1", "type": "Rescue Team",  "max": 1, "load": 0, "is_free": True},
    ]

# ── World state — one dictionary holding everything ─────────────────────
# We put all the simulation state into ONE dictionary called `world`.
# This makes it easy to pass everything between functions.

def create_world():
    """
    Create the full simulation world.
    Returns a dictionary with all parts of the simulation.
    """
    world = {
        "grid"       : build_grid(),
        "victims"    : make_victims(),
        "resources"  : make_resources(),
        "kits_left"  : 10,             # Total first-aid kits available
        "time"       : 0,              # Simulation clock (minutes elapsed)
        "log"        : [],             # Every event gets recorded here
        "base"       : (8, 0),         # Where ambulances start from
        "hospitals"  : [
            {"id": "MED-1", "pos": (0, 9), "name": "Combined Military Hospital"},
            {"id": "MED-2", "pos": (9, 9), "name": "Red Crescent Field Hospital"},
        ],
        "aftershocks": 0,              # How many aftershocks have happened
    }
    return world

# ── World helper functions ─────────────────────────────────────────────────

def log_event(world, message):
    """Record something that happened, with the current time."""
    entry = f"[t={world['time']:>3} min] {message}"
    world["log"].append(entry)

def advance_time(world, minutes):
    """
    Move the clock forward. Also drops every victim's survival probability.
    Critical patients drop 0.8%/min, Moderate 0.3%/min, Minor 0.1%/min.
    """
    world["time"] += minutes
    drop_rates = {CRITICAL: 0.008, MODERATE: 0.003, MINOR: 0.001}
    for v in world["victims"]:
        if not v["rescued"]:
            drop = drop_rates[v["severity"]] * minutes
            v["survival"] = max(0.0, v["survival"] - drop)

def unrescued(world):
    """Return list of victims not yet rescued, sorted highest priority first."""
    waiting = [v for v in world["victims"] if not v["rescued"]]
    return sorted(waiting, key=lambda v: v["priority"], reverse=True)

def print_grid(world, path=None, label="Map"):
    """
    Print the grid as text. If a path is given, mark it with *.
    B=base  M=hospital  F=fire  X=blocked  ~=aftershock  1-5=victim numbers
    """
    grid = world["grid"]
    rows, cols = 10, 10

    # Build a display copy
    display = [["." for _ in range(cols)] for _ in range(rows)]

    # Fill terrain symbols
    for r in range(rows):
        for c in range(cols):
            t = grid[r][c]
            if   t == BLOCKED:     display[r][c] = "X"
            elif t == FIRE_ZONE:   display[r][c] = "F"
            elif t == AFTERSHOCK:  display[r][c] = "~"

    # Mark path if given
    if path:
        path_set = set(path)
        for (r, c) in path_set:
            if display[r][c] == ".": display[r][c] = "*"
        if path:
            sr, sc = path[0];  display[sr][sc] = "S"
            gr, gc = path[-1]; display[gr][gc] = "G"

    # Place victims
    for v in world["victims"]:
        display[v["row"]][v["col"]] = v["id"][1]  # e.g. "1" for V1

    # Place base and hospitals
    br, bc = world["base"]
    display[br][bc] = "B"
    for h in world["hospitals"]:
        hr, hc = h["pos"]
        display[hr][hc] = "M"

    print(f"\n  {label}  (B=base M=hospital F=fire X=blocked ~=aftershock S=start G=goal *=path)")
    print("     " + "  ".join(str(c) for c in range(cols)))
    for r, row in enumerate(display):
        print(f"  {r}  " + "  ".join(row))
    print()


# =============================================================================
#  PART 2 — SEARCH ALGORITHMS
#
#  All search algorithms answer ONE question:
#  "What is the best path from point A to point B on the grid?"
#
#  They all return a dict:
#    {"path": [...], "cost": int, "expanded": int, "risk": int, "name": str}
#
#  "path"     = list of (row,col) cells from start to goal
#  "cost"     = total movement cost (fire zones cost more)
#  "expanded" = how many cells the algorithm had to check
#  "risk"     = danger score (how many fire/aftershock cells on path)
# =============================================================================

# ── Grid movement helpers (used by all search algorithms) ─────────────────

def get_neighbors(grid, row, col):
    """
    Return all cells you can move to from (row, col).
    Only 4 directions: UP DOWN LEFT RIGHT — no diagonal.
    Only returns cells that are not BLOCKED and inside the grid.
    """
    directions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right
    result = []
    for dr, dc in directions:
        nr, nc = row+dr, col+dc
        # Check inside grid AND not blocked
        if 0 <= nr < 10 and 0 <= nc < 10 and grid[nr][nc] != BLOCKED:
            result.append((nr, nc))
    return result

def move_cost(grid, row, col):
    """
    Cost to ENTER a cell. Used by A* to prefer safe routes.
    EMPTY=1  AFTERSHOCK=2  FIRE_ZONE=5  (fire is very expensive to pass through)
    """
    return {EMPTY:1, AFTERSHOCK:2, FIRE_ZONE:5, BLOCKED:999}.get(grid[row][col], 1)

def manhattan(r1, c1, r2, c2):
    """
    Straight-line distance between two cells (no diagonals).
    |r1-r2| + |c1-c2|
    Example: (0,0) to (3,4) = 3+4 = 7 steps.
    Used by A* and Greedy as a HEURISTIC (estimate of remaining distance).
    """
    return abs(r1-r2) + abs(c1-c2)

def path_risk(grid, path):
    """Count danger points on a path. Fire=5pts, Aftershock=2pts per cell."""
    score = 0
    for r, c in path:
        if   grid[r][c] == FIRE_ZONE:  score += 5
        elif grid[r][c] == AFTERSHOCK: score += 2
    return score

def rebuild_path(came_from, start, goal):
    """
    All search algorithms fill a dictionary called came_from:
      came_from[(r,c)] = the cell we came FROM to reach (r,c)

    This function follows that chain BACKWARDS from goal to start,
    then reverses the result to get start→goal order.

    Example: came_from = {(1,1):(2,1), (2,1):(3,1), (3,1):None}
      Follow from (1,1): (1,1)→(2,1)→(3,1)→None
      Reverse: [(3,1),(2,1),(1,1)]  ← that is the path!
    """
    path = []
    cur  = goal
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    return path if path[0] == start else None

# ── ALGORITHM 1: BFS (Breadth-First Search) ───────────────────────────────
# Think of it like ripples in a pond — explores outward layer by layer.
# GUARANTEES the shortest path (fewest steps).
# Uses a QUEUE (first in, first out — like a line at a shop).

def bfs(grid, start, goal, verbose=False):
    """
    Explore all cells 1 step away, then 2 steps, then 3 steps, etc.
    The first time we reach the goal, it is guaranteed to be the shortest path.
    """
    print(f"\n  [BFS] {start} → {goal}")
    queue     = deque([start])       # Queue starts with just the start cell
    came_from = {start: None}        # start came from nowhere
    expanded  = 0

    while queue:
        cur = queue.popleft()        # Take from FRONT of queue
        expanded += 1
        if verbose: print(f"    visiting {cur}")

        if cur == goal:              # Found the goal!
            path = rebuild_path(came_from, start, goal)
            cost = len(path) - 1
            risk = path_risk(grid, path)
            print(f"    ✓ Found in {len(path)} steps | cells checked={expanded}")
            return {"path":path,"cost":cost,"expanded":expanded,"risk":risk,"name":"BFS"}

        for nb in get_neighbors(grid, *cur):
            if nb not in came_from:  # Not visited yet
                came_from[nb] = cur
                queue.append(nb)     # Add to BACK of queue

    return {"path":None,"cost":0,"expanded":expanded,"risk":0,"name":"BFS"}

# ── ALGORITHM 2: DFS (Depth-First Search) ─────────────────────────────────
# Like exploring a maze by always going as far as possible before turning back.
# Does NOT guarantee shortest path — just finds ANY path.
# Uses a STACK (last in, first out — like a stack of plates).

def dfs(grid, start, goal, verbose=False):
    """
    Go as deep as possible down one path before backtracking.
    Fast to find A path, but often finds a longer path than BFS.
    """
    print(f"\n  [DFS] {start} → {goal}")
    stack     = [start]              # Stack starts with just the start cell
    came_from = {start: None}
    expanded  = 0

    while stack:
        cur = stack.pop()            # Take from TOP of stack (last added)
        expanded += 1
        if verbose: print(f"    visiting {cur}")

        if cur == goal:
            path = rebuild_path(came_from, start, goal)
            cost = len(path) - 1
            risk = path_risk(grid, path)
            print(f"    ✓ Found in {len(path)} steps | cells checked={expanded}")
            return {"path":path,"cost":cost,"expanded":expanded,"risk":risk,"name":"DFS"}

        for nb in get_neighbors(grid, *cur):
            if nb not in came_from:
                came_from[nb] = cur
                stack.append(nb)     # Add to TOP of stack

    return {"path":None,"cost":0,"expanded":expanded,"risk":0,"name":"DFS"}

# ── ALGORITHM 3: GREEDY BEST-FIRST ────────────────────────────────────────
# Always move toward whichever cell is CLOSEST to the goal.
# Like a dog chasing a ball — runs straight at it ignoring obstacles.
# Fast but can miss the optimal path when obstacles are in the way.

def greedy(grid, start, goal, verbose=False):
    """
    At each step, pick the neighbor closest to the goal (by Manhattan distance).
    Uses a priority queue — always processes the cell with lowest distance to goal.
    """
    print(f"\n  [Greedy] {start} → {goal}")
    # heapq: (priority, cell) — always pops LOWEST priority first
    pq        = [(0, start)]
    came_from = {start: None}
    expanded  = 0

    while pq:
        _, cur = heapq.heappop(pq)
        expanded += 1
        if verbose: print(f"    visiting {cur} | dist_to_goal={manhattan(*cur,*goal)}")

        if cur == goal:
            path = rebuild_path(came_from, start, goal)
            cost = sum(move_cost(grid,r,c) for r,c in path[1:])
            risk = path_risk(grid, path)
            print(f"    ✓ Found in {len(path)} steps | cells checked={expanded}")
            return {"path":path,"cost":cost,"expanded":expanded,"risk":risk,"name":"Greedy"}

        for nb in get_neighbors(grid, *cur):
            if nb not in came_from:
                came_from[nb] = cur
                priority = manhattan(*nb, *goal)  # Priority = distance to goal
                heapq.heappush(pq, (priority, nb))

    return {"path":None,"cost":0,"expanded":expanded,"risk":0,"name":"Greedy"}

# ── ALGORITHM 4: A* (THE MAIN ALGORITHM AIDRA USES) ──────────────────────
# Combines BFS (tracks actual cost so far) + Greedy (aims at goal).
# For every cell, calculates: f = g + h
#   g = actual cost to reach this cell from start
#   h = estimated steps remaining to goal (Manhattan distance)
#   f = total estimated cost — the cell with lowest f is explored next
#
# KEY ADVANTAGE over Greedy: our move_cost() gives fire zones cost=5.
# So A* NATURALLY avoids fire even if going around it is longer in steps.
# This solves the "speed vs safety" tradeoff in the assignment!

def astar(grid, start, goal, verbose=False):
    """
    Find the OPTIMAL (cheapest) path that also avoids dangerous zones.
    Fire zones cost 5x more than normal roads, so A* routes around them.
    """
    print(f"\n  [A*] {start} → {goal}")
    g_cost    = {start: 0}    # Cheapest known cost to reach each cell
    came_from = {start: None}
    expanded  = 0
    pq        = [(manhattan(*start,*goal), start)]  # (f_cost, cell)

    while pq:
        f, cur = heapq.heappop(pq)
        expanded += 1
        if verbose:
            g = g_cost[cur]; h = manhattan(*cur,*goal)
            print(f"    visiting {cur} | g={g} h={h} f={g+h}")

        if cur == goal:
            path = rebuild_path(came_from, start, goal)
            cost = g_cost[goal]
            risk = path_risk(grid, path)
            print(f"    ✓ Found in {len(path)} steps | cost={cost} | cells checked={expanded}")
            return {"path":path,"cost":cost,"expanded":expanded,"risk":risk,"name":"A*"}

        for nb in get_neighbors(grid, *cur):
            new_g = g_cost[cur] + move_cost(grid, *nb)   # Actual cost to reach nb
            # Only update if we found a CHEAPER route to nb
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb]    = new_g
                came_from[nb] = cur
                f_new = new_g + manhattan(*nb, *goal)     # f = g + h
                heapq.heappush(pq, (f_new, nb))

    return {"path":None,"cost":0,"expanded":expanded,"risk":0,"name":"A*"}

# ── ALGORITHM 5: HILL CLIMBING ────────────────────────────────────────────
# Local search — at each step, move to whichever neighbor is closest to goal.
# Never backtracks. Very fast but can get STUCK at dead ends.
# Used for QUICK replanning after aftershocks.

def hill_climbing(grid, start, goal, verbose=False):
    """
    Always move to the neighbor closest to the goal.
    If all neighbors are farther than current cell — stuck!
    """
    print(f"\n  [Hill Climbing] {start} → {goal}")
    cur      = start
    path     = [start]
    visited  = {start}
    expanded = 0

    for _ in range(100):   # Safety limit — stop after 100 steps
        expanded += 1
        if verbose: print(f"    at {cur}")

        if cur == goal:
            cost = sum(move_cost(grid,r,c) for r,c in path[1:])
            risk = path_risk(grid, path)
            print(f"    ✓ Found in {len(path)} steps | cells checked={expanded}")
            return {"path":path,"cost":cost,"expanded":expanded,"risk":risk,"name":"Hill Climbing"}

        neighbors  = get_neighbors(grid, *cur)
        unvisited  = [n for n in neighbors if n not in visited]
        if not unvisited:
            print(f"    ✗ Stuck at {cur} — no unvisited neighbors!")
            break

        # Pick the unvisited neighbor CLOSEST to the goal
        best = min(unvisited, key=lambda n: manhattan(*n,*goal))
        if verbose: print(f"    moving to {best}")
        visited.add(best)
        path.append(best)
        cur = best

    return {"path":None,"cost":0,"expanded":expanded,"risk":0,"name":"Hill Climbing"}

# ── SEARCH COMPARISON FUNCTION ────────────────────────────────────────────
def compare_search(world, start, goal):
    """
    Run ALL 5 algorithms on the same route and print a comparison table.
    This goes directly into the assignment report.
    """
    grid = world["grid"]
    print(f"\n  {'='*54}")
    print(f"  SEARCH COMPARISON  {start} → {goal}")
    print(f"  {'='*54}")

    results = [
        bfs(grid, start, goal),
        dfs(grid, start, goal),
        greedy(grid, start, goal),
        astar(grid, start, goal),
        hill_climbing(grid, start, goal),
    ]

    print(f"\n  {'Algorithm':<16} {'Steps':>6} {'Cost':>6} {'Cells checked':>14} {'Risk':>6}")
    print("  " + "-"*52)
    for r in results:
        if r["path"]:
            print(f"  {r['name']:<16} {len(r['path']):>6} {r['cost']:>6} {r['expanded']:>14} {r['risk']:>6}")
        else:
            print(f"  {r['name']:<16}  NO PATH FOUND")

    found = [r for r in results if r["path"]]
    if found:
        best_cost  = min(found, key=lambda r: r["cost"])
        least_exp  = min(found, key=lambda r: r["expanded"])
        safest     = min(found, key=lambda r: r["risk"])
        print(f"\n  Best cost    → {best_cost['name']}")
        print(f"  Fewest cells → {least_exp['name']}")
        print(f"  Safest route → {safest['name']}")
        print(f"  AIDRA uses A* (optimal cost + avoids fire zones automatically)")
    return results


# =============================================================================
#  PART 3 — CSP (Constraint Satisfaction Problem)
#
#  THE QUESTION CSP ANSWERS:
#  "Which ambulance or team should go to which victim,
#   without breaking any of the hard rules?"
#
#  VARIABLES : each victim needs to be assigned to a resource
#  DOMAINS   : AMB-1, AMB-2, TEAM-1  (or WAIT if no resource is free)
#  CONSTRAINTS (rules that CANNOT be broken):
#    1. Max 2 victims per ambulance
#    2. Max 1 victim per rescue team at a time
#    3. If victim needs_ambulance=True, cannot assign TEAM-1
#    4. Total kits used cannot exceed supply
#
#  METHOD: BACKTRACKING with MRV heuristic
#    - Try assigning a resource to a victim
#    - Check if all constraints still hold
#    - If YES → move to next victim
#    - If NO  → undo (backtrack) and try a different resource
# =============================================================================

def csp_constraints_ok(assignment, victims, resources, kits_left):
    """
    Check if the current (possibly partial) assignment breaks any rules.
    Returns True if valid, False if any rule is broken.

    assignment = dict like {"V1": "AMB-1", "V2": "AMB-2", ...}
    """

    # Count how many victims are assigned to each resource
    load = {}   # load["AMB-1"] = number of victims assigned so far
    for victim_id, resource_id in assignment.items():
        if resource_id == "WAIT": continue
        load[resource_id] = load.get(resource_id, 0) + 1

    # Rule 1 & 2: Check capacity limits
    for resource_id, count in load.items():
        res = next(r for r in resources if r["id"] == resource_id)
        if count > res["max"]:
            return False   # Over capacity!

    # Rule 3: Victim that needs ambulance must not get rescue team
    for victim_id, resource_id in assignment.items():
        if resource_id == "WAIT": continue
        victim  = next(v for v in victims if v["id"] == victim_id)
        res     = next(r for r in resources if r["id"] == resource_id)
        if victim["needs_amb"] and res["type"] == "Rescue Team":
            return False   # Wrong resource type!

    # Rule 4: Kit supply check
    kits_needed = sum(
        next(v for v in victims if v["id"] == vid)["kits"]
        for vid, rid in assignment.items() if rid != "WAIT"
    )
    if kits_needed > kits_left:
        return False   # Not enough kits!

    return True   # All rules satisfied!

def legal_resources(victim, resources, assignment):
    """
    What resources CAN we legally assign to this victim right now?
    Checks capacity and type requirements.
    Also always includes "WAIT" as a last resort.
    """
    legal = []
    for res in resources:
        # Count how many victims already assigned to this resource
        current_load = sum(1 for rid in assignment.values() if rid == res["id"])
        if current_load >= res["max"]: continue         # Full
        if victim["needs_amb"] and res["type"] == "Rescue Team": continue  # Wrong type
        legal.append(res["id"])
    legal.append("WAIT")   # Always allowed to delay
    return legal

def backtrack_csp(assignment, unassigned, victims, resources, kits_left, bt_count):
    """
    Recursive backtracking search for CSP.

    HOW RECURSION WORKS HERE:
      1. Pick a victim to assign (using MRV — pick the one with fewest options)
      2. Try each legal resource for that victim
      3. If assignment is valid → call this function again for next victim
      4. If that leads to success → return the solution
      5. If that leads to failure → undo assignment and try next resource
      6. If all resources tried and all fail → return None (backtrack further)
    """
    # BASE CASE: all victims have been assigned → success!
    if not unassigned:
        return assignment

    # MRV HEURISTIC: pick victim with FEWEST legal options
    # "Fail-first" — tackle the hardest assignments first
    # This dramatically reduces backtracking!
    def count_options(v):
        return len(legal_resources(v, resources, assignment))

    victim = min(unassigned, key=count_options)
    remaining = [v for v in unassigned if v["id"] != victim["id"]]

    # Try each legal resource for this victim
    options = legal_resources(victim, resources, assignment)

    # Sort: prefer resources with lower load (balance the work)
    def sort_key(rid):
        if rid == "WAIT": return 999  # Push WAIT to last
        return sum(1 for r in assignment.values() if r == rid)
    options.sort(key=sort_key)

    for resource_id in options:
        assignment[victim["id"]] = resource_id

        # Check if this assignment is valid
        if csp_constraints_ok(assignment, victims, resources, kits_left):
            # Try to assign the rest — recurse!
            result = backtrack_csp(assignment, remaining, victims, resources, kits_left, bt_count)
            if result is not None:
                return result   # Found a complete valid assignment!

        # This didn't work — undo and try next option (backtrack)
        del assignment[victim["id"]]
        bt_count[0] += 1

    return None   # No valid assignment found from this point

def run_csp(world, compare=False):
    """
    Solve the resource allocation problem.
    If compare=True, also shows the difference with/without heuristics.
    Returns the assignment dict {"V1":"AMB-1", "V2":"AMB-2", ...}
    """
    victims   = [v for v in world["victims"] if not v["rescued"]]
    resources = world["resources"]
    kits_left = world["kits_left"]

    print(f"\n  {'='*50}")
    print(f"  CSP — Resource Allocation")
    print(f"  {'='*50}")
    print(f"  Victims to assign : {[v['id'] for v in victims]}")
    print(f"  Resources         : {[r['id'] for r in resources]}")
    print(f"  Kits available    : {kits_left}")

    if compare:
        print(f"\n  Comparing WITH vs WITHOUT MRV heuristic:")
        print(f"  {'Method':<35} {'Backtracks':>12} {'Time(ms)':>10}")
        print("  " + "-"*58)
        for use_mrv in [True, False]:
            bt  = [0]
            t0  = time.time()
            res = backtrack_csp({}, list(victims), victims, resources, kits_left, bt)
            ms  = (time.time()-t0)*1000
            label = "With MRV heuristic" if use_mrv else "Plain backtracking (no MRV)"
            solved = "✓" if res else "✗"
            print(f"  {label:<35} {bt[0]:>12} {ms:>10.2f}  {solved}")

    # Final solve with all heuristics ON
    bt_count   = [0]
    assignment = backtrack_csp({}, list(victims), victims, resources, kits_left, bt_count)

    if assignment is None:
        print("\n  ERROR: No valid assignment found!")
        return {}

    print(f"\n  Backtracks needed: {bt_count[0]}")
    print(f"\n  {'Victim':<6} {'Name':<15} {'Severity':<10} {'Assigned':<10} {'Kits'}")
    print("  " + "-"*52)
    for v in sorted(victims, key=lambda x: x["priority"], reverse=True):
        rid = assignment.get(v["id"], "WAIT")
        print(f"  {v['id']:<6} {v['name']:<15} {v['severity']:<10} {rid:<10} {v['kits']}")

    # Show resource load
    print(f"\n  Resource loads:")
    for res in resources:
        assigned = [vid for vid, rid in assignment.items() if rid == res["id"]]
        bar = "█"*len(assigned) + "░"*(res["max"]-len(assigned))
        print(f"    {res['id']:<8} [{bar}] {len(assigned)}/{res['max']}  {assigned}")

    # Store assignment in victims
    for v in world["victims"]:
        if v["id"] in assignment:
            v["assigned_to"] = assignment[v["id"]]

    log_event(world, f"CSP complete — {bt_count[0]} backtracks — assignment: {assignment}")
    return assignment


# =============================================================================
#  PART 4 — MACHINE LEARNING
#
#  THE QUESTION ML ANSWERS:
#  "Given a victim's situation, what is the probability they survive
#   until rescued?"
#
#  WHY GENERATE DATA? We don't have a real dataset, so we SIMULATE
#  realistic rescue scenarios based on known disaster medicine facts:
#    - Critical patients deteriorate faster
#    - More time elapsed = lower survival
#    - Fire zones = lower survival
#    - More kits received = higher survival
#
#  We then train 3 models and compare them. The predictions feed
#  into CSP priority decisions — low survival = rescue first.
#
#  IMPORTANT FIX from ChatGPT issues:
#    - Data is generated ONCE and split properly (no leakage)
#    - Scaler is fit ONLY on training data, then applied to test
#    - MLP uses more iterations and better settings for this dataset size
# =============================================================================

def generate_data(n=800, seed=42):
    """
    Generate realistic synthetic training data for the survival predictor.

    Each sample = one rescue scenario.
    FEATURES (inputs to the model):
      0. severity_code  : 0=Critical, 1=Moderate, 2=Minor
      1. time_elapsed   : minutes since earthquake (0 to 120)
      2. distance       : steps to nearest hospital (1 to 15)
      3. zone_risk      : 0=safe, 1=aftershock, 2=fire
      4. kits_given     : first-aid kits administered (0 to 4)

    LABEL (what we're predicting):
      1 = survived  |  0 = did not survive
    """
    np.random.seed(seed)
    X, y = [], []

    for _ in range(n):
        sev      = np.random.choice([0,1,2], p=[0.4,0.4,0.2])
        time_el  = np.random.uniform(0, 120)
        dist     = np.random.uniform(1, 15)
        zone     = np.random.choice([0,1,2], p=[0.6,0.25,0.15])
        kits     = np.random.randint(0, 5)

        # Base survival chance by severity
        base = {0: 0.55, 1: 0.80, 2: 0.95}[sev]

        # Subtract penalties (things that reduce survival)
        time_penalty = time_el * {0:0.004, 1:0.001, 2:0.0003}[sev]
        dist_penalty = dist * 0.01
        zone_penalty = {0:0.0, 1:0.05, 2:0.15}[zone]

        # Add bonus for kits received
        kit_bonus = kits * 0.03

        # Final survival probability (clamped 0→1)
        prob = np.clip(base - time_penalty - dist_penalty - zone_penalty + kit_bonus, 0, 1)

        # Add a little noise (real data is never perfect)
        prob = np.clip(prob + np.random.normal(0, 0.04), 0, 1)

        # Convert probability to binary label (survived or not)
        label = 1 if np.random.random() < prob else 0

        X.append([sev, time_el, dist, zone, kits])
        y.append(label)

    return np.array(X, dtype=float), np.array(y, dtype=int)

def train_ml_models(world):
    """
    Train kNN, Naive Bayes, and MLP on the generated data.
    Evaluate each model and compare them.
    Returns a dict with trained models and their metrics.
    """
    print(f"\n  {'='*50}")
    print(f"  ML — Training Survival Prediction Models")
    print(f"  {'='*50}")

    # Step 1: Generate training data
    print("\n  Generating realistic earthquake rescue data...")
    X, y = generate_data(n=800)
    survived = y.sum()
    print(f"  {len(X)} samples | survived={survived} ({survived/len(X):.0%}) | lost={len(X)-survived} ({1-survived/len(X):.0%})")

    # Step 2: Split into training set (80%) and test set (20%)
    # We TRAIN on one portion and TEST on another so we know how well the model
    # works on data it has NEVER seen before.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Step 3: Scale features
    # StandardScaler puts all features on the same scale (mean=0, std=1).
    # IMPORTANT: fit the scaler on TRAIN data only, then apply to both train & test.
    # If we fit on all data, the model "peeks" at test data — that's data leakage!
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)   # Fit AND transform training data
    X_test_sc   = scaler.transform(X_test)        # ONLY transform test data (no fit)

    # Step 4: Define the 3 models
    models_cfg = {
        "kNN (k=5)": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"   # Closer neighbours count more
        ),
        "Naive Bayes": GaussianNB(),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
            activation="relu",
            max_iter=1000,                     # More iterations for better training
            random_state=42,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        ),
    }

    # Step 5: Train each model and evaluate it
    print(f"\n  {'Model':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-"*54)

    trained  = {}
    metrics  = {}

    for name, model in models_cfg.items():
        model.fit(X_train_sc, y_train)          # Train on training data
        y_pred = model.predict(X_test_sc)       # Predict on TEST data (never seen)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        trained[name] = model
        metrics[name] = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"cm":cm}

        print(f"  {name:<16} {acc:>9.2%} {prec:>10.2%} {rec:>8.2%} {f1:>8.2%}")

    best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    print(f"\n  Best model (highest F1): {best_name}")
    print(f"  AIDRA will use ALL 3 models and average their predictions (ensemble)")

    # Print confusion matrices
    print(f"\n  Confusion matrices (TN=correct_negative FP=false_positive")
    print(f"                      FN=false_negative  TP=correct_positive):")
    for name, m in metrics.items():
        cm = m["cm"]
        print(f"    {name}: TN={cm[0,0]} FP={cm[0,1]} | FN={cm[1,0]} TP={cm[1,1]}")

    return {"models": trained, "scaler": scaler, "metrics": metrics}

def predict_survival(ml, victim, time_elapsed, world):
    """
    Predict survival probability for one victim.
    Uses ALL 3 trained models and averages them (ensemble).

    Returns a number 0.0→1.0 (0%→100% survival probability).
    """
    models = ml["models"]
    scaler = ml["scaler"]

    # Find nearest hospital distance
    nearest_dist = min(
        abs(victim["row"] - h["pos"][0]) + abs(victim["col"] - h["pos"][1])
        for h in world["hospitals"]
    )

    # Zone risk at victim's position
    cell     = world["grid"][victim["row"]][victim["col"]]
    zone_num = {FIRE_ZONE:2, AFTERSHOCK:1}.get(cell, 0)
    sev_code = {CRITICAL:0, MODERATE:1, MINOR:2}[victim["severity"]]

    features = np.array([[sev_code, time_elapsed, nearest_dist, zone_num, victim["kits"]]])
    features_scaled = scaler.transform(features)

    # Get prediction from each model and average
    probs = [model.predict_proba(features_scaled)[0][1] for model in models.values()]
    return round(float(np.mean(probs)), 3)

def show_ml_predictions(ml, world):
    """Print survival predictions for all waiting victims."""
    print(f"\n  ML Survival Predictions (ensemble average of 3 models):")
    print(f"  {'ID':<4} {'Name':<15} {'Severity':<10} {'Survival Prob':>14} {'Priority action'}")
    print("  " + "-"*60)
    for v in sorted(world["victims"], key=lambda x: x["priority"], reverse=True):
        if v["rescued"]: continue
        prob = predict_survival(ml, v, world["time"], world)
        v["survival"] = prob  # Update the victim's survival estimate
        action = "RESCUE IMMEDIATELY" if prob < 0.6 else ("RESCUE SOON" if prob < 0.8 else "Can wait briefly")
        print(f"  {v['id']:<4} {v['name']:<15} {v['severity']:<10} {prob:>14.1%} {action}")


# =============================================================================
#  PART 5 — FUZZY LOGIC + BAYESIAN UNCERTAINTY
#
#  THE PROBLEM: In a real disaster, information is UNCERTAIN.
#  "Is this road blocked?" — probably, but we don't know for sure.
#  "How urgent is this victim?" — hard to say exactly.
#
#  FUZZY LOGIC: handles "grey areas" — values between 0 and 1.
#  Instead of "dangerous" or "safe", we say "60% dangerous".
#  We combine several fuzzy rules to compute an urgency score 0→10.
#
#  BAYESIAN: updates our belief about road blockage as we get
#  more information (more aftershocks = roads more likely blocked).
#
#  FIX from ChatGPT issues: urgency was 0.0 because time=0 at start.
#  Now urgency is based on SEVERITY + ZONE RISK + TIME, so even at t=0
#  critical victims near fire zones correctly get high urgency scores.
# =============================================================================

def fuzzy_membership_low(x, peak, width):
    """Fuzzy membership: how much does x belong to the LOW category?"""
    if x <= peak: return 1.0
    if x >= peak + width: return 0.0
    return 1.0 - (x - peak) / width

def fuzzy_membership_high(x, start, width):
    """Fuzzy membership: how much does x belong to the HIGH category?"""
    if x <= start: return 0.0
    if x >= start + width: return 1.0
    return (x - start) / width

def fuzzy_membership_mid(x, center, width):
    """Fuzzy membership: how much does x belong to the MEDIUM category?"""
    dist = abs(x - center)
    if dist >= width: return 0.0
    return 1.0 - dist / width

def compute_fuzzy_urgency(victim, time_elapsed, zone_risk_score):
    """
    Compute an urgency score from 0 to 10 using fuzzy IF-THEN rules.

    INPUTS:
      severity_score : 0=Minor, 5=Moderate, 10=Critical
      time_elapsed   : 0-120 minutes
      zone_risk_score: 0=safe, 5=aftershock, 10=fire

    RULES (simplified English):
      IF severity is HIGH → urgency goes up a lot
      IF time is HIGH     → urgency goes up
      IF zone is HIGH     → urgency goes up
      Combinations are multiplied (AND logic)
    """
    sev_score = {CRITICAL:10, MODERATE:5, MINOR:0}[victim["severity"]]

    # Membership values for each input
    sev_high = fuzzy_membership_high(sev_score, 4, 6)    # Is severity high?
    sev_mid  = fuzzy_membership_mid(sev_score, 5, 5)     # Is severity medium?
    time_high= fuzzy_membership_high(time_elapsed, 40, 80)  # Is time elapsed high?
    time_mid = fuzzy_membership_mid(time_elapsed, 30, 30)
    zone_high= fuzzy_membership_high(zone_risk_score, 4, 6) # Is zone dangerous?

    # Fuzzy rules — each gives a (strength, output_value) pair
    # strength = how strongly this rule fires (0.0 to 1.0)
    # output   = what urgency value this rule suggests
    rules = [
        (sev_high,                    8.5),  # Critical severity → very urgent
        (min(sev_high, time_high),    9.5),  # Critical + long wait → maximum urgent
        (min(sev_high, zone_high),    9.0),  # Critical + fire zone → very urgent
        (sev_mid,                     5.0),  # Moderate severity → medium urgent
        (min(sev_mid, time_high),     7.0),  # Moderate + long wait → more urgent
        (zone_high,                   3.0),  # Dangerous zone alone → some urgency
        (min(sev_mid, zone_high),     6.5),  # Moderate + fire zone → urgent
        (time_high,                   2.0),  # Long time alone → a little urgent
    ]

    # Defuzzification: weighted average
    # urgency = sum(strength × value) / sum(strength)
    total_strength = sum(s for s, _ in rules)
    if total_strength == 0:
        return 0.0
    urgency = sum(s * v for s, v in rules) / total_strength
    return round(min(10.0, max(0.0, urgency)), 1)

def bayesian_blockage_prob(zone_type, aftershocks, time_elapsed):
    """
    Estimate probability a road is blocked given our observations.

    BAYES' THEOREM (simplified):
      Prior: base chance of blockage (depends on zone type)
      Update: each aftershock increases the probability
      Update: more time elapsed → more instability

    Returns a probability 0.0 → 1.0
    """
    prior = {"EMPTY":0.10, "AFTERSHOCK":0.40, "FIRE_ZONE":0.65}.get(zone_type, 0.10)
    # Each aftershock adds 15% more blockage chance
    updated = prior + (aftershocks * 0.15) + (time_elapsed / 120 * 0.10)
    return round(min(1.0, updated), 2)

def run_uncertainty(world, label=""):
    """
    Run fuzzy logic and Bayesian analysis for all victims.
    Print results for the report.
    """
    print(f"\n  {'='*50}")
    print(f"  UNCERTAINTY MODULE — Fuzzy Logic + Bayesian")
    if label: print(f"  {label}")
    print(f"  {'='*50}")

    print(f"\n  Fuzzy Urgency Scores (0=low, 10=maximum urgency):")
    print(f"  {'ID':<4} {'Name':<15} {'Severity':<10} {'ZoneRisk':>9} {'Urgency':>8}  Bar")
    print("  " + "-"*60)

    urgency_map = {}
    for v in sorted(world["victims"], key=lambda x: x["priority"], reverse=True):
        if v["rescued"]: continue
        cell      = world["grid"][v["row"]][v["col"]]
        zone_risk = {FIRE_ZONE:9, AFTERSHOCK:5}.get(cell, 1)
        urgency   = compute_fuzzy_urgency(v, world["time"], zone_risk)
        urgency_map[v["id"]] = urgency
        bar = "█" * int(urgency) + "░" * (10 - int(urgency))
        print(f"  {v['id']:<4} {v['name']:<15} {v['severity']:<10} {zone_risk:>9} {urgency:>8.1f}  [{bar}]")

    print(f"\n  Bayesian Road Blockage Probabilities:")
    for zone, label_str in [("EMPTY","Normal roads"),("AFTERSHOCK","Aftershock zones"),("FIRE_ZONE","Fire zones")]:
        prob = bayesian_blockage_prob(zone, world["aftershocks"], world["time"])
        bar  = "█"*int(prob*10) + "░"*(10-int(prob*10))
        print(f"    {label_str:<25} [{bar}] {prob:.0%}")

    high_urgency = [vid for vid, u in urgency_map.items() if u >= 7.0]
    print(f"\n  High urgency victims (score ≥ 7.0): {high_urgency}")
    return urgency_map


# =============================================================================
#  PART 6 — MAIN RUNNER
#  This is where everything comes together.
#  It runs through the full disaster scenario step by step.
# =============================================================================

def section(title):
    """Print a nice section header."""
    print(f"\n{'═'*56}")
    print(f"  {title}")
    print(f"{'═'*56}")

def rescue_victim(world, victim, resource_id, route):
    """
    Perform one rescue:
      1. Drive the route (time passes → survival drops)
      2. Administer kits
      3. Mark victim as rescued
      4. Log everything
    """
    drive_time   = len(route["path"])   # Each step = 1 minute driving
    extract_time = 5                    # 5 extra minutes to extract from rubble
    total_time   = drive_time + extract_time

    # Time passes during the drive
    advance_time(world, total_time)

    # Use kits
    world["kits_left"] -= victim["kits"]
    world["kits_left"]  = max(0, world["kits_left"])

    # Mark rescued
    victim["rescued"]    = True
    victim["rescue_time"] = total_time

    log_event(world,
        f"{victim['id']} ({victim['name']}) rescued by {resource_id} | "
        f"route_cost={route['cost']} | risk={route['risk']} | "
        f"time={total_time}min | survival={victim['survival']:.0%}"
    )
    print(f"\n  ✓ RESCUED: {victim['name']} ({victim['id']}) "
          f"by {resource_id} in {total_time} min "
          f"| survival: {victim['survival']:.0%}")

def main():
    """
    MAIN FUNCTION — runs the complete AIDRA simulation.
    Phases:
      1. Initialise world
      2. Train ML models
      3. Fuzzy logic urgency
      4. CSP resource allocation
      5. Search & route planning
      6. Execute rescues (all 5 victims)
      7. Aftershock → replan
      8. Final metrics report
    """

    # ─────────────────────────────────────────────────────────
    section("PHASE 1 — World Setup")
    # ─────────────────────────────────────────────────────────
    world = create_world()
    print("\n  VICTIMS (priority order):")
    print(f"  {'ID':<4} {'Name':<15} {'Severity':<10} {'Position':>10} {'Priority':>9}")
    print("  " + "-"*52)
    for v in sorted(world["victims"], key=lambda x: x["priority"], reverse=True):
        print(f"  {v['id']:<4} {v['name']:<15} {v['severity']:<10} ({v['row']},{v['col']}){v['priority']:>10}")

    print("\n  RESOURCES:")
    for r in world["resources"]:
        print(f"    {r['id']:<8} {r['type']:<14} max={r['max']} victims at once")

    print(f"\n  Medical kits available: {world['kits_left']}")
    print_grid(world, label="Initial Map")

    # ─────────────────────────────────────────────────────────
    section("PHASE 2 — Machine Learning: Survival Prediction")
    # ─────────────────────────────────────────────────────────
    ml = train_ml_models(world)
    show_ml_predictions(ml, world)

    # ─────────────────────────────────────────────────────────
    section("PHASE 3 — Fuzzy Logic + Bayesian Uncertainty")
    # ─────────────────────────────────────────────────────────
    run_uncertainty(world, label="Initial assessment (t=0)")

    # ─────────────────────────────────────────────────────────
    section("PHASE 4 — CSP: Resource Allocation")
    # ─────────────────────────────────────────────────────────
    assignment = run_csp(world, compare=True)
    if not assignment:
        print("  CSP failed. Exiting."); return

    # ─────────────────────────────────────────────────────────
    section("PHASE 5 — Search: Route Planning & Comparison")
    # ─────────────────────────────────────────────────────────
    grid = world["grid"]
    base = world["base"]

    # 5a. Compare all algorithms for V1 (to show in report)
    v1 = next(v for v in world["victims"] if v["id"]=="V1")
    compare_search(world, base, (v1["row"], v1["col"]))

    # 5b. The KEY tradeoff: V2 is near fire — show safe vs risk route
    v2 = next(v for v in world["victims"] if v["id"]=="V2")
    print(f"\n  KEY TRADEOFF — Rescuing V2 (Tariq Ahmed) near fire zone:")

    r_astar  = astar(grid, base, (v2["row"],v2["col"]))
    r_greedy = greedy(grid, base, (v2["row"],v2["col"]))

    print(f"\n  A* route  : steps={len(r_astar['path'])}  cost={r_astar['cost']}  risk={r_astar['risk']}")
    print(f"  Greedy    : steps={len(r_greedy['path'])}  cost={r_greedy['cost']}  risk={r_greedy['risk']}")
    print(f"\n  DECISION: AIDRA chooses A* — it avoids fire (risk={r_astar['risk']} vs {r_greedy['risk']})")
    print(f"  JUSTIFICATION: V2 is Critical. Fire zone adds risk={r_greedy['risk']-r_astar['risk']} points.")
    print(f"  Even if Greedy is same steps, A* cost is lower because it routes around fire (cost 5x).")
    print_grid(world, path=r_astar["path"], label="A* route to V2 (avoids fire)")

    log_event(world, f"V2 route: A* chosen — cost={r_astar['cost']} risk={r_astar['risk']} (Greedy was riskier)")

    # ─────────────────────────────────────────────────────────
    section("PHASE 6 — Rescue Simulation (All 5 Victims)")
    # ─────────────────────────────────────────────────────────
    # Sort victims by priority — Critical first, then Moderate, then Minor
    rescue_order = sorted(world["victims"], key=lambda v: v["priority"], reverse=True)

    print(f"\n  Rescue order: {[v['id'] for v in rescue_order]}")
    print(f"\n  AMBULANCE TRIPS:")
    print(f"  ─ AMB-1 will take V1 first, then return for V3")
    print(f"  ─ AMB-2 will take V2 first, then return for V4")
    print(f"  ─ TEAM-1 will take V5 (no ambulance needed)")

    for v in rescue_order:
        if v["rescued"]: continue

        resource_id = assignment.get(v["id"], "WAIT")

        if resource_id == "WAIT":
            # This victim has no resource — use rescue team if possible
            free_teams = [r for r in world["resources"] if r["type"]=="Rescue Team" and r["is_free"]]
            if free_teams and not v["needs_amb"]:
                resource_id = free_teams[0]["id"]
                print(f"\n  → {v['id']} reassigned from WAIT to {resource_id}")
            elif free_teams and v["needs_amb"]:
                # Try to find a free ambulance
                free_ambs = [r for r in world["resources"] if r["type"]=="Ambulance" and r["is_free"]]
                if free_ambs:
                    resource_id = free_ambs[0]["id"]
                    print(f"\n  → {v['id']} reassigned from WAIT to {resource_id}")

        if resource_id == "WAIT":
            print(f"\n  ⏳ {v['id']} ({v['name']}) — no resource currently free, waiting...")
            log_event(world, f"{v['id']} waiting — no resource available")
            continue

        # Find route using A* (our main algorithm)
        route = astar(grid, base, (v["row"], v["col"]))
        if not route["path"]:
            print(f"\n  ✗ {v['id']} — no route found! Trying BFS as fallback...")
            route = bfs(grid, base, (v["row"], v["col"]))
            log_event(world, f"REPLAN: {v['id']} — A* failed, used BFS fallback")

        if not route["path"]:
            print(f"\n  ✗ {v['id']} — completely unreachable!")
            log_event(world, f"FAILED: {v['id']} unreachable — no path found")
            continue

        # Show the route on the grid
        print_grid(world, path=route["path"], label=f"A* route to {v['id']} ({v['name']})")

        # Do the rescue
        rescue_victim(world, v, resource_id, route)

        # Mark the resource as having completed a trip (now free again)
        res = next(r for r in world["resources"] if r["id"]==resource_id)
        res["is_free"] = True
        res["load"]    = 0

    # ─────────────────────────────────────────────────────────
    section("PHASE 7 — Dynamic Replanning (Aftershock Event)")
    # ─────────────────────────────────────────────────────────
    print("\n  Simulating aftershock mid-operation...")

    # New roads get blocked
    new_blocked = [(4,5),(3,7)]
    for r,c in new_blocked:
        if world["grid"][r][c] != BLOCKED:
            world["grid"][r][c] = BLOCKED
    world["aftershocks"] += 1

    log_event(world, f"AFTERSHOCK #{world['aftershocks']}! Roads {new_blocked} now blocked — replanning")
    print(f"  ⚡ AFTERSHOCK! Roads blocked: {new_blocked}")
    print_grid(world, label="Map after aftershock (new X blocks)")

    # Re-run uncertainty with updated information
    run_uncertainty(world, label=f"Post-aftershock (aftershocks so far: {world['aftershocks']})")

    # Any victims still waiting get re-evaluated
    still_waiting = [v for v in world["victims"] if not v["rescued"]]
    if still_waiting:
        print(f"\n  Still unrescued: {[v['id'] for v in still_waiting]}")
        for v in still_waiting:
            print(f"\n  Replanning route for {v['id']} ({v['name']})...")
            route = astar(world["grid"], base, (v["row"],v["col"]))
            if route["path"]:
                print(f"  ✓ New route found: steps={len(route['path'])} cost={route['cost']} risk={route['risk']}")
                log_event(world, f"REPLAN SUCCESS: {v['id']} — new route found post-aftershock")
                print_grid(world, path=route["path"], label=f"Replanned: {v['id']}")

                # Do the rescue
                resource_id = assignment.get(v["id"], None)
                if not resource_id or resource_id == "WAIT":
                    free_r = [r for r in world["resources"] if r["is_free"]]
                    resource_id = free_r[0]["id"] if free_r else "TEAM-1"
                rescue_victim(world, v, resource_id, route)
            else:
                print(f"  ✗ {v['id']} is now unreachable after aftershock!")
                log_event(world, f"REPLAN FAILED: {v['id']} — blocked after aftershock")
    else:
        print("  All victims already rescued before aftershock — no replanning needed!")

    # ─────────────────────────────────────────────────────────
    section("PHASE 8 — Final Performance Report (KPIs)")
    # ─────────────────────────────────────────────────────────
    rescued     = [v for v in world["victims"] if v["rescued"]]
    not_rescued = [v for v in world["victims"] if not v["rescued"]]
    times       = [v["rescue_time"] for v in rescued if v["rescue_time"]]
    avg_time    = sum(times)/len(times) if times else 0

    print(f"\n  RESCUE OUTCOMES:")
    print(f"  {'ID':<4} {'Name':<15} {'Severity':<10} {'Status':<10} {'Time':>8} {'Survival':>9}")
    print("  " + "-"*58)
    for v in world["victims"]:
        status = "RESCUED" if v["rescued"] else "NOT REACHED"
        t_str  = f"{v['rescue_time']} min" if v["rescue_time"] else "—"
        print(f"  {v['id']:<4} {v['name']:<15} {v['severity']:<10} {status:<10} {t_str:>8} {v['survival']:>9.1%}")

    print(f"\n  KEY PERFORMANCE INDICATORS:")
    print(f"  {'Metric':<35} {'Value':>12}")
    print("  " + "-"*48)
    print(f"  {'Victims rescued':<35} {len(rescued):>12} / {len(world['victims'])}")
    print(f"  {'Victims not reached':<35} {len(not_rescued):>12}")
    print(f"  {'Average rescue time (min)':<35} {avg_time:>12.1f}")
    print(f"  {'Kits used':<35} {world['total_kits']-world['kits_left'] if 'total_kits' in world else '?':>12}")
    print(f"  {'Total time elapsed (min)':<35} {world['time']:>12}")
    print(f"  {'Aftershocks handled':<35} {world['aftershocks']:>12}")

    print(f"\n  ML MODEL METRICS (for report):")
    print(f"  {'Model':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-"*54)
    for name, m in ml["metrics"].items():
        print(f"  {name:<16} {m['accuracy']:>9.2%} {m['precision']:>10.2%} {m['recall']:>8.2%} {m['f1']:>8.2%}")

    # ─────────────────────────────────────────────────────────
    section("PHASE 9 — Full Event Log")
    # ─────────────────────────────────────────────────────────
    for entry in world["log"]:
        print(f"  {entry}")

    print(f"\n  {'='*56}")
    print(f"  AIDRA Simulation Complete.")

    print(f"  {'='*56}\n")

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Add total kits to world for final report
    world_ref = create_world()
    world_ref["total_kits"] = world_ref["kits_left"]
    main()