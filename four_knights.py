import heapq
import time
from collections import deque
from typing import Tuple, List

# ------------------------------------------------------------------------------
# 1) Knight-move offsets on a 3×3 board
# ------------------------------------------------------------------------------
# A knight moves two squares in one direction (horizontal/vertical)
# then one square perpendicular. On a 3×3 grid, some “jumps” go off the board.
# KNIGHT_DELTAS lists all eight relative (row, col) moves a knight can make.
KNIGHT_DELTAS = [
    (-2, -1), (-2, +1),
    (-1, -2), (-1, +2),
    (+1, -2), (+1, +2),
    (+2, -1), (+2, +1)
]

def get_knight_moves(idx: int) -> List[int]:
    """
    Given a board index (0..8) on a 3×3 board, return all valid
    destination indices (0..8) reachable by a knight from idx.
    - We first convert idx → (row, col) via divmod(idx, 3).
    - For each of the eight deltas, check if (new_row, new_col) is inside 0..2.
    - If valid, convert back to a single index = new_row*3 + new_col.
    """
    row, col = divmod(idx, 3)
    moves = []
    for dr, dc in KNIGHT_DELTAS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            moves.append(nr * 3 + nc)
    return moves

def expand(state: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    """
    Generate all successor states by moving any one knight exactly once.
    - 'state' is a length-9 tuple of 'L' (light), 'D' (dark), or '.' (empty).
    - For each idx where state[idx] is 'L' or 'D', try all knight jumps:
      * If the target square is '.', swap piece→dest, idx→'.'.
      * Append that new board (as a tuple) to successors.
    - Return the list of all valid successor tuples.
    """
    successors = []
    for idx, piece in enumerate(state):
        if piece in ('L', 'D'):
            # This square has a knight; attempt each of its possible moves
            for dest in get_knight_moves(idx):
                # Only move if destination is empty
                if state[dest] == '.':
                    new_board = list(state)
                    new_board[dest] = piece   # place knight at dest
                    new_board[idx] = '.'      # vacate old square
                    successors.append(tuple(new_board))
    return successors

# ------------------------------------------------------------------------------
# 2) Precompute “knight-distance” between every pair of squares (0..8)
# ------------------------------------------------------------------------------
def precompute_knight_distances() -> List[List[int]]:
    """
    Build a 9×9 matrix dist, where dist[i][j] = minimum number of knight moves
    on a 3×3 board to go from square i → square j.
    - For each start in 0..8, run a BFS:
      * Initialize queue = [(start, 0)], visited = {start}.
      * Pop (curr, d); set dist[start][curr] = d.
      * For each neighbor m in get_knight_moves(curr), if unseen, enqueue (m, d+1).
    - Return the filled 9×9 dist matrix.
    """
    N = 9
    dist = [[float('inf')] * N for _ in range(N)]
    for start in range(N):
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            curr, d = queue.popleft()
            dist[start][curr] = d
            for m in get_knight_moves(curr):
                if m not in visited:
                    visited.add(m)
                    queue.append((m, d + 1))
    return dist

# Build the lookup once at module load.
DIST_LOOKUP = precompute_knight_distances()

def heuristic(state: Tuple[str, ...]) -> int:
    """
    Admissible heuristic for A*:
    - In the goal, dark knights must occupy corners {0, 2}; light knights must occupy {6, 8}.
    - For each dark knight not already in {0,2}, add the minimum knight-distance to {0,2}.
    - For each light knight not in {6,8}, add distance to {6,8}.
    - Sum all four distances, then divide by 2 (integer division). This division keeps
      the heuristic ≤ actual cost because, in the worst case, two knights might move in parallel.
    """
    dark_corners = [0, 2]
    light_corners = [6, 8]
    total = 0
    for idx, piece in enumerate(state):
        if piece == 'D' and idx not in dark_corners:
            # distance from this dark knight to nearest dark corner
            total += min(DIST_LOOKUP[idx][dc] for dc in dark_corners)
        elif piece == 'L' and idx not in light_corners:
            total += min(DIST_LOOKUP[idx][lc] for lc in light_corners)
    return total // 2

# ------------------------------------------------------------------------------
# 3) Node class for the priority queue (used by B&B and A*)
# ------------------------------------------------------------------------------
class Node:
    __slots__ = ('state', 'g', 'h', 'f', 'parent')

    def __init__(self, state: Tuple[str, ...], g: int, h: int, parent: 'Node' = None):
        # state: the 3×3 board as a length-9 tuple
        # g: cost so far (# moves from root)
        # h: heuristic estimate to goal
        # f: total priority = g + h
        # parent: pointer to the parent Node (for path reconstruction)
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other: 'Node') -> bool:
        """
        Compare nodes by f. If f ties, compare by smaller h. This ensures
        consistent tie-breaking in heapq.
        """
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

# ------------------------------------------------------------------------------
# 4) Branch & Bound (Uniform-Cost Search) Implementation
# ------------------------------------------------------------------------------
def branch_and_bound(start: Tuple[str, ...], goal: Tuple[str, ...]):
    """
    Uniform-Cost Search (each move costs 1). Returns:
      - path: list of states from start → … → goal
      - nodes_expanded: how many nodes were popped from the frontier
      - elapsed_time: runtime (seconds, float)
    Algorithm:
    1) frontier = min-heap, ordered by g (since h=0 for B&B)
    2) closed = dict mapping state → best g seen so far
    3) Push root Node(start, g=0, h=0)
    4) While frontier not empty:
         a) pop current = smallest‐g Node
         b) if current.state == goal: reconstruct path via parent pointers
         c) if closed[current.state] ≤ current.g: continue  (we’ve already
            expanded this state with equal/lower cost)
         d) closed[current.state] = current.g
         e) for each succ_state in expand(current.state):
                new_g = current.g + 1  (knight move cost = 1)
                old_best = closed.get(succ_state, ∞)
                if new_g < old_best:
                    child = Node(succ_state, new_g, h=0, parent=current)
                    push child into frontier
    5) If loop ends without reaching goal (shouldn’t on this puzzle), return empty path.
    """
    t0 = time.time()
    frontier = []
    root = Node(state=start, g=0, h=0, parent=None)
    heapq.heappush(frontier, root)

    closed = {}            # closed[state] = best g found so far
    nodes_expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current.state == goal:
            # Reconstruct solution path by following parent pointers
            path = []
            n = current
            while n:
                path.append(n.state)
                n = n.parent
            path.reverse()
            return path, nodes_expanded, time.time() - t0

        prev_g = closed.get(current.state)
        if prev_g is not None and prev_g <= current.g:
            continue

        closed[current.state] = current.g

        for succ_state in expand(current.state):
            new_g = current.g + 1
            old_best = closed.get(succ_state, float('inf'))
            if new_g < old_best:
                child = Node(state=succ_state, g=new_g, h=0, parent=current)
                heapq.heappush(frontier, child)

    return [], nodes_expanded, time.time() - t0

# ------------------------------------------------------------------------------
# 5) A* Search Implementation
# ------------------------------------------------------------------------------
def astar_search(start: Tuple[str, ...], goal: Tuple[str, ...]):
    """
    A* Search using f = g + h. Returns:
      - path: list of states from start → … → goal
      - nodes_expanded: how many nodes popped from frontier
      - elapsed_time: runtime in seconds
    Algorithm:
    1) frontier = min-heap ordered by f = g + h
    2) closed = dict mapping state → best f seen so far
    3) Push root Node(start, g=0, h=heuristic(start))
    4) While frontier not empty:
         a) pop current = Node with smallest f
         b) if current.state == goal: reconstruct path via parents
         c) if closed[current.state] ≤ current.f: continue
         d) closed[current.state] = current.f
         e) for each succ_state in expand(current.state):
                new_g = current.g + 1
                new_h = heuristic(succ_state)
                new_f = new_g + new_h
                old_best_f = closed.get(succ_state, ∞)
                if new_f < old_best_f:
                    child = Node(succ_state, new_g, new_h, parent=current)
                    push child into frontier
    """
    t0 = time.time()
    frontier = []
    start_h = heuristic(start)
    root = Node(state=start, g=0, h=start_h, parent=None)
    heapq.heappush(frontier, root)

    closed = {}            # closed[state] = best f found so far
    nodes_expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current.state == goal:
            path = []
            n = current
            while n:
                path.append(n.state)
                n = n.parent
            path.reverse()
            return path, nodes_expanded, time.time() - t0

        prev_f = closed.get(current.state)
        if prev_f is not None and prev_f <= current.f:
            continue

        closed[current.state] = current.f

        for succ_state in expand(current.state):
            new_g = current.g + 1
            new_h = heuristic(succ_state)
            new_f = new_g + new_h
            old_best_f = closed.get(succ_state, float('inf'))
            if new_f < old_best_f:
                child = Node(state=succ_state, g=new_g, h=new_h, parent=current)
                heapq.heappush(frontier, child)

    return [], nodes_expanded, time.time() - t0

# ------------------------------------------------------------------------------
# 6) Utility: Print a 3×3 state in human-readable form
# ------------------------------------------------------------------------------
def print_state(state: Tuple[str, ...]):
    """
    Given a length-9 tuple, print it as three rows of three.
    E.g. ('D','.','D','.','.','.','L','.','L') → 
       D . D
       . . .
       L . L
    """
    for i in range(3):
        row = state[3*i : 3*i + 3]
        print(" ".join(row))
    print()

# ------------------------------------------------------------------------------
# 7) Quick “smoke-test” when running this file directly
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the start state (two dark knights in top corners, two light in bottom corners)
    initial_state = (
        'D', '.', 'D',
        '.', '.', '.',
        'L', '.', 'L'
    )
    # Define the goal state (swap dark ↔ light)
    goal_state = (
        'L', '.', 'L',
        '.', '.', '.',
        'D', '.', 'D'
    )

    print("Branch & Bound (Uniform Cost Search):")
    path_bb, expanded_bb, time_bb = branch_and_bound(initial_state, goal_state)
    print(f"  # moves = {len(path_bb) - 1}")
    print(f"  nodes expanded = {expanded_bb}")
    print(f"  time taken = {time_bb:.6f} seconds\n")

    print("A* Search (heuristic = corner-distance ÷ 2):")
    path_astar, expanded_astar, time_astar = astar_search(initial_state, goal_state)
    print(f"  # moves = {len(path_astar) - 1}")
    print(f"  nodes expanded = {expanded_astar}")
    print(f"  time taken = {time_astar:.6f} seconds\n")

    # If you want to see the actual sequence of states for one solution:
    # print("Solution path:")
    # for s in path_astar:
    #     print_state(s)
