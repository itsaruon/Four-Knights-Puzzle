import heapq
import time
from collections import deque
from typing import Tuple, List

KNIGHT_DELTAS = [
    (-2, -1), (-2, +1),
    (-1, -2), (-1, +2),
    (+1, -2), (+1, +2),
    (+2, -1), (+2, +1)
]

def get_knight_moves(idx: int) -> List[int]:
    row, col = divmod(idx, 3)
    moves = []
    for dr, dc in KNIGHT_DELTAS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            moves.append(nr * 3 + nc)
    return moves

def expand(state: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    successors = []
    for idx, piece in enumerate(state):
        if piece in ('L', 'D'):
            for dest in get_knight_moves(idx):
                if state[dest] == '.':
                    new_board = list(state)
                    new_board[dest] = piece
                    new_board[idx] = '.'
                    successors.append(tuple(new_board))
    return successors

def precompute_knight_distances() -> List[List[int]]:
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

DIST_LOOKUP = precompute_knight_distances()

def heuristic(state: Tuple[str, ...]) -> int:
    dark_corners = [0, 2]
    light_corners = [6, 8]
    total = 0
    for idx, piece in enumerate(state):
        if piece == 'D' and idx not in dark_corners:
            total += min(DIST_LOOKUP[idx][dc] for dc in dark_corners)
        elif piece == 'L' and idx not in light_corners:
            total += min(DIST_LOOKUP[idx][lc] for lc in light_corners)
    return total // 2

class Node:
    __slots__ = ('state', 'g', 'h', 'f', 'parent')
    def __init__(self, state: Tuple[str, ...], g: int, h: int, parent: 'Node' = None):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    def __lt__(self, other: 'Node') -> bool:
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

def branch_and_bound(start: Tuple[str, ...], goal: Tuple[str, ...]):
    t0 = time.time()
    frontier = []
    root = Node(state=start, g=0, h=0, parent=None)
    heapq.heappush(frontier, root)
    closed = {}
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

def astar_search(start: Tuple[str, ...], goal: Tuple[str, ...]):
    t0 = time.time()
    frontier = []
    start_h = heuristic(start)
    root = Node(state=start, g=0, h=start_h, parent=None)
    heapq.heappush(frontier, root)
    closed = {}
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

def print_state(state: Tuple[str, ...]):
    for i in range(3):
        row = state[3*i : 3*i + 3]
        print(" ".join(row))
    print()

if __name__ == "__main__":
    initial_state = (
        'D', '.', 'D',
        '.', '.', '.',
        'L', '.', 'L'
    )
    goal_state = (
        'L', '.', 'L',
        '.', '.', '.',
        'D', '.', 'D'
    )
    print("Branch & Bound:")
    path_bb, expanded_bb, time_bb = branch_and_bound(initial_state, goal_state)
    print(f"  moves = {len(path_bb)-1}, nodes = {expanded_bb}, time = {time_bb:.6f}s\n")
    print("A* Search:")
    path_astar, expanded_astar, time_astar = astar_search(initial_state, goal_state)
    print(f"  moves = {len(path_astar)-1}, nodes = {expanded_astar}, time = {time_astar:.6f}s\n")
