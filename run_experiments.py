import four_knights as fk
import pandas as pd
import matplotlib.pyplot as plt

def run_once(method_name: str, heuristic_name: str = None):
    start_state = (
        'D', '.', 'D',
        '.', '.', '.',
        'L', '.', 'L'
    )
    goal_state = (
        'L', '.', 'L',
        '.', '.', '.',
        'D', '.', 'D'
    )
    if method_name == 'B&B':
        path, nodes, t_elapsed = fk.branch_and_bound(start_state, goal_state)
        method_label = 'B&B'
    elif method_name.startswith('A*'):
        original_h = fk.heuristic
        if heuristic_name == 'div2':
            pass
        elif heuristic_name == 'no_div':
            def h_no_div(state):
                dark_corners = [0, 2]
                light_corners = [6, 8]
                total = 0
                for idx, piece in enumerate(state):
                    if piece == 'D' and idx not in dark_corners:
                        total += min(fk.DIST_LOOKUP[idx][dc] for dc in dark_corners)
                    elif piece == 'L' and idx not in light_corners:
                        total += min(fk.DIST_LOOKUP[idx][lc] for lc in light_corners)
                return total
            fk.heuristic = h_no_div
        elif heuristic_name == 'misplaced':
            def h_misplaced(state):
                dark_corners = [0, 2]
                light_corners = [6, 8]
                count = 0
                for idx, piece in enumerate(state):
                    if piece == 'D' and idx not in dark_corners:
                        count += 1
                    elif piece == 'L' and idx not in light_corners:
                        count += 1
                return count
            fk.heuristic = h_misplaced
        path, nodes, t_elapsed = fk.astar_search(start_state, goal_state)
        method_label = f"A* ({heuristic_name})"
        fk.heuristic = original_h
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return {
        'Method': method_label,
        'Moves': len(path) - 1,
        'NodesExpanded': nodes,
        'TimeSeconds': t_elapsed
    }

def main():
    experiments = []
    experiments.append(run_once('B&B'))
    experiments.append(run_once('A*', heuristic_name='div2'))
    experiments.append(run_once('A*', heuristic_name='no_div'))
    experiments.append(run_once('A*', heuristic_name='misplaced'))
    df = pd.DataFrame(experiments)
    print("\n=== Results ===")
    print(df.to_string(index=False))
    df.to_csv("four_knights_results.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(df['Method'], df['NodesExpanded'], color='skyblue')
    plt.xlabel("Search Method")
    plt.ylabel("Nodes Expanded")
    plt.title("Four Knights: Nodes Expanded by Search Strategy")
    plt.tight_layout()
    plt.savefig("nodes_expanded_comparison.png")
    print("\nChart saved as nodes_expanded_comparison.png")

if __name__ == "__main__":
    main()
