import four_knights as fk
import pandas as pd
import matplotlib.pyplot as plt

def run_once(method_name: str, heuristic_name: str = None):
    """
    Run one experiment, return a dict with:
      - 'Method': e.g. 'B&B' or 'A* (div2)' etc.
      - 'Moves': number of moves in the found solution
      - 'NodesExpanded': how many nodes were popped from the frontier
      - 'TimeSeconds': how long (seconds) the search took
    """
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
            # default heuristic already does “÷2”
            pass
        elif heuristic_name == 'no_div':
            # Inadmissible variant: do not divide by 2
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
            # Weak admissible: count knights not in correct corner
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
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

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
    # Branch & Bound with no heuristic
    experiments.append(run_once('B&B'))
    # A* with default (÷2) heuristic
    experiments.append(run_once('A*', heuristic_name='div2'))
    # A* with no ÷2 (inadmissible)
    experiments.append(run_once('A*', heuristic_name='no_div'))
    # A* with “misplaced” heuristic (admissible but weaker)
    experiments.append(run_once('A*', heuristic_name='misplaced'))

    df = pd.DataFrame(experiments)

    # Print a neatly formatted table
    print("\n=== Four Knights Search Experiment Results ===")
    print(df.to_string(index=False))

    # Save results to CSV (for inclusion in your memo later)
    df.to_csv("four_knights_results.csv", index=False)

    # Plot a bar chart of NodesExpanded vs. Method
    plt.figure(figsize=(7, 4))
    plt.bar(df['Method'], df['NodesExpanded'], color='skyblue')
    plt.xlabel("Search Method")
    plt.ylabel("Nodes Expanded")
    plt.title("Four Knights: Nodes Expanded by Search Strategy")
    plt.tight_layout()
    plt.savefig("nodes_expanded_comparison.png")
    print("\nBar chart saved as nodes_expanded_comparison.png")

if __name__ == "__main__":
    main()
