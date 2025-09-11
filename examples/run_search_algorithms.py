"""
Search Algorithms Example

This example demonstrates the classical search algorithms (BFS, DFS, A*) 
on a GridWorld environment and compares them with RL-based solutions.

This shows how search algorithms can serve as baselines for RL performance
and demonstrates the difference between:
- Search algorithms: Find optimal path to known goal
- RL algorithms: Learn optimal policy through experience
"""

import numpy as np
from typing import List, Tuple
from rlnexus.environments.grid_world import GridWorld
from rlnexus.search.algorithms import (
    breadth_first_search,
    depth_first_search,
    a_star_search,
    manhattan_distance,
    euclidean_distance
)
from rlnexus.planning.dp import value_iteration, extract_policy_from_value_function


def create_neighbors_function(env: GridWorld):
    """
    Create a neighbor function for search algorithms based on GridWorld.
    
    Returns a function that gives valid neighbors and their costs for any position.
    """
    def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        neighbors = []
        actions = [0, 1, 2, 3]  # up, right, down, left
        
        for action in actions:
            # Use environment's internal state transition
            next_pos, reward, done, _ = env._get_next_state_info(pos, action)
            
            # Only add if it's a valid move (position changed or we're at goal)
            if next_pos != pos or next_pos == env.goal_pos:
                cost = -reward  # Convert reward to cost (higher reward = lower cost)
                neighbors.append((next_pos, cost))
        
        return neighbors
    
    return get_neighbors


def visualize_path(env: GridWorld, path: List[Tuple[int, int]], algorithm_name: str):
    """Visualize the path found by search algorithm."""
    print(f"\n{algorithm_name} Path Visualization:")
    print("S = Start, G = Goal, # = Obstacle, . = Empty, * = Path")
    
    grid = [['.' for _ in range(env.size)] for _ in range(env.size)]
    
    # Mark obstacles
    for obs in env.obstacles:
        grid[obs[0]][obs[1]] = '#'
    
    # Mark path
    for pos in path[1:-1]:  # Exclude start and goal
        grid[pos[0]][pos[1]] = '*'
    
    # Mark start and goal
    grid[env.start_pos[0]][env.start_pos[1]] = 'S'
    grid[env.goal_pos[0]][env.goal_pos[1]] = 'G'
    
    # Print grid
    for row in grid:
        print(' '.join(row))


def run_search_comparison():
    """Compare different search algorithms on the same GridWorld."""
    print("Search Algorithms Comparison on GridWorld")
    print("=" * 60)
    
    # Create a challenging grid world
    env = GridWorld(
        size=8,
        start=(0, 0),
        goal=(7, 7),
        obstacles=[(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (5, 5), (5, 6), (6, 5)],
        rewards={'step': -0.1, 'goal': 10.0, 'obstacle': -1.0}
    )
    
    print(f"Environment: {env.size}x{env.size} grid")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {len(env.obstacles)} total")
    
    # Create neighbor function
    get_neighbors = create_neighbors_function(env)
    
    # Run different search algorithms
    algorithms = [
        ("Breadth-First Search", lambda: breadth_first_search(env.start_pos, env.goal_pos, get_neighbors)),
        ("Depth-First Search", lambda: depth_first_search(env.start_pos, env.goal_pos, get_neighbors)),
        ("A* (Manhattan)", lambda: a_star_search(env.start_pos, env.goal_pos, get_neighbors, manhattan_distance)),
        ("A* (Euclidean)", lambda: a_star_search(env.start_pos, env.goal_pos, get_neighbors, euclidean_distance)),
    ]
    
    results = []
    
    for name, search_func in algorithms:
        print(f"\nRunning {name}...")
        result = search_func()
        results.append((name, result))
        
        if result.found:
            print(f"✅ Path found! Length: {len(result.path)}, Cost: {result.cost:.1f}, Nodes explored: {result.nodes_explored}")
            visualize_path(env, result.path, name)
        else:
            print(f"❌ No path found. Nodes explored: {result.nodes_explored}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Found':<8} {'Path Length':<12} {'Cost':<8} {'Nodes Explored':<15}")
    print("-" * 60)
    
    for name, result in results:
        found = "Yes" if result.found else "No"
        path_len = len(result.path) if result.found else "N/A"
        cost = f"{result.cost:.1f}" if result.found else "N/A"
        
        print(f"{name:<20} {found:<8} {str(path_len):<12} {cost:<8} {result.nodes_explored:<15}")


def compare_search_vs_rl():
    """Compare search algorithms with RL-based solutions."""
    print("\n" + "=" * 60)
    print("SEARCH vs REINFORCEMENT LEARNING COMPARISON")
    print("=" * 60)
    
    # Same environment as above
    env = GridWorld(
        size=8,
        start=(0, 0),
        goal=(7, 7),
        obstacles=[(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (5, 5), (5, 6), (6, 5)],
        rewards={'step': -0.01, 'goal': 10.0, 'obstacle': -1.0}
    )
    
    # 1. Search algorithm (A* as baseline)
    get_neighbors = create_neighbors_function(env)
    search_result = a_star_search(env.start_pos, env.goal_pos, get_neighbors, manhattan_distance)
    
    # 2. RL algorithm (Value Iteration)
    print("Running Value Iteration (RL approach)...")
    V_optimal = value_iteration(env, theta=1e-6)
    optimal_policy = extract_policy_from_value_function(V_optimal, env)
    
    # Simulate RL policy to get path
    rl_path = [env.start_pos]
    current_state = env.start_pos
    steps = 0
    max_steps = 50
    
    while current_state != env.goal_pos and steps < max_steps:
        action_probs = optimal_policy[current_state]
        action = np.argmax(action_probs)  # Get the action with highest probability
        next_state, _, done, _ = env._get_next_state_info(current_state, action)
        rl_path.append(next_state)
        current_state = next_state
        steps += 1
        
        if done:
            break
    
    # Compare results
    print(f"\nComparison Results:")
    print(f"{'Method':<15} {'Path Length':<12} {'Optimal':<8} {'Computation':<20}")
    print("-" * 55)
    
    search_optimal = "Yes" if search_result.found else "No"
    rl_optimal = "Yes" if current_state == env.goal_pos else "No"
    
    print(f"{'A* Search':<15} {len(search_result.path) if search_result.found else 'N/A':<12} {search_optimal:<8} {'Fast, needs model':<20}")
    print(f"{'Value Iteration':<15} {len(rl_path):<12} {rl_optimal:<8} {'Slow, learns model':<20}")
    
    print(f"\nKey Differences:")
    print(f"• Search algorithms: Fast, require complete environment model")
    print(f"• RL algorithms: Learn through interaction, work with unknown environments")
    print(f"• Search gives single optimal path")
    print(f"• RL learns optimal policy for any start state")


def main():
    """Main function to run all search examples."""
    try:
        # Run search algorithm comparison
        run_search_comparison()
        
        # Compare search vs RL approaches
        compare_search_vs_rl()
        
        print(f"\n Key Takeaways:")
        print(f"1. A* with good heuristics explores fewer nodes than BFS/DFS")
        print(f"2. Search algorithms need complete environment knowledge")
        print(f"3. RL algorithms learn policies that work from any start state")
        print(f"4. Both approaches have their place in AI systems")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure all required modules are properly installed.")


if __name__ == "__main__":
    main()
