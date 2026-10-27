# Search Algorithms

Classical search algorithms are fundamental tools in AI that complement reinforcement learning by providing optimal solutions for navigation and pathfinding problems. This page covers the search algorithms implemented in RLNexus and how they relate to RL methods.

## Overview

Search algorithms solve pathfinding problems by systematically exploring a graph or state space to find a path from start to goal. Unlike RL algorithms that learn through trial and error, search algorithms use complete knowledge of the environment.

### Search vs Reinforcement Learning

| Aspect | Search Algorithms | Reinforcement Learning |
|--------|-------------------|------------------------|
| **Environment Knowledge** | Complete model required | Learns through interaction |
| **Goal** | Find path to specific target | Learn optimal policy for any state |
| **Usage** | Known start/goal positions | Unknown reward structure |
| **Optimality** | Guaranteed (with right algorithm) | Converges to optimal |
| **Speed** | Fast with good heuristics | Slow, requires many episodes |

## Algorithm Implementations

### Breadth-First Search (BFS)

BFS explores nodes level by level, guaranteeing the shortest path in unweighted graphs.

**Characteristics:**
- **Completeness**: Always finds solution if one exists
- **Optimality**: Finds shortest path (minimum steps)
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)

**When to Use:**
- Need guaranteed shortest path
- Graph is unweighted or has uniform costs
- Memory is not a constraint

```python
from rlnexus.search import breadth_first_search

result = breadth_first_search(
    start=(0, 0),
    goal=(5, 5), 
    get_neighbors=grid_neighbors_function
)

if result.found:
    print(f"Path: {result.path}")
    print(f"Cost: {result.cost}")
    print(f"Nodes explored: {result.nodes_explored}")
```

### Depth-First Search (DFS)

DFS explores as deeply as possible before backtracking. Memory efficient but doesn't guarantee optimal solution.

**Characteristics:**
- **Completeness**: Finds solution if one exists (in finite graphs)
- **Optimality**: No guarantee of shortest path
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(d) where d is maximum depth

**When to Use:**
- Memory is limited
- Any solution is acceptable (not necessarily optimal)
- Graph is very deep but has solution near start

```python
from rlnexus.search import depth_first_search

result = depth_first_search(
    start=(0, 0),
    goal=(5, 5),
    get_neighbors=grid_neighbors_function
)
```

### A* Search

A* uses both actual cost and heuristic estimates to guide search efficiently toward the goal.

**Algorithm:**
```
f(n) = g(n) + h(n)
where:
- g(n) = actual cost from start to n
- h(n) = heuristic estimate from n to goal  
- f(n) = estimated total cost through n
```

**Characteristics:**
- **Completeness**: Always finds solution if one exists
- **Optimality**: Optimal if heuristic is admissible
- **Time Complexity**: O(b^d) exponential in worst case
- **Space Complexity**: O(b^d) stores all nodes in memory

**When to Use:**
- Need optimal solution
- Have good heuristic function
- Can afford memory usage

```python
from rlnexus.search import a_star_search, manhattan_distance

result = a_star_search(
    start=(0, 0),
    goal=(5, 5),
    get_neighbors=grid_neighbors_function,
    heuristic=manhattan_distance
)
```

### Dijkstra's Algorithm

Dijkstra's is essentially A* with zero heuristic. Finds shortest path in weighted graphs.

**Characteristics:**
- **Completeness**: Always finds solution if one exists  
- **Optimality**: Guarantees shortest path in weighted graphs
- **Use Case**: When you don't have a good heuristic function

```python
from rlnexus.search import dijkstra_search

result = dijkstra_search(
    start=(0, 0),
    goal=(5, 5), 
    get_neighbors=weighted_grid_neighbors
)
```

## Heuristic Functions

Good heuristics are crucial for A* performance. They must be **admissible** (never overestimate) for optimality.

### Manhattan Distance
```python
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```
- **Use**: Grid worlds with 4-directional movement
- **Admissible**: Yes, for grid navigation

### Euclidean Distance
```python
def euclidean_distance(pos1, pos2):
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```
- **Use**: Continuous spaces or diagonal movement allowed
- **Admissible**: Yes, straight-line distance is minimum possible

## Practical Usage with RLNexus

### Creating Neighbor Functions

For grid environments like GridWorld:

```python
def create_grid_neighbors(env):
    def get_neighbors(pos):
        neighbors = []
        for action in [0, 1, 2, 3]:  # up, right, down, left
            # Use environment's transition function
            next_pos, reward, done, _ = env.get_transition(pos, action)
            if next_pos != pos:  # Valid move
                cost = -reward  # Convert reward to cost
                neighbors.append((next_pos, cost))
        return neighbors
    return get_neighbors
```

### Comparing with RL Solutions

```python
# Search solution (optimal path)
search_result = a_star_search(start, goal, get_neighbors, manhattan_distance)

# RL solution (learned policy)  
rl_policy = train_value_iteration(env)
rl_path = simulate_policy(rl_policy, start, goal)

print(f"Search path length: {len(search_result.path)}")
print(f"RL path length: {len(rl_path)}")
print(f"Search nodes explored: {search_result.nodes_explored}")
```

## Algorithm Selection Guide

Choose the right algorithm based on your needs:

### Use BFS when:
- ✅ Need guaranteed shortest path
- ✅ Graph is unweighted  
- ✅ Have sufficient memory
- ❌ Graph is very large

### Use DFS when:
- ✅ Memory is severely limited
- ✅ Solution depth is small
- ✅ Any solution is acceptable
- ❌ Need optimal solution

### Use A* when:
- ✅ Need optimal solution
- ✅ Have good admissible heuristic
- ✅ Can afford memory usage
- ❌ No good heuristic available

### Use Dijkstra when:
- ✅ Need optimal solution in weighted graph
- ✅ No good heuristic available
- ✅ Graph is not too large
- ❌ Have good heuristic (use A* instead)

## Performance Characteristics

For a 10x10 grid from (0,0) to (9,9):

| Algorithm | Nodes Explored | Path Length | Optimal |
|-----------|----------------|-------------|---------|
| BFS | ~50-60 | 18 | ✅ |
| DFS | ~10-30 | 20-40 | ❌ |
| A* (Manhattan) | ~20-25 | 18 | ✅ |
| A* (Euclidean) | ~18-22 | 18 | ✅ |

## Integration with RL

Search algorithms complement RL in several ways:

### 1. Baseline Comparison
```python
# Compare RL agent performance against optimal
optimal_score = len(a_star_path)
rl_score = len(rl_agent_path)
efficiency = optimal_score / rl_score
```

### 2. Heuristic Guidance
```python
# Use A* to guide exploration in RL
def guided_epsilon_greedy(state, q_values, epsilon):
    if random() < epsilon:
        # Use A* to suggest good actions
        astar_action = get_astar_direction(state, goal)
        return astar_action
    else:
        return argmax(q_values)
```

### 3. Curriculum Learning
```python
# Start with search-guided policy, gradually remove guidance
guidance_weight = max(0, 1 - episode / 1000) 
action = guidance_weight * search_action + (1 - guidance_weight) * rl_action
```

### 4. Verification
```python
# Verify that learned policy achieves near-optimal performance
if rl_path_length <= optimal_path_length * 1.1:
    print("RL agent learned near-optimal policy!")
```

## Example: Search vs RL on GridWorld

```python
from rlnexus.environments.grid_world import GridWorld
from rlnexus.search import a_star_search, manhattan_distance
from rlnexus.planning.dp import value_iteration

# Create environment
env = GridWorld(size=8, obstacles=[(3,3), (3,4), (4,3)])

# Method 1: Search (fast, needs model)
neighbors = create_grid_neighbors(env)
search_result = a_star_search(env.start, env.goal, neighbors, manhattan_distance)

# Method 2: RL (slow, learns from interaction)
V_optimal, _ = value_iteration(env)
rl_policy = extract_policy_from_value_function(env, V_optimal)

print("Search: Fast optimal solution with complete knowledge")
print(f"RL: Learned optimal policy applicable to any start state")
```

Both approaches have their place in intelligent systems - search for known problems, RL for unknown environments!
