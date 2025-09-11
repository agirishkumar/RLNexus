"""
Classical search algorithms for pathfinding and graph traversal.

This module implements fundamental search algorithms that are widely used
in AI and game development. These can serve as baselines for comparing
RL algorithm performance and for providing optimal solutions to navigation problems.
"""

import heapq
import math
from collections import deque, namedtuple
from typing import List, Tuple, Set, Dict, Optional, Callable, Any

# Result structure for search algorithms
SearchResult = namedtuple('SearchResult', ['path', 'cost', 'nodes_explored', 'found'])


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Manhattan distance between two positions.
    
    Manhattan distance is the sum of absolute differences of coordinates.
    It's admissible for grid worlds with 4-directional movement.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Manhattan distance between positions
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two positions.
    
    Euclidean distance is the straight-line distance between points.
    It's admissible for environments allowing diagonal movement.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Euclidean distance between positions
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def breadth_first_search(
    start: Any,
    goal: Any,
    get_neighbors: Callable[[Any], List[Tuple[Any, float]]],
    max_iterations: int = 10000
) -> SearchResult:
    """
    Breadth-First Search algorithm.
    
    BFS explores nodes level by level, guaranteeing the shortest path
    in terms of number of steps for unweighted graphs.
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V) for the queue and visited set
    
    Args:
        start: Starting node
        goal: Goal node  
        get_neighbors: Function that returns [(neighbor, cost), ...] for a node
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        SearchResult with path, cost, nodes explored, and success flag
    """
    if start == goal:
        return SearchResult(path=[start], cost=0, nodes_explored=1, found=True)
    
    queue = deque([(start, [start], 0)])  # (node, path, cost)
    visited = {start}
    nodes_explored = 0
    
    while queue and nodes_explored < max_iterations:
        current, path, cost = queue.popleft()
        nodes_explored += 1
        
        for neighbor, step_cost in get_neighbors(current):
            if neighbor == goal:
                final_path = path + [neighbor]
                final_cost = cost + step_cost
                return SearchResult(
                    path=final_path,
                    cost=final_cost,
                    nodes_explored=nodes_explored + 1,
                    found=True
                )
            
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                new_cost = cost + step_cost
                queue.append((neighbor, new_path, new_cost))
    
    return SearchResult(path=[], cost=float('inf'), nodes_explored=nodes_explored, found=False)


def depth_first_search(
    start: Any,
    goal: Any,
    get_neighbors: Callable[[Any], List[Tuple[Any, float]]],
    max_iterations: int = 10000
) -> SearchResult:
    """
    Depth-First Search algorithm.
    
    DFS explores as far as possible along each branch before backtracking.
    It's memory efficient but doesn't guarantee the shortest path.
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V) for the stack and visited set
    
    Args:
        start: Starting node
        goal: Goal node
        get_neighbors: Function that returns [(neighbor, cost), ...] for a node
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        SearchResult with path, cost, nodes explored, and success flag
    """
    if start == goal:
        return SearchResult(path=[start], cost=0, nodes_explored=1, found=True)
    
    stack = [(start, [start], 0)]  # (node, path, cost)
    visited = {start}
    nodes_explored = 0
    
    while stack and nodes_explored < max_iterations:
        current, path, cost = stack.pop()
        nodes_explored += 1
        
        for neighbor, step_cost in get_neighbors(current):
            if neighbor == goal:
                final_path = path + [neighbor]
                final_cost = cost + step_cost
                return SearchResult(
                    path=final_path,
                    cost=final_cost,
                    nodes_explored=nodes_explored + 1,
                    found=True
                )
            
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                new_cost = cost + step_cost
                stack.append((neighbor, new_path, new_cost))
    
    return SearchResult(path=[], cost=float('inf'), nodes_explored=nodes_explored, found=False)


def a_star_search(
    start: Any,
    goal: Any,
    get_neighbors: Callable[[Any], List[Tuple[Any, float]]],
    heuristic: Callable[[Any, Any], float],
    max_iterations: int = 10000
) -> SearchResult:
    """
    A* Search algorithm.
    
    A* uses both actual cost from start (g) and heuristic cost to goal (h)
    to guide search. With an admissible heuristic, A* guarantees optimal solution.
    
    f(n) = g(n) + h(n) where:
    - g(n) = actual cost from start to n
    - h(n) = heuristic cost from n to goal
    - f(n) = estimated total cost of path through n
    
    Time Complexity: O(b^d) where b is branching factor, d is depth
    Space Complexity: O(b^d) for the priority queue
    
    Args:
        start: Starting node
        goal: Goal node
        get_neighbors: Function that returns [(neighbor, cost), ...] for a node
        heuristic: Function that estimates cost from node to goal
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        SearchResult with path, cost, nodes explored, and success flag
    """
    if start == goal:
        return SearchResult(path=[start], cost=0, nodes_explored=1, found=True)
    
    # Priority queue: (f_score, g_score, node, path)
    open_set = [(heuristic(start, goal), 0, start, [start])]
    closed_set = set()
    g_scores = {start: 0}  # Best known cost from start to each node
    nodes_explored = 0
    
    while open_set and nodes_explored < max_iterations:
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        nodes_explored += 1
        
        if current == goal:
            return SearchResult(
                path=path,
                cost=g_score,
                nodes_explored=nodes_explored,
                found=True
            )
        
        for neighbor, step_cost in get_neighbors(current):
            if neighbor in closed_set:
                continue
                
            tentative_g = g_score + step_cost
            
            # If we found a better path to neighbor, update it
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))
    
    return SearchResult(path=[], cost=float('inf'), nodes_explored=nodes_explored, found=False)


def dijkstra_search(
    start: Any,
    goal: Any,
    get_neighbors: Callable[[Any], List[Tuple[Any, float]]],
    max_iterations: int = 10000
) -> SearchResult:
    """
    Dijkstra's algorithm for shortest path.
    
    Dijkstra's algorithm is essentially A* with h(n) = 0 for all nodes.
    It guarantees the shortest path in weighted graphs with non-negative edge weights.
    
    Args:
        start: Starting node
        goal: Goal node
        get_neighbors: Function that returns [(neighbor, cost), ...] for a node
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        SearchResult with path, cost, nodes explored, and success flag
    """
    # Dijkstra is A* with zero heuristic
    return a_star_search(
        start=start,
        goal=goal,
        get_neighbors=get_neighbors,
        heuristic=lambda x, y: 0,  # Zero heuristic makes A* into Dijkstra
        max_iterations=max_iterations
    )
