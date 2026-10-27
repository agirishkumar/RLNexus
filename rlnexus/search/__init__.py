"""
Search algorithms module for pathfinding and graph traversal.

This module provides classical search algorithms that can be used for:
- Pathfinding in grid environments
- Baseline comparisons with RL algorithms
- Heuristic guidance for exploration
- Analysis of optimal solutions

Available algorithms:
- Breadth-First Search (BFS): Guarantees shortest path in unweighted graphs
- Depth-First Search (DFS): Memory efficient, explores deeply
- A* Search: Optimal pathfinding with admissible heuristics
"""

from .algorithms import (
    breadth_first_search,
    depth_first_search,
    a_star_search,
    manhattan_distance,
    euclidean_distance,
    SearchResult
)

__all__ = [
    'breadth_first_search',
    'depth_first_search', 
    'a_star_search',
    'manhattan_distance',
    'euclidean_distance',
    'SearchResult'
]
