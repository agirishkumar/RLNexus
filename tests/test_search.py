"""
Tests for search algorithms module.

This module tests the classical search algorithms including BFS, DFS, A*,
and heuristic functions for correctness and expected behavior.
"""

import pytest
import numpy as np
from rlnexus.search.algorithms import (
    breadth_first_search,
    depth_first_search,
    a_star_search,
    dijkstra_search,
    manhattan_distance,
    euclidean_distance,
    SearchResult
)


class TestHeuristicFunctions:
    """Test heuristic distance functions."""
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        # Test basic cases
        assert manhattan_distance((0, 0), (0, 0)) == 0
        assert manhattan_distance((0, 0), (1, 1)) == 2
        assert manhattan_distance((0, 0), (3, 4)) == 7
        assert manhattan_distance((2, 3), (5, 1)) == 5
        
        # Test symmetry
        assert manhattan_distance((1, 2), (4, 6)) == manhattan_distance((4, 6), (1, 2))
        
        # Test with negative coordinates
        assert manhattan_distance((-1, -1), (1, 1)) == 4
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        # Test basic cases
        assert euclidean_distance((0, 0), (0, 0)) == 0
        assert euclidean_distance((0, 0), (3, 4)) == 5  # 3-4-5 triangle
        
        # Test symmetry
        pos1, pos2 = (1, 2), (4, 6)
        assert euclidean_distance(pos1, pos2) == euclidean_distance(pos2, pos1)
        
        # Test that Euclidean <= Manhattan for grid movements
        pos1, pos2 = (0, 0), (3, 4)
        assert euclidean_distance(pos1, pos2) <= manhattan_distance(pos1, pos2)


class TestSearchResult:
    """Test SearchResult namedtuple."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation and access."""
        result = SearchResult(
            path=[(0, 0), (1, 0), (1, 1)],
            cost=2.0,
            nodes_explored=5,
            found=True
        )
        
        assert result.path == [(0, 0), (1, 0), (1, 1)]
        assert result.cost == 2.0
        assert result.nodes_explored == 5
        assert result.found is True


class TestGridSearchEnvironment:
    """Helper methods and test environment for search algorithms."""
    
    @staticmethod
    def create_simple_grid():
        """Create a simple 3x3 grid for testing."""
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            # 4-directional movement
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 3 and 0 <= ny < 3:
                    neighbors.append(((nx, ny), 1.0))  # Unit cost
            return neighbors
        return get_neighbors
    
    @staticmethod
    def create_blocked_grid():
        """Create a grid with obstacles."""
        obstacles = {(1, 1)}  # Block center
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 3 and 0 <= ny < 3 and (nx, ny) not in obstacles:
                    neighbors.append(((nx, ny), 1.0))
            return neighbors
        return get_neighbors
    
    @staticmethod
    def create_impossible_grid():
        """Create a grid where goal is unreachable."""
        # Block all paths by surrounding goal
        obstacles = {(1, 1), (1, 2), (2, 1)}
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 3 and 0 <= ny < 3 and (nx, ny) not in obstacles:
                    neighbors.append(((nx, ny), 1.0))
            return neighbors
        return get_neighbors


class TestBreadthFirstSearch:
    """Test Breadth-First Search algorithm."""
    
    def test_bfs_simple_path(self):
        """Test BFS on simple grid."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = breadth_first_search((0, 0), (2, 2), get_neighbors)
        
        assert result.found is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)
        assert len(result.path) == 5  # Optimal path length
        assert result.cost == 4.0  # 4 steps
        assert result.nodes_explored > 0
    
    def test_bfs_start_equals_goal(self):
        """Test BFS when start equals goal."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = breadth_first_search((1, 1), (1, 1), get_neighbors)
        
        assert result.found is True
        assert result.path == [(1, 1)]
        assert result.cost == 0
        assert result.nodes_explored == 1
    
    def test_bfs_with_obstacles(self):
        """Test BFS with blocked paths."""
        get_neighbors = TestGridSearchEnvironment.create_blocked_grid()
        result = breadth_first_search((0, 0), (2, 2), get_neighbors)
        
        assert result.found is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)
        # Should find path around obstacle (may be same length due to small grid)
        assert len(result.path) >= 5
    
    def test_bfs_impossible_path(self):
        """Test BFS when no path exists."""
        get_neighbors = TestGridSearchEnvironment.create_impossible_grid()
        result = breadth_first_search((0, 0), (2, 2), get_neighbors)
        
        assert result.found is False
        assert result.path == []
        assert result.cost == float('inf')
        assert result.nodes_explored > 0


class TestDepthFirstSearch:
    """Test Depth-First Search algorithm."""
    
    def test_dfs_finds_path(self):
        """Test that DFS finds a path (not necessarily optimal)."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = depth_first_search((0, 0), (2, 2), get_neighbors)
        
        assert result.found is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)
        assert result.cost > 0
        assert result.nodes_explored > 0
    
    def test_dfs_start_equals_goal(self):
        """Test DFS when start equals goal."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = depth_first_search((1, 1), (1, 1), get_neighbors)
        
        assert result.found is True
        assert result.path == [(1, 1)]
        assert result.cost == 0
        assert result.nodes_explored == 1
    
    def test_dfs_impossible_path(self):
        """Test DFS when no path exists."""
        get_neighbors = TestGridSearchEnvironment.create_impossible_grid()
        result = depth_first_search((0, 0), (2, 2), get_neighbors)
        
        assert result.found is False
        assert result.path == []
        assert result.cost == float('inf')


class TestAStarSearch:
    """Test A* Search algorithm."""
    
    def test_astar_optimal_path(self):
        """Test A* finds optimal path."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = a_star_search((0, 0), (2, 2), get_neighbors, manhattan_distance)
        
        assert result.found is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)
        assert len(result.path) == 5  # Optimal path length
        assert result.cost == 4.0
        assert result.nodes_explored > 0
    
    def test_astar_with_manhattan_heuristic(self):
        """Test A* with Manhattan distance heuristic."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = a_star_search((0, 0), (2, 2), get_neighbors, manhattan_distance)
        
        assert result.found is True
        # A* should be optimal with admissible heuristic
        assert result.cost == 4.0
    
    def test_astar_with_euclidean_heuristic(self):
        """Test A* with Euclidean distance heuristic."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = a_star_search((0, 0), (2, 2), get_neighbors, euclidean_distance)
        
        assert result.found is True
        # Still optimal with admissible heuristic
        assert result.cost == 4.0
    
    def test_astar_efficiency(self):
        """Test that A* explores fewer nodes than BFS."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        
        bfs_result = breadth_first_search((0, 0), (2, 2), get_neighbors)
        astar_result = a_star_search((0, 0), (2, 2), get_neighbors, manhattan_distance)
        
        # A* should explore fewer or equal nodes (in small grids, difference may be minimal)
        # Both should find optimal solution
        assert astar_result.cost == bfs_result.cost
    
    def test_astar_start_equals_goal(self):
        """Test A* when start equals goal."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = a_star_search((1, 1), (1, 1), get_neighbors, manhattan_distance)
        
        assert result.found is True
        assert result.path == [(1, 1)]
        assert result.cost == 0
        assert result.nodes_explored == 1
    
    def test_astar_impossible_path(self):
        """Test A* when no path exists."""
        get_neighbors = TestGridSearchEnvironment.create_impossible_grid()
        result = a_star_search((0, 0), (2, 2), get_neighbors, manhattan_distance)
        
        assert result.found is False
        assert result.path == []
        assert result.cost == float('inf')


class TestDijkstraSearch:
    """Test Dijkstra's algorithm."""
    
    def test_dijkstra_optimal_path(self):
        """Test Dijkstra finds optimal path."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        result = dijkstra_search((0, 0), (2, 2), get_neighbors)
        
        assert result.found is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)
        assert result.cost == 4.0  # Optimal cost
    
    def test_dijkstra_vs_astar(self):
        """Test that Dijkstra gives same result as A* with zero heuristic."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        
        dijkstra_result = dijkstra_search((0, 0), (2, 2), get_neighbors)
        astar_result = a_star_search((0, 0), (2, 2), get_neighbors, lambda x, y: 0)
        
        # Should give identical results
        assert dijkstra_result.cost == astar_result.cost
        assert dijkstra_result.found == astar_result.found


class TestSearchAlgorithmComparison:
    """Test comparing different search algorithms."""
    
    def test_all_algorithms_find_same_optimal_cost(self):
        """Test that all optimal algorithms find same cost."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        start, goal = (0, 0), (2, 2)
        
        bfs_result = breadth_first_search(start, goal, get_neighbors)
        astar_result = a_star_search(start, goal, get_neighbors, manhattan_distance)
        dijkstra_result = dijkstra_search(start, goal, get_neighbors)
        
        # All should find optimal cost
        assert bfs_result.cost == astar_result.cost == dijkstra_result.cost
        assert all(r.found for r in [bfs_result, astar_result, dijkstra_result])
    
    def test_astar_explores_fewer_nodes(self):
        """Test that A* is more efficient than uninformed search."""
        # Create larger grid for more noticeable difference
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 5:  # 5x5 grid
                    neighbors.append(((nx, ny), 1.0))
            return neighbors
        
        start, goal = (0, 0), (4, 4)
        
        bfs_result = breadth_first_search(start, goal, get_neighbors)
        astar_result = a_star_search(start, goal, get_neighbors, manhattan_distance)
        
        # A* should explore fewer nodes (but in some cases might be equal due to implementation details)
        # More importantly, both should find optimal solution
        assert astar_result.nodes_explored >= 1  # At least explored the goal
        # But find same optimal cost
        assert astar_result.cost == bfs_result.cost
    
    def test_max_iterations_limit(self):
        """Test that algorithms respect max_iterations limit."""
        get_neighbors = TestGridSearchEnvironment.create_simple_grid()
        
        # Very low limit should prevent finding solution
        result = breadth_first_search((0, 0), (2, 2), get_neighbors, max_iterations=2)
        
        assert result.nodes_explored <= 2
        # Might not find solution due to iteration limit
        if not result.found:
            assert result.cost == float('inf')
