"""
Phase 4: Local Planner — A* path planning + pure-pursuit steering + unstuck
===========================================================================
Converts a high-level waypoint (x, z) into per-step PolarAction(forward, rotate):
  1. A* search on occupancy grid for collision-free path
  2. Pure-pursuit lookahead for smooth steering
  3. Unstuck heuristic: back up + turn + mark front as obstacle
"""

import heapq
import math
import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from occupancy_map import OccupancyMap, OCCUPIED, FREE, UNKNOWN


@dataclass
class LocalPlannerConfig:
    lookahead_cells: int = 8
    max_forward_m: float = 0.5
    max_rotate_rad: float = 0.5        # ~28 deg
    stuck_window: int = 5
    stuck_threshold_m: float = 0.15
    escape_rotate_rad: float = 2.5     # ~143 deg
    escape_back_m: float = -0.2
    escape_steps: int = 3
    inflation_radius: int = 5
    astar_max_iters: int = 5000
    replan_dist_cells: int = 10


class LocalPlanner:

    def __init__(self, occ_map: OccupancyMap, cfg: Optional[LocalPlannerConfig] = None):
        self.map = occ_map
        self.cfg = cfg or LocalPlannerConfig()

        self._path_grid: List[Tuple[int, int]] = []
        self._path_index: int = 0
        self._target_rc: Optional[Tuple[int, int]] = None
        self._target_world_xz: Optional[Tuple[float, float]] = None

        self._recent_positions: deque = deque(maxlen=self.cfg.stuck_window)
        self._stuck_count: int = 0
        self._is_stuck: bool = False
        self._escape_steps_remaining: int = 0

    # ── Public API ───────────────────────────────────────────────────────

    def set_target(self, world_xz: Tuple[float, float], agent_pos: np.ndarray):
        """Plan A* path from agent to target."""
        self._target_world_xz = world_xz
        agent_rc = self.map.world_to_grid(agent_pos[0], agent_pos[2])
        target_rc = self.map.world_to_grid(world_xz[0], world_xz[1])
        self._target_rc = target_rc

        path = self._astar(agent_rc, target_rc)
        if path is not None:
            self._path_grid = path
            self._path_index = 0
            logging.info(f'[LocalPlanner] A* path: {len(path)} cells')
        else:
            logging.warning('[LocalPlanner] A* failed, direct steering fallback')
            self._path_grid = [target_rc]
            self._path_index = 0

    def get_action(self, agent_pos: np.ndarray, agent_rot) -> dict:
        """
        Returns dict:
          forward: float (m)
          rotate:  float (rad)
          is_stuck: bool
          reached: bool
        """
        self._recent_positions.append(np.array([agent_pos[0], agent_pos[2]]))
        self._is_stuck = self._check_stuck()

        # ── Escape mode ──────────────────────────────────────────────────
        if self._escape_steps_remaining > 0:
            self._escape_steps_remaining -= 1
            return {
                'forward': self.cfg.escape_back_m,
                'rotate': self.cfg.escape_rotate_rad,
                'is_stuck': True,
                'reached': False,
            }

        if self._is_stuck:
            self._stuck_count += 1
            if self._stuck_count >= 2:
                logging.info(f'[LocalPlanner] Escape triggered (stuck_count={self._stuck_count})')
                self._initiate_escape(agent_pos, agent_rot)
                return {
                    'forward': self.cfg.escape_back_m,
                    'rotate': self.cfg.escape_rotate_rad,
                    'is_stuck': True,
                    'reached': False,
                }
        else:
            self._stuck_count = max(0, self._stuck_count - 1)

        # ── Reached? ─────────────────────────────────────────────────────
        if self._target_world_xz is not None:
            dx = agent_pos[0] - self._target_world_xz[0]
            dz = agent_pos[2] - self._target_world_xz[1]
            if math.sqrt(dx*dx + dz*dz) < 0.4:
                return {'forward': 0, 'rotate': 0, 'is_stuck': False, 'reached': True}

        if not self._path_grid:
            return {'forward': 0, 'rotate': 0, 'is_stuck': False, 'reached': True}

        # ── Follow path ──────────────────────────────────────────────────
        agent_rc = self.map.world_to_grid(agent_pos[0], agent_pos[2])
        self._advance_path_index(agent_rc)

        lookahead_idx = min(self._path_index + self.cfg.lookahead_cells,
                            len(self._path_grid) - 1)
        look_rc = self._path_grid[lookahead_idx]
        look_world = self.map.grid_to_world(look_rc[0], look_rc[1])

        forward, rotate = self._steer_to(agent_pos, agent_rot, look_world)
        return {'forward': forward, 'rotate': rotate, 'is_stuck': False, 'reached': False}

    def needs_replan(self, agent_pos: np.ndarray) -> bool:
        if not self._path_grid:
            return True
        if self._path_index >= len(self._path_grid):
            return True
        agent_rc = self.map.world_to_grid(agent_pos[0], agent_pos[2])
        pr, pc = self._path_grid[min(self._path_index, len(self._path_grid) - 1)]
        return abs(agent_rc[0] - pr) + abs(agent_rc[1] - pc) > self.cfg.replan_dist_cells

    @property
    def stuck_count(self) -> int:
        return self._stuck_count

    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    def reset(self):
        self._path_grid.clear()
        self._path_index = 0
        self._target_rc = None
        self._target_world_xz = None
        self._recent_positions.clear()
        self._stuck_count = 0
        self._is_stuck = False
        self._escape_steps_remaining = 0

    # ── A* ───────────────────────────────────────────────────────────────

    def _astar(self, start_rc, goal_rc) -> Optional[List[Tuple[int, int]]]:
        sr, sc = start_rc
        gr, gc = goal_rc
        if not self.map.in_bounds(gr, gc):
            return None
        if self.map.is_occupied(gr, gc):
            gr, gc = self._nearest_free(gr, gc)
            if gr is None:
                return None

        counter = 0
        open_set = [(self._h(sr, sc, gr, gc), counter, sr, sc)]
        came_from = {}
        g_score = {(sr, sc): 0}
        closed = set()

        while open_set and counter < self.cfg.astar_max_iters:
            _, _, cr, cc = heapq.heappop(open_set)
            if (cr, cc) in closed:
                continue
            closed.add((cr, cc))
            if cr == gr and cc == gc:
                return self._reconstruct(came_from, (gr, gc))

            for dr, dc, cost in [(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),
                                  (-1,-1,1.41),(-1,1,1.41),(1,-1,1.41),(1,1,1.41)]:
                nr, nc = cr + dr, cc + dc
                if not self.map.in_bounds(nr, nc) or self.map.is_occupied(nr, nc):
                    continue
                if self.map.is_unknown(nr, nc) and self._h(nr, nc, gr, gc) > 20:
                    continue
                tent_g = g_score[(cr, cc)] + cost
                if tent_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tent_g
                    came_from[(nr, nc)] = (cr, cc)
                    counter += 1
                    heapq.heappush(open_set, (tent_g + self._h(nr, nc, gr, gc), counter, nr, nc))

        return None

    @staticmethod
    def _h(r1, c1, r2, c2) -> float:
        dr, dc = abs(r1 - r2), abs(c1 - c2)
        return max(dr, dc) + 0.41 * min(dr, dc)

    def _reconstruct(self, came_from, goal):
        path = [goal]
        cur = goal
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def _nearest_free(self, r, c, radius=10):
        queue = deque([(r, c)])
        visited = {(r, c)}
        while queue:
            cr, cc = queue.popleft()
            if self.map.is_free(cr, cc):
                return cr, cc
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) not in visited and self.map.in_bounds(nr, nc):
                    visited.add((nr, nc))
                    if abs(nr - r) + abs(nc - c) <= radius:
                        queue.append((nr, nc))
        return None, None

    # ── Steering ─────────────────────────────────────────────────────────

    def _steer_to(self, agent_pos, agent_rot, target_xz):
        fwd_xz = OccupancyMap._quat_forward_xz(agent_rot)
        agent_yaw = math.atan2(fwd_xz[0], -fwd_xz[1])

        dx = target_xz[0] - agent_pos[0]
        dz = target_xz[1] - agent_pos[2]
        target_yaw = math.atan2(dx, -dz)

        angle_diff = (target_yaw - agent_yaw + math.pi) % (2 * math.pi) - math.pi
        rotate = float(np.clip(angle_diff, -self.cfg.max_rotate_rad, self.cfg.max_rotate_rad))

        alignment = max(0, math.cos(angle_diff))
        dist = math.sqrt(dx * dx + dz * dz)
        forward = min(self.cfg.max_forward_m, dist) * alignment

        if abs(angle_diff) > math.radians(45):
            forward = 0  # turn in place

        return forward, -rotate

    def _advance_path_index(self, agent_rc):
        best_idx, best_dist = self._path_index, float('inf')
        for i in range(self._path_index, min(self._path_index + 20, len(self._path_grid))):
            pr, pc = self._path_grid[i]
            d = abs(agent_rc[0] - pr) + abs(agent_rc[1] - pc)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self._path_index = best_idx

    # ── Stuck / escape ───────────────────────────────────────────────────

    def _check_stuck(self) -> bool:
        if len(self._recent_positions) < self.cfg.stuck_window:
            return False
        positions = list(self._recent_positions)
        return float(np.linalg.norm(positions[-1] - positions[0])) < self.cfg.stuck_threshold_m

    def _initiate_escape(self, agent_pos, agent_rot):
        self._escape_steps_remaining = self.cfg.escape_steps
        self._path_grid.clear()
        self._path_index = 0
        # Inflate obstacles in front
        fwd = OccupancyMap._quat_forward_xz(agent_rot)
        for s in range(1, self.cfg.inflation_radius + 1):
            wx = agent_pos[0] + fwd[0] * s * self.map.cfg.resolution * 3
            wz = agent_pos[2] + fwd[1] * s * self.map.cfg.resolution * 3
            r, c = self.map.world_to_grid(wx, wz)
            if self.map.in_bounds(r, c):
                self.map.occupancy[r, c] = OCCUPIED
