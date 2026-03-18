"""
Memory Module (Minimal)
========================
Only two components remain:
  - LongTermMemory:  exploration map + target coordinate storage
  - MemoryManager:   thin wrapper with direction-failure tracking

ShortTermMemory removed — stuck detection is now handled entirely by
LocalPlanner (position-window check + escape manoeuvre + obstacle inflation).
"""

import logging
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------
class LongTermMemory:
    """
    Coarse exploration map (separate from the fine-grained OccupancyMap)
    and persistent target coordinate storage across steps within an episode.
    """
    UNVISITED = 0
    VISITED = 1

    def __init__(self, map_size: int = 5000, scale: float = 50.0):
        self.map_size = map_size
        self.scale = scale
        self.exploration_map = np.zeros((map_size, map_size), dtype=np.uint8)
        self.target_coords: List[np.ndarray] = []
        self.init_pos: Optional[np.ndarray] = None
        self._visit_radius = 30

    def set_init_pos(self, pos: np.ndarray):
        self.init_pos = pos

    def _global_to_grid(self, position) -> Tuple[int, int]:
        if self.init_pos is None:
            return (self.map_size // 2, self.map_size // 2)
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        x = int(self.map_size // 2 + dx * self.scale)
        y = int(self.map_size // 2 + dz * self.scale)
        return (np.clip(x, 0, self.map_size - 1), np.clip(y, 0, self.map_size - 1))

    def mark_visited(self, position, radius: Optional[int] = None):
        gx, gy = self._global_to_grid(position)
        cv2.circle(self.exploration_map, (gx, gy), radius or self._visit_radius,
                   self.VISITED, -1)

    def record_target_coord(self, world_coord):
        """Record a confirmed target coordinate (dedup within 0.5m)."""
        arr = np.array(world_coord).copy()
        for c in self.target_coords:
            if np.linalg.norm(c - arr) < 0.5:
                return
        self.target_coords.append(arr)
        logging.info(f"[LongTermMemory] Recorded target at {world_coord}")

    def has_target_coords(self) -> bool:
        return len(self.target_coords) > 0

    def get_nearest_target_coord(self, agent_pos) -> Optional[np.ndarray]:
        if not self.target_coords:
            return None
        dists = [
            np.linalg.norm(np.array([c[0] - agent_pos[0], c[2] - agent_pos[2]]))
            for c in self.target_coords
        ]
        return self.target_coords[int(np.argmin(dists))]

    def get_largest_unexplored_direction(self, agent_pos,
                                          num_directions: int = 12,
                                          search_radius: int = 200) -> Optional[float]:
        """Return angle (radians) of the direction with most unexplored cells."""
        gx, gy = self._global_to_grid(agent_pos)
        best_angle, best_count = None, 0
        for i in range(num_directions):
            angle = 2 * np.pi * i / num_directions
            count = 0
            for r in range(10, search_radius, 5):
                cx = int(gx + r * np.cos(angle))
                cy = int(gy + r * np.sin(angle))
                if 0 <= cx < self.map_size and 0 <= cy < self.map_size:
                    if self.exploration_map[cy, cx] == self.UNVISITED:
                        count += 1
            if count > best_count:
                best_count = count
                best_angle = angle
        return best_angle

    def get_exploration_ratio(self, agent_pos, radius: int = 300) -> float:
        gx, gy = self._global_to_grid(agent_pos)
        x1, x2 = max(0, gx - radius), min(self.map_size, gx + radius)
        y1, y2 = max(0, gy - radius), min(self.map_size, gy + radius)
        region = self.exploration_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region == self.VISITED)) / float(region.size)

    def is_fully_explored(self, agent_pos, radius: int = 300,
                          threshold: float = 0.92) -> bool:
        return self.get_exploration_ratio(agent_pos, radius) >= threshold

    def reset(self):
        self.exploration_map[:] = self.UNVISITED
        self.target_coords.clear()
        self.init_pos = None


# ---------------------------------------------------------------------------
# MemoryManager — thin wrapper
# ---------------------------------------------------------------------------
class MemoryManager:
    """
    Wraps LongTermMemory + direction-failure tracking.
    No ShortTermMemory — stuck detection lives in LocalPlanner.
    """

    def __init__(self, map_size: int = 5000, scale: float = 50.0):
        self.long_term = LongTermMemory(map_size=map_size, scale=scale)

        # Direction failure tracking (episode-scoped)
        self.direction_fail_count: Dict[int, int] = {}
        self._last_direction: Optional[int] = None
        self._last_position: Optional[np.ndarray] = None

    # ── Per-step update ───────────────────────────────────────────────────

    def process_step(self, agent_pos: np.ndarray) -> dict:
        """
        Lightweight per-step update.  Returns signals dict for the agent.
        """
        self.long_term.mark_visited(agent_pos)

        has_hist = self.long_term.has_target_coords()
        nearest = self.long_term.get_nearest_target_coord(agent_pos) if has_hist else None

        return {
            'has_historical_target': has_hist,
            'nearest_target_coord': nearest,
            'fully_explored': self.long_term.is_fully_explored(agent_pos),
            'exploration_ratio': self.long_term.get_exploration_ratio(agent_pos),
        }

    # ── Direction failure tracking ────────────────────────────────────────

    def update_direction_feedback(self, current_pos: np.ndarray,
                                  chosen_direction: int):
        """Track which panoramic directions consistently fail to make progress."""
        if self._last_position is not None and self._last_direction is not None:
            disp = np.linalg.norm(np.array([
                current_pos[0] - self._last_position[0],
                current_pos[2] - self._last_position[2],
            ]))
            if disp < 0.15:
                self.direction_fail_count[self._last_direction] = \
                    self.direction_fail_count.get(self._last_direction, 0) + 1
            else:
                self.direction_fail_count[self._last_direction] = 0
        self._last_position = np.array(current_pos)
        self._last_direction = chosen_direction

    def get_direction_penalty(self, direction_idx: int) -> float:
        return self.direction_fail_count.get(direction_idx, 0) * 3.0

    # ── Termination helpers ───────────────────────────────────────────────

    def should_terminate_success(self, agent_pos, threshold: float = 0.8) -> bool:
        return any(
            np.linalg.norm(np.array([c[0] - agent_pos[0], c[2] - agent_pos[2]])) < threshold
            for c in self.long_term.target_coords
        )

    def should_terminate_failure(self, agent_pos) -> bool:
        return (self.long_term.is_fully_explored(agent_pos)
                and not self.long_term.has_target_coords())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        self.long_term.reset()
        self.direction_fail_count.clear()
        self._last_direction = None
        self._last_position = None
