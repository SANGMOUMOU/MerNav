"""
Phase 1: Occupancy Grid Map + Semantic Coverage Map
====================================================
Core principle: "Seeing is Exploring" — if the camera's frustum has swept over
a region and no target was detected, that region is marked Checked and will
never receive a waypoint again.

Three-layer grid (all share the same metric resolution):
  - occupancy : uint8  (UNKNOWN=0, FREE=1, OCCUPIED=2)
  - checked   : bool   (True = visually verified, no target present)
  - visit     : uint16 (physical visit count per cell)
"""

import numpy as np
import math
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

# ── Cell states ──────────────────────────────────────────────────────────
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


@dataclass
class OccupancyMapConfig:
    resolution: float = 0.05          # metres per cell
    map_size_m: float = 50.0          # half-extent in metres (total side = 2×this)
    agent_radius_cells: int = 3       # inflation radius around obstacles
    frustum_max_depth: float = 5.0    # max reliable depth (metres)
    floor_height_tol: float = 0.25    # |h - agent_h| < tol → floor
    ceiling_height: float = 0.15      # above agent+this → ceiling, skip
    min_depth: float = 0.1            # discard depth < this
    checked_fov_shrink: float = 0.9   # conservative shrink of FOV polygon
    ray_subsample: int = 4            # subsample depth pixels by this factor


class OccupancyMap:
    """
    2-D bird's-eye occupancy grid updated via depth-based raycasting,
    plus a semantic-coverage (checked) layer updated via FOV polygon projection.
    """

    def __init__(self, cfg: Optional[OccupancyMapConfig] = None):
        self.cfg = cfg or OccupancyMapConfig()
        self.grid_size = int(2 * self.cfg.map_size_m / self.cfg.resolution)

        self.occupancy = np.full((self.grid_size, self.grid_size), UNKNOWN, dtype=np.uint8)
        self.checked   = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visit     = np.zeros((self.grid_size, self.grid_size), dtype=np.uint16)

        self._origin_world: Optional[np.ndarray] = None  # (x, z)

    # ── Coordinate transforms ────────────────────────────────────────────

    def set_origin(self, world_pos: np.ndarray):
        """Call once with the agent's starting (x, y, z) position."""
        self._origin_world = np.array([world_pos[0], world_pos[2]], dtype=np.float64)

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """World (x, z) → grid (row, col)."""
        if self._origin_world is None:
            return (self.grid_size // 2, self.grid_size // 2)
        ox, oz = self._origin_world
        col = int((x - ox) / self.cfg.resolution) + self.grid_size // 2
        row = int((z - oz) / self.cfg.resolution) + self.grid_size // 2
        col = np.clip(col, 0, self.grid_size - 1)
        row = np.clip(row, 0, self.grid_size - 1)
        return (int(row), int(col))

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Grid (row, col) → world (x, z)."""
        if self._origin_world is None:
            return (0.0, 0.0)
        ox, oz = self._origin_world
        x = (col - self.grid_size // 2) * self.cfg.resolution + ox
        z = (row - self.grid_size // 2) * self.cfg.resolution + oz
        return (x, z)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    # ── Raycasting update from depth image ───────────────────────────────

    def update_from_depth(
        self,
        depth_image: np.ndarray,
        agent_pos: np.ndarray,
        agent_rot,
        sensor_pos: np.ndarray,
        sensor_rot,
        fov_deg: float,
        resolution_hw: Tuple[int, int],
    ):
        """
        Project depth pixels to world XZ, Bresenham-raycast to mark
        FREE along rays and OCCUPIED at depth discontinuity endpoints.
        """
        if self._origin_world is None:
            self.set_origin(agent_pos)

        H, W = resolution_hw
        focal = W / (2.0 * math.tan(math.radians(fov_deg / 2)))
        step = self.cfg.ray_subsample

        vs = np.arange(0, H, step)
        us = np.arange(0, W, step)
        uu, vv = np.meshgrid(us, vs)
        depths = depth_image[vv, uu].astype(np.float64)

        valid = (depths > self.cfg.min_depth) & (depths < self.cfg.frustum_max_depth)
        if not np.any(valid):
            return

        uu_v = uu[valid].astype(np.float64)
        vv_v = vv[valid].astype(np.float64)
        dd   = depths[valid]

        # Unproject to camera-local 3-D
        cx, cy = W / 2.0, H / 2.0
        cam_x = (uu_v - cx) * dd / focal
        cam_y = (vv_v - cy) * dd / focal
        # Habitat camera: forward = -z
        pts_cam = np.stack([cam_x, cam_y, -dd], axis=-1)

        # Transform to world
        rot_mat = self._quat_to_mat(sensor_rot)
        pts_world = (rot_mat @ pts_cam.T).T + sensor_pos[np.newaxis, :]

        # Height filter
        agent_h = agent_pos[1]
        floor_mask = np.abs(pts_world[:, 1] - agent_h) < self.cfg.floor_height_tol
        obstacle_mask = (
            (pts_world[:, 1] > agent_h - self.cfg.floor_height_tol) &
            (pts_world[:, 1] < agent_h + self.cfg.ceiling_height) &
            ~floor_mask
        )

        agent_rc = self.world_to_grid(agent_pos[0], agent_pos[2])

        # Free rays (floor points)
        for pt in pts_world[floor_mask][::2]:
            end_rc = self.world_to_grid(pt[0], pt[2])
            self._raycast_free(agent_rc, end_rc)

        # Obstacle rays
        inflate = self.cfg.agent_radius_cells
        for pt in pts_world[obstacle_mask][::2]:
            end_rc = self.world_to_grid(pt[0], pt[2])
            self._raycast_free(agent_rc, end_rc, exclude_end=True)
            self._mark_occupied(end_rc, inflate)

        # Agent cell always free + visited
        if self.in_bounds(*agent_rc):
            self.occupancy[agent_rc] = FREE
            self.visit[agent_rc] = min(self.visit[agent_rc] + 1, 65535)

    # ── Semantic coverage (Checked) update ───────────────────────────────

    def update_checked_fov(
        self,
        agent_pos: np.ndarray,
        agent_rot,
        fov_deg: float,
        max_range: float,
        target_detected: bool,
    ):
        """
        If target NOT detected in current frame, project horizontal FOV triangle
        onto the grid and mark those FREE cells as Checked.
        """
        if target_detected or self._origin_world is None:
            return

        fwd = self._quat_forward_xz(agent_rot)
        angle = math.radians(fov_deg / 2) * self.cfg.checked_fov_shrink
        range_m = min(max_range, self.cfg.frustum_max_depth)

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        left  = np.array([fwd[0]*cos_a - fwd[1]*sin_a,
                          fwd[0]*sin_a + fwd[1]*cos_a])
        right = np.array([fwd[0]*cos_a + fwd[1]*sin_a,
                         -fwd[0]*sin_a + fwd[1]*cos_a])

        origin_xz = np.array([agent_pos[0], agent_pos[2]])
        self._fill_triangle_checked(origin_xz,
                                     origin_xz + left * range_m,
                                     origin_xz + right * range_m)

    # ── Query helpers ────────────────────────────────────────────────────

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.occupancy[r, c] == FREE

    def is_checked(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c):
            return True
        return bool(self.checked[r, c])

    def is_unknown(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.occupancy[r, c] == UNKNOWN

    def is_occupied(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c):
            return True
        return self.occupancy[r, c] == OCCUPIED

    def get_exploration_ratio(self) -> float:
        known = np.sum(self.occupancy != UNKNOWN)
        return float(np.sum(self.checked)) / max(float(known), 1.0)

    def get_free_mask(self) -> np.ndarray:
        return self.occupancy == FREE

    def get_frontier_mask(self) -> np.ndarray:
        """FREE cells adjacent to at least one UNKNOWN cell."""
        free = (self.occupancy == FREE)
        unknown = (self.occupancy == UNKNOWN)
        adj_unknown = np.zeros_like(unknown)
        adj_unknown[1:, :]  |= unknown[:-1, :]
        adj_unknown[:-1, :] |= unknown[1:, :]
        adj_unknown[:, 1:]  |= unknown[:, :-1]
        adj_unknown[:, :-1] |= unknown[:, 1:]
        return free & adj_unknown

    def reset(self):
        self.occupancy[:] = UNKNOWN
        self.checked[:] = False
        self.visit[:] = 0
        self._origin_world = None

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _quat_to_mat(q) -> np.ndarray:
        """Quaternion → 3×3 rotation matrix."""
        try:
            import habitat_sim
            coeffs = np.array(habitat_sim.utils.quat_to_coeffs(q))
            x, y, z, w = coeffs
        except Exception:
            if hasattr(q, 'x'):
                x, y, z, w = q.x, q.y, q.z, q.w
            else:
                x, y, z, w = q[0], q[1], q[2], q[3]
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _quat_forward_xz(q) -> np.ndarray:
        """Unit forward direction in XZ plane from quaternion."""
        try:
            import habitat_sim
            fwd_3d = habitat_sim.utils.quat_rotate_vector(q, habitat_sim.geo.FRONT)
        except Exception:
            R = OccupancyMap._quat_to_mat(q)
            fwd_3d = R @ np.array([0, 0, -1])
        xz = np.array([fwd_3d[0], fwd_3d[2]])
        norm = np.linalg.norm(xz)
        return xz / norm if norm > 1e-6 else np.array([0.0, -1.0])

    def _raycast_free(self, start_rc, end_rc, exclude_end=False):
        cells = self._bresenham(*start_rc, *end_rc)
        end_idx = len(cells) - (1 if exclude_end else 0)
        for i in range(end_idx):
            r, c = cells[i]
            if self.in_bounds(r, c) and self.occupancy[r, c] == UNKNOWN:
                self.occupancy[r, c] = FREE

    def _mark_occupied(self, rc, inflate=0):
        r, c = rc
        for dr in range(-inflate, inflate + 1):
            for dc in range(-inflate, inflate + 1):
                rr, cc = r + dr, c + dc
                if self.in_bounds(rr, cc):
                    self.occupancy[rr, cc] = OCCUPIED

    @staticmethod
    def _bresenham(r0, c0, r1, c1) -> List[Tuple[int, int]]:
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        while True:
            cells.append((r0, c0))
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc
        return cells

    def _fill_triangle_checked(self, v0_xz, v1_xz, v2_xz):
        """Rasterise a triangle (world XZ coords) onto the checked grid."""
        pts = [self.world_to_grid(v[0], v[1]) for v in [v0_xz, v1_xz, v2_xz]]
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        r_min = max(0, min(rows))
        r_max = min(self.grid_size - 1, max(rows))
        c_min = max(0, min(cols))
        c_max = min(self.grid_size - 1, max(cols))

        (r0, c0), (r1, c1), (r2, c2) = pts
        denom = (r1 - r2) * (c0 - c2) + (c2 - c1) * (r0 - r2)
        if abs(denom) < 1e-10:
            return

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                w0 = ((r1-r2)*(c-c2) + (c2-c1)*(r-r2)) / denom
                w1 = ((r2-r0)*(c-c2) + (c0-c2)*(r-r2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    if self.in_bounds(r, c) and self.occupancy[r, c] == FREE:
                        self.checked[r, c] = True
