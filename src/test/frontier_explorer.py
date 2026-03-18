"""
Phase 2+3: Frontier Extraction, Geometric Filtering, Semantic Scoring
======================================================================
Pipeline:
  1. Extract frontier cells (FREE adjacent to UNKNOWN)
  2. Cluster via BFS → centroid per cluster
  3. Hard geometric filter (distance, LOS, not-checked)
  4. Score = -α·distance + β·semantic + γ·info_gain
  5. Return top-K candidates
"""

import numpy as np
import math
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque

from occupancy_map import OccupancyMap, FREE, OCCUPIED, UNKNOWN


@dataclass
class FrontierConfig:
    min_dist_m: float = 1.0
    max_dist_m: float = 3.5
    cluster_radius_cells: int = 10
    min_cluster_size: int = 3
    los_step: int = 2
    alpha_distance: float = 0.3
    beta_semantic: float = 0.5
    gamma_info: float = 0.2
    top_k: int = 5


# ── Semantic co-occurrence table ─────────────────────────────────────────
COOCCURRENCE: Dict[str, Dict[str, float]] = {
    'chair': {
        'table': 0.9, 'desk': 0.9, 'dining table': 0.9, 'tv': 0.4,
        'sofa': 0.3, 'couch': 0.3, 'kitchen': 0.5, 'office': 0.8,
    },
    'couch': {
        'tv': 0.8, 'sofa': 1.0, 'pillow': 0.7, 'living room': 0.9,
        'coffee table': 0.8, 'rug': 0.5,
    },
    'bed': {
        'pillow': 0.9, 'bedroom': 1.0, 'nightstand': 0.9, 'lamp': 0.5,
        'dresser': 0.7, 'closet': 0.6, 'blanket': 0.8,
    },
    'toilet': {
        'bathroom': 1.0, 'sink': 0.9, 'bathtub': 0.7, 'mirror': 0.6,
        'shower': 0.7, 'towel': 0.6,
    },
    'tv screen': {
        'sofa': 0.8, 'couch': 0.8, 'living room': 0.9, 'remote': 0.7,
        'entertainment center': 0.8, 'tv stand': 0.9,
    },
    'tv_monitor': {
        'sofa': 0.8, 'couch': 0.8, 'living room': 0.9, 'remote': 0.7,
    },
    'plant': {
        'window': 0.6, 'living room': 0.5, 'pot': 0.8, 'shelf': 0.4,
        'garden': 0.7,
    },
    'sink': {
        'kitchen': 0.9, 'bathroom': 0.8, 'faucet': 0.9, 'counter': 0.7,
    },
    'refrigerator': {
        'kitchen': 1.0, 'oven': 0.8, 'counter': 0.7, 'microwave': 0.6,
    },
}


# ── Language prior (moved from deleted GeneralMemory) ────────────────────
LANGUAGE_PRIOR: Dict[str, List[str]] = {
    'bed':          ['bedroom', 'master bedroom', 'guest room'],
    'chair':        ['dining room', 'living room', 'office', 'bedroom'],
    'couch':        ['living room', 'family room', 'lounge'],
    'sofa':         ['living room', 'family room', 'lounge'],
    'toilet':       ['bathroom', 'restroom', 'washroom'],
    'tv':           ['living room', 'bedroom', 'family room'],
    'tv screen':    ['living room', 'bedroom', 'family room'],
    'tv_monitor':   ['living room', 'bedroom', 'family room'],
    'plant':        ['living room', 'patio', 'garden', 'balcony'],
    'sink':         ['kitchen', 'bathroom'],
    'refrigerator': ['kitchen'],
    'book':         ['study', 'bedroom', 'library', 'office'],
    'table':        ['dining room', 'kitchen', 'living room'],
}


def get_language_prior_hint(target_object: str) -> str:
    """Returns a natural-language hint about where to look for the target."""
    t = target_object.lower()
    for key, scenes in LANGUAGE_PRIOR.items():
        if key in t or t in key:
            return f"A {target_object} is most likely found in: {', '.join(scenes)}."
    return f"No strong prior for {target_object}; explore open doors and hallways."


def get_semantic_score(target: str, context_objects: List[str]) -> float:
    """Score: how likely is the target nearby given detected context objects."""
    target_lower = target.lower().strip()
    table = COOCCURRENCE.get(target_lower, {})
    if not table or not context_objects:
        return 0.0
    scores = []
    for obj in context_objects:
        obj_lower = obj.lower().strip()
        for key, val in table.items():
            if key in obj_lower or obj_lower in key:
                scores.append(val)
    return max(scores) if scores else 0.0


# ── Frontier Explorer ────────────────────────────────────────────────────

class FrontierExplorer:

    def __init__(self, occ_map: OccupancyMap, cfg: Optional[FrontierConfig] = None):
        self.map = occ_map
        self.cfg = cfg or FrontierConfig()

    def extract_waypoints(
        self,
        agent_pos_world: np.ndarray,
        target_category: str = '',
        context_objects: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Full pipeline: frontier → cluster → filter → score → rank.
        Returns list of dicts sorted by score (highest first).
        """
        context_objects = context_objects or []
        agent_rc = self.map.world_to_grid(agent_pos_world[0], agent_pos_world[2])

        # 1. Frontier mask, excluding checked regions
        frontier_mask = self.map.get_frontier_mask() & ~self.map.checked

        # 2. Cluster
        clusters = self._cluster_frontiers(frontier_mask)

        # 3. Filter + score
        candidates = []
        for cluster_cells in clusters:
            if len(cluster_cells) < self.cfg.min_cluster_size:
                continue

            centroid_r = int(np.mean([c[0] for c in cluster_cells]))
            centroid_c = int(np.mean([c[1] for c in cluster_cells]))
            centroid_r, centroid_c = self._snap_to_free(centroid_r, centroid_c)
            if centroid_r is None:
                continue

            world_xz = self.map.grid_to_world(centroid_r, centroid_c)
            dx = world_xz[0] - agent_pos_world[0]
            dz = world_xz[1] - agent_pos_world[2]
            dist = math.sqrt(dx * dx + dz * dz)

            # Hard distance filter
            if dist < self.cfg.min_dist_m or dist > self.cfg.max_dist_m:
                continue

            # Hard LOS filter
            if not self._line_of_sight(agent_rc, (centroid_r, centroid_c)):
                continue

            info_gain = self._information_gain(centroid_r, centroid_c)
            sem_score = get_semantic_score(target_category, context_objects)
            score = (
                -self.cfg.alpha_distance * dist
                + self.cfg.beta_semantic * sem_score
                + self.cfg.gamma_info * info_gain
            )

            candidates.append({
                'world_xz': world_xz,
                'grid_rc': (centroid_r, centroid_c),
                'score': score,
                'dist': dist,
                'info_gain': info_gain,
                'sem_score': sem_score,
                'cluster_size': len(cluster_cells),
            })

        candidates.sort(key=lambda c: c['score'], reverse=True)
        return candidates[:self.cfg.top_k]

    def has_frontiers(self) -> bool:
        return bool(np.any(self.map.get_frontier_mask() & ~self.map.checked))

    # ── Internal ─────────────────────────────────────────────────────────

    def _cluster_frontiers(self, frontier_mask) -> List[List[Tuple[int, int]]]:
        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters = []
        rows, cols = np.where(frontier_mask)
        for idx in range(len(rows)):
            r, c = int(rows[idx]), int(cols[idx])
            if visited[r, c]:
                continue
            cluster = []
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                cluster.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < self.map.grid_size and 0 <= nc < self.map.grid_size
                            and frontier_mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            clusters.append(cluster)
        return clusters

    def _snap_to_free(self, r, c, radius=5):
        if self.map.in_bounds(r, c) and self.map.is_free(r, c):
            return r, c
        best, best_d = None, float('inf')
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if self.map.in_bounds(nr, nc) and self.map.is_free(nr, nc):
                    d = dr * dr + dc * dc
                    if d < best_d:
                        best_d = d
                        best = (nr, nc)
        return best if best else (None, None)

    def _line_of_sight(self, start_rc, end_rc) -> bool:
        cells = OccupancyMap._bresenham(*start_rc, *end_rc)
        for i in range(0, len(cells), self.cfg.los_step):
            r, c = cells[i]
            if self.map.in_bounds(r, c) and self.map.is_occupied(r, c):
                return False
        return True

    def _information_gain(self, r, c, radius=15) -> float:
        r_min = max(0, r - radius)
        r_max = min(self.map.grid_size, r + radius)
        c_min = max(0, c - radius)
        c_max = min(self.map.grid_size, c + radius)
        patch = self.map.occupancy[r_min:r_max, c_min:c_max]
        return float(np.sum(patch == UNKNOWN)) / max(float(patch.size), 1.0)
