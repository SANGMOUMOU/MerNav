"""
WMNav Agent — Refactored with 4-Phase Architecture
====================================================
Class hierarchy preserved:  Agent → VLMNavAgent → WMNavAgent

Phase 1: OccupancyMap    — depth raycasting + checked-FOV
Phase 2: FrontierExplorer — frontier extraction + geometric filter
Phase 3: Semantic scoring — co-occurrence + information gain
Phase 4: LocalPlanner    — A* + pure-pursuit + unstuck escape

VLM role reduced to:
  - Object detection (every N steps)
  - Stopping confirmation
  - Optional tie-breaking between frontier candidates
"""

import logging
import math
import random
import ast
import re
import concurrent.futures

import habitat_sim
import numpy as np
import cv2

from simWrapper import PolarAction
from utils import (
    calculate_focal_length, local_to_global, global_to_local,
    agent_frame_to_image_coords, depth_to_height, unproject_2d,
    put_text_on_image, find_intersections,
    RED, GREEN, WHITE, BLACK, GREY,
)
from api import *

from occupancy_map import OccupancyMap, OccupancyMapConfig, FREE, OCCUPIED, UNKNOWN
from frontier_explorer import (
    FrontierExplorer, FrontierConfig,
    get_semantic_score, get_language_prior_hint,
)
from local_planner import LocalPlanner, LocalPlannerConfig
from Memory_module import MemoryManager


# ═════════════════════════════════════════════════════════════════════════
# Base classes (unchanged)
# ═════════════════════════════════════════════════════════════════════════

class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        raise NotImplementedError

    def get_spend(self):
        return 0

    def reset(self):
        pass


class RandomAgent(Agent):
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)
        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1},
            'logging_data': {},
            'images': {'color_sensor': obs['color_sensor']},
        }
        return agent_action, metadata


# ═════════════════════════════════════════════════════════════════════════
# VLMNavAgent — lightweight base with VLM + navigability preserved
# ═════════════════════════════════════════════════════════════════════════

class VLMNavAgent(Agent):
    """
    Base for VLM-augmented navigation.
    Now delegates action generation to subclasses;
    retains navigability / voxel map infrastructure for backward compat.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (
            cfg['sensor_cfg']['img_height'],
            cfg['sensor_cfg']['img_width'],
        )
        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])
        self.depth_estimator = None
        self.segmentor = None

        # Memory (LongTermMemory only — stuck detection is in LocalPlanner)
        self.memory = MemoryManager(map_size=self.map_size, scale=self.scale)
        self.reset()

    def step(self, obs: dict):
        raise NotImplementedError

    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg.get('turn_around_cooldown', 5)
        self.actionVLM.reset()
        self.memory.reset()

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. "
            "You observe the image and instructions given to you and output a "
            "textual response, which is converted into actions that physically "
            "move you within the environment. You cannot move through closed doors."
        )
        self.actionVLM: VLM = vlm_cls(
            **cfg['model_kwargs'], system_instruction=system_instruction
        )
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _eval_response(self, response: str):
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        for attempt in [
            lambda: ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]),
            lambda: ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]),
            lambda: ast.literal_eval(result[result.index('{'):result.rindex('}') + 1]),
        ]:
            try:
                ev = attempt()
                if isinstance(ev, dict):
                    return ev
            except Exception:
                continue
        logging.error(f'Error parsing response: {response}')
        return {}

    def _global_to_grid(self, position, rotation=None):
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        res = self.voxel_map.shape
        x = int(res[1] // 2 + dx * self.scale)
        y = int(res[0] // 2 + dz * self.scale)
        if rotation is not None:
            coords = np.array([x, y, 1])
            new = np.dot(rotation, coords)
            return (int(new[0]), int(new[1]))
        return (x, y)


# ═════════════════════════════════════════════════════════════════════════
# WMNavAgent — REFACTORED with 4-phase architecture
# ═════════════════════════════════════════════════════════════════════════

class WMNavAgent(VLMNavAgent):
    """
    Refactored navigation agent.

    step() flow:
      1. Update OccupancyMap from depth (raycasting)
      2. Run lightweight object detection
      3. Update checked-FOV coverage
      4. If target seen → steer directly / stop
      5. Else → FrontierExplorer picks waypoint → LocalPlanner executes
      6. Return PolarAction
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        ref = cfg.get('refactored_cfg', {})

        # ── Phase 1: OccupancyMap ────────────────────────────────────────
        occ_cfg = OccupancyMapConfig(
            resolution=ref.get('occ_resolution', 0.05),
            map_size_m=ref.get('occ_map_size_m', 50.0),
            agent_radius_cells=ref.get('occ_agent_radius', 3),
            frustum_max_depth=ref.get('occ_frustum_depth', 5.0),
            floor_height_tol=ref.get('occ_floor_tol', 0.25),
        )
        self.occ_map = OccupancyMap(occ_cfg)

        # ── Phase 2+3: FrontierExplorer ──────────────────────────────────
        fr_cfg = FrontierConfig(
            min_dist_m=ref.get('frontier_min_dist', 1.0),
            max_dist_m=ref.get('frontier_max_dist', 3.5),
            min_cluster_size=ref.get('frontier_min_cluster', 3),
            top_k=ref.get('frontier_top_k', 5),
            alpha_distance=ref.get('alpha_distance', 0.3),
            beta_semantic=ref.get('beta_semantic', 0.5),
            gamma_info=ref.get('gamma_info', 0.2),
        )
        self.frontier = FrontierExplorer(self.occ_map, fr_cfg)

        # ── Phase 4: LocalPlanner ────────────────────────────────────────
        lp_cfg = LocalPlannerConfig(
            lookahead_cells=ref.get('planner_lookahead', 8),
            max_forward_m=ref.get('planner_max_fwd', 0.5),
            max_rotate_rad=ref.get('planner_max_rot', 0.5),
            stuck_window=ref.get('stuck_window', 5),
            stuck_threshold_m=ref.get('stuck_threshold', 0.15),
            escape_rotate_rad=ref.get('escape_rotate', 2.5),
        )
        self.planner = LocalPlanner(self.occ_map, lp_cfg)

        # ── Config ───────────────────────────────────────────────────────
        self._stop_dist = ref.get('stop_dist_threshold', 1.0)
        self._detect_interval = ref.get('detect_interval', 3)
        self._use_vlm_tiebreak = ref.get('use_vlm_tiebreak', True)
        self._vlm_tiebreak_thresh = ref.get('vlm_tiebreak_threshold', 0.1)

        # ── Per-episode state ────────────────────────────────────────────
        self._current_waypoint = None
        self._target_detected_pos = None
        self._context_objects = []
        self._target_category = ''
        self._episode_done = False

    # ── Overrides ────────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        if hasattr(self, 'occ_map'):
            self.occ_map.reset()
        if hasattr(self, 'planner'):
            self.planner.reset()
        self._current_waypoint = None
        self._target_detected_pos = None
        self._context_objects = []
        self._episode_done = False

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. "
            "You observe the image and instructions given to you and output a "
            "textual response, which is converted into actions that physically "
            "move you within the environment. You cannot move through closed doors."
        )
        self.ActionVLM: VLM = vlm_cls(
            **cfg['model_kwargs'], system_instruction=system_instruction
        )
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.GoalVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        # Aliases for base class compatibility
        self.actionVLM = self.ActionVLM
        self.stoppingVLM = self.PlanVLM

    def get_spend(self):
        return (self.ActionVLM.get_spend() + self.PlanVLM.get_spend()
                + self.PredictVLM.get_spend() + self.GoalVLM.get_spend())

    # ── Main step ────────────────────────────────────────────────────────

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        agent_pos = agent_state.position
        agent_rot = agent_state.rotation
        depth_image = obs.get('depth_sensor')
        rgb_image = obs['color_sensor']
        self._target_category = obs.get('goal', '')

        if self.step_ndx == 0:
            self.init_pos = agent_pos
            self.occ_map.set_origin(agent_pos)
            self.memory.long_term.set_init_pos(agent_pos)

        sensor_state = agent_state.sensor_states['color_sensor']

        # ═══ PHASE 1: Update occupancy map ═══════════════════════════════
        if depth_image is not None:
            self.occ_map.update_from_depth(
                depth_image=depth_image,
                agent_pos=agent_pos,
                agent_rot=agent_rot,
                sensor_pos=sensor_state.position,
                sensor_rot=sensor_state.rotation,
                fov_deg=self.fov,
                resolution_hw=self.resolution,
            )

        # ═══ Object detection ════════════════════════════════════════════
        target_detected, detected_objects = self._detect_objects(
            rgb_image, self._target_category
        )
        self._context_objects = detected_objects

        # ═══ PHASE 1b: Update checked FOV ════════════════════════════════
        median_depth = 3.0
        if depth_image is not None:
            valid = depth_image[depth_image > 0.1]
            if len(valid) > 0:
                median_depth = float(np.median(valid))
        self.occ_map.update_checked_fov(
            agent_pos=agent_pos,
            agent_rot=agent_rot,
            fov_deg=self.fov,
            max_range=median_depth,
            target_detected=target_detected,
        )

        # ═══ Memory update ═══════════════════════════════════════════════
        mem_signals = self.memory.process_step(agent_pos=agent_pos)

        # ═══ DECISION: Target detected → direct nav / stop ══════════════
        action_source = 'frontier'

        if target_detected:
            logging.info(f'[Agent] Target "{self._target_category}" detected!')
            target_xz = self._estimate_target_position(
                depth_image, agent_pos, sensor_state
            )
            if target_xz is not None:
                self._target_detected_pos = target_xz
                self.planner.set_target(target_xz, agent_pos)
                self._current_waypoint = target_xz
                self.memory.long_term.record_target_coord(
                    np.array([target_xz[0], agent_pos[1], target_xz[1]])
                )
                action_source = 'target_direct'

        # Also check historical target from memory
        if action_source != 'target_direct' and mem_signals['has_historical_target']:
            nearest = mem_signals['nearest_target_coord']
            if nearest is not None:
                dist = np.linalg.norm(np.array([
                    nearest[0] - agent_pos[0], nearest[2] - agent_pos[2]
                ]))
                if dist > 0.5:
                    logging.info(f'[Agent] Navigating to historical target at {dist:.2f}m')
                    self.planner.set_target((nearest[0], nearest[2]), agent_pos)
                    self._current_waypoint = (nearest[0], nearest[2])
                    action_source = 'historical_target'

        # ═══ PHASE 2+3: Frontier waypoint selection ══════════════════════
        if action_source == 'frontier':
            need_replan = (
                self._current_waypoint is None
                or self.planner.needs_replan(agent_pos)
                or self.planner.is_stuck
            )
            if need_replan:
                candidates = self.frontier.extract_waypoints(
                    agent_pos_world=agent_pos,
                    target_category=self._target_category,
                    context_objects=self._context_objects,
                )
                if candidates:
                    best = self._select_waypoint(candidates, rgb_image, obs)
                    self._current_waypoint = best['world_xz']
                    self.planner.set_target(best['world_xz'], agent_pos)
                    logging.info(
                        f"[Agent] Waypoint: dist={best['dist']:.2f}m, "
                        f"score={best['score']:.3f}, info={best['info_gain']:.2f}"
                    )
                else:
                    if not self.frontier.has_frontiers():
                        logging.info('[Agent] No frontiers remain. Exploration complete.')
                        self._episode_done = True
                    else:
                        # Relax distance and retry
                        old_max = self.frontier.cfg.max_dist_m
                        self.frontier.cfg.max_dist_m *= 1.5
                        candidates = self.frontier.extract_waypoints(
                            agent_pos_world=agent_pos,
                            target_category=self._target_category,
                        )
                        self.frontier.cfg.max_dist_m = old_max
                        if candidates:
                            best = candidates[0]
                            self._current_waypoint = best['world_xz']
                            self.planner.set_target(best['world_xz'], agent_pos)

        # ═══ PHASE 4: Local planner action ═══════════════════════════════
        planner_out = self.planner.get_action(agent_pos, agent_rot)
        forward = planner_out['forward']
        rotate = planner_out['rotate']

        # ═══ Stopping check ══════════════════════════════════════════════
        stopping = False
        if target_detected and self._target_detected_pos is not None:
            dx = agent_pos[0] - self._target_detected_pos[0]
            dz = agent_pos[2] - self._target_detected_pos[1]
            dist_to_target = math.sqrt(dx * dx + dz * dz)
            if dist_to_target < self._stop_dist:
                if self._confirm_stop(rgb_image, self._target_category):
                    logging.info(f'[Agent] Stopping — target within {dist_to_target:.2f}m')
                    stopping = True

        if self.memory.should_terminate_success(agent_pos, self._stop_dist):
            if self._confirm_stop(rgb_image, self._target_category):
                stopping = True

        if stopping or self._episode_done:
            agent_action = PolarAction.stop
        else:
            agent_action = PolarAction(max(forward, 0), rotate)

        # ═══ Build metadata ══════════════════════════════════════════════
        step_metadata = {
            'action_number': -1 if stopping else 1,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': agent_pos,
            'called_stopping': stopping,
            'is_stuck': planner_out['is_stuck'],
            'stuck_count': self.planner.stuck_count,
            'action_source': action_source,
            'exploration_ratio': self.occ_map.get_exploration_ratio(),
            'target_detected': target_detected,
            'object': self._target_category,
        }

        # Preserve original config keys in metadata
        step_metadata.update(self.cfg)

        images = {
            'color_sensor': rgb_image.copy(),
            'occupancy_map': self._render_occupancy(agent_pos, agent_rot),
        }
        if self._current_waypoint:
            self._draw_waypoint_on_image(
                images['color_sensor'], self._current_waypoint,
                agent_pos, agent_state, sensor_state
            )

        logging_data = {
            'ACTION_SOURCE': action_source,
            'FORWARD': f'{forward:.3f}',
            'ROTATE': f'{math.degrees(rotate):.1f}°',
            'CONTEXT_OBJECTS': str(self._context_objects[:10]),
            'EXPLORATION_RATIO': f'{self.occ_map.get_exploration_ratio():.2%}',
            'MEMORY_SIGNALS': str({
                'is_stuck': planner_out['is_stuck'],
                'stuck_count': self.planner.stuck_count,
                'has_historical_target': mem_signals['has_historical_target'],
                'fully_explored': mem_signals['fully_explored'],
            }),
            'LANGUAGE_PRIOR': get_language_prior_hint(self._target_category),
            'WAYPOINT': str(self._current_waypoint),
        }

        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': {},
            'images': images,
            'step': self.step_ndx,
        }

        self.step_ndx += 1
        return agent_action, metadata

    # ── Object detection ─────────────────────────────────────────────────

    def _detect_objects(self, rgb_image, target):
        """Lightweight VLM-based detection, called every N steps."""
        if self.step_ndx % self._detect_interval != 0 and self.step_ndx > 0:
            return False, self._context_objects

        try:
            prompt = (
                f"List the main objects visible in this image as a comma-separated list. "
                f"Also state whether a '{target}' is clearly visible (yes/no). "
                f"Format: objects: <list> | target_visible: <yes/no>"
            )
            response = self.GoalVLM.call([rgb_image], prompt)
            objects = []
            target_visible = False

            if 'objects:' in response.lower():
                obj_part = response.lower().split('objects:')[1]
                if '|' in obj_part:
                    obj_part = obj_part.split('|')[0]
                objects = [o.strip() for o in obj_part.split(',') if o.strip()]

            if 'target_visible:' in response.lower():
                vis_part = response.lower().split('target_visible:')[1].strip()
                target_visible = vis_part.startswith('yes')

            return target_visible, objects
        except Exception as e:
            logging.error(f'[Agent] Detection error: {e}')
            return False, []

    # ── Target position estimation ───────────────────────────────────────

    def _estimate_target_position(self, depth_image, agent_pos, sensor_state):
        """Rough target (x, z) from centre of depth image."""
        if depth_image is None:
            return None
        H, W = depth_image.shape[:2]
        cy, cx = H // 2, W // 2
        d = float(depth_image[cy, cx])
        if d < 0.1 or d > 5.0:
            return None
        cam_x = (cx - W / 2) * d / self.focal_length
        cam_pt = np.array([cam_x, 0, -d])
        R = OccupancyMap._quat_to_mat(sensor_state.rotation)
        world_pt = R @ cam_pt + sensor_state.position
        return (float(world_pt[0]), float(world_pt[2]))

    # ── Waypoint selection ───────────────────────────────────────────────

    def _select_waypoint(self, candidates, rgb_image, obs):
        if len(candidates) <= 1:
            return candidates[0]
        # VLM tie-breaking if top-2 are close
        if (self._use_vlm_tiebreak
            and abs(candidates[0]['score'] - candidates[1]['score'])
                < self._vlm_tiebreak_thresh):
            try:
                dirs = []
                for i, c in enumerate(candidates[:3]):
                    wx, wz = c['world_xz']
                    dx = wx - obs['agent_state'].position[0]
                    dz = wz - obs['agent_state'].position[2]
                    angle = math.degrees(math.atan2(dx, -dz))
                    dirs.append(
                        f"Option {i+1}: {angle:.0f}° at {c['dist']:.1f}m "
                        f"(info={c['info_gain']:.2f})"
                    )
                prompt = (
                    f"I'm searching for a {self._target_category}. "
                    f"Which direction should I explore next?\n"
                    + "\n".join(dirs)
                    + "\nReturn just the option number (1, 2, or 3)."
                )
                resp = self.PlanVLM.call([rgb_image], prompt)
                for i in range(min(3, len(candidates))):
                    if str(i + 1) in resp:
                        return candidates[i]
            except Exception:
                pass
        return candidates[0]

    # ── Stop confirmation ────────────────────────────────────────────────

    def _confirm_stop(self, rgb_image, target):
        try:
            prompt = (
                f"Is a {target} clearly visible and close in this image? "
                f"A {target} must actually be a {target}, not a similar object. "
                f"Reply with only 'yes' or 'no'."
            )
            resp = self.PredictVLM.call([rgb_image], prompt)
            return 'yes' in resp.lower()
        except Exception:
            return True

    # ── Visualisation ────────────────────────────────────────────────────

    def _render_occupancy(self, agent_pos, agent_rot, zoom_m=10):
        vis = np.zeros((*self.occ_map.occupancy.shape, 3), dtype=np.uint8)
        vis[self.occ_map.occupancy == UNKNOWN]  = [40, 40, 40]
        vis[self.occ_map.occupancy == FREE]     = [220, 220, 220]
        vis[self.occ_map.occupancy == OCCUPIED] = [0, 0, 0]
        vis[self.occ_map.checked]               = [180, 220, 255]

        if self.occ_map._origin_world is not None:
            ar, ac = self.occ_map.world_to_grid(agent_pos[0], agent_pos[2])
            cv2.circle(vis, (ac, ar), 5, (0, 0, 255), -1)
            # Draw forward direction
            fwd = OccupancyMap._quat_forward_xz(agent_rot)
            fr, fc = self.occ_map.world_to_grid(
                agent_pos[0] + fwd[0] * 1.0, agent_pos[2] + fwd[1] * 1.0
            )
            cv2.arrowedLine(vis, (ac, ar), (fc, fr), (0, 0, 255), 2, tipLength=0.3)

            if self._current_waypoint:
                wr, wc = self.occ_map.world_to_grid(*self._current_waypoint)
                cv2.circle(vis, (wc, wr), 5, (0, 255, 0), -1)
                cv2.arrowedLine(vis, (ac, ar), (wc, wr), (0, 200, 0), 2)

            cells = int(zoom_m / self.occ_map.cfg.resolution)
            r1, r2 = max(0, ar - cells), min(vis.shape[0], ar + cells)
            c1, c2 = max(0, ac - cells), min(vis.shape[1], ac + cells)
            vis = vis[r1:r2, c1:c2]

        vis = cv2.resize(vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        cv2.putText(vis, f'step {self.step_ndx}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis, f'expl:{self.occ_map.get_exploration_ratio():.0%}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return vis

    def _draw_waypoint_on_image(self, rgb_image, waypoint_xz, agent_pos,
                                 agent_state, sensor_state):
        local = global_to_local(
            agent_state.position, agent_state.rotation,
            np.array([waypoint_xz[0], agent_pos[1], waypoint_xz[1]])
        )
        px = agent_frame_to_image_coords(
            local, agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        if px is not None:
            cv2.circle(rgb_image, tuple(px), 20, (0, 255, 0), 3)
            cv2.putText(rgb_image, 'WP', (px[0] - 12, px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Legacy voxel generation (for env backward compat) ────────────────

    def generate_voxel(self, agent_state=None, zoom=9):
        """Returns the occupancy-map render instead of legacy voxel."""
        if agent_state is None:
            return np.zeros((480, 480, 3), dtype=np.uint8)
        return self._render_occupancy(agent_state.position, agent_state.rotation, zoom)

    def navigability(self, obs, direction_idx):
        """Legacy stub — occupancy map now handles this via depth raycasting."""
        agent_state = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
            self.occ_map.set_origin(agent_state.position)
            self.memory.long_term.set_init_pos(agent_state.position)

        depth_image = obs.get('depth_sensor')
        if depth_image is not None:
            sensor_state = agent_state.sensor_states['color_sensor']
            self.occ_map.update_from_depth(
                depth_image=depth_image,
                agent_pos=agent_state.position,
                agent_rot=agent_state.rotation,
                sensor_pos=sensor_state.position,
                sensor_rot=sensor_state.rotation,
                fov_deg=self.fov,
                resolution_hw=self.resolution,
            )
        self.memory.long_term.mark_visited(agent_state.position)
