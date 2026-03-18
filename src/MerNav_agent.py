"""
MerNav Agent — 优化版

核心优化:
1. 初始化顺序: 用 _initialized 标志替代 hasattr 泛滥
2. cvalue_map: 从 3 通道降为 1 通道，节省 2/3 内存
3. _extract_frontiers: 向量化操作替代嵌套循环
4. MemCell 快照: 用 snapshot() 工厂方法替代手动复制
5. 异常处理: 明确捕获类型 + 降级日志
6. 消除重复: _local_valuation 与 update_curiosity_value 统一
7. reset() 不再执行 I/O，持久化由外部显式调用
8. 目标距离估算: 使用深度图全区域搜索而非仅中心
9. Prompt: 模板化 + 长度控制
10. 学习逻辑: 独立为 _online_learning 方法
"""

import logging
import math
import random
import json
import os
import time
import copy
import habitat_sim
import numpy as np
import cv2
import ast
import re
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

from simWrapper import PolarAction
from utils import *
from api import *


# ============================================================
# 常量
# ============================================================
MEMORY_PERSIST_DIR = os.path.join(
    os.environ.get("LOG_DIR", "/tmp/wmnav_logs"), "memory_db"
)

# 不参与场景学习的通用物体
GENERIC_OBJECTS = frozenset({
    'door', 'window', 'wall', 'floor', 'ceiling', 'light', 'lamp'
})

# 场景学习置信度阈值
SCENE_CONFIDENCE_THRESHOLD = 0.6


# ============================================================
# 三级记忆数据结构
# ============================================================

class MemScene:
    """场景级记忆模板。"""

    __slots__ = ('scene_type', 'object_priors', 'foresight_rules', 'spatial_features')

    def __init__(self, scene_type: str, object_priors: dict,
                 foresight_rules: list = None):
        self.scene_type = scene_type
        self.object_priors = object_priors
        self.foresight_rules = foresight_rules or []
        self.spatial_features = {}

    def to_dict(self) -> dict:
        return {
            'scene_type': self.scene_type,
            'object_priors': self.object_priors,
            'foresight_rules': self.foresight_rules,
            'spatial_features': self.spatial_features
        }

    @staticmethod
    def from_dict(data: dict) -> 'MemScene':
        ms = MemScene(
            scene_type=data['scene_type'],
            object_priors=data['object_priors'],
            foresight_rules=data.get('foresight_rules', [])
        )
        ms.spatial_features = data.get('spatial_features', {})
        return ms


class UniversalSemanticMemory:
    """
    通用语义记忆 (MemScenes) — 跨任务共享的知识库。
    
    优化点:
    - 保存由外部显式调用，不在 learn_* 中自动触发（减少I/O）
    - _build_memscenes 优化为增量构建
    """

    _PERSIST_FILENAME = "universal_semantic_memory.json"
    _DEFAULT_PRIORS_FILENAME = "default_semantic_priors.json"

    def __init__(self, persist_dir: str = None, config_dir: str = None):
        self.persist_dir = persist_dir or MEMORY_PERSIST_DIR
        self._persist_path = os.path.join(self.persist_dir, self._PERSIST_FILENAME)

        if config_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(os.path.dirname(current_dir), 'config')
        self._default_priors_path = os.path.join(config_dir, self._DEFAULT_PRIORS_FILENAME)

        self.SCENE_OBJECT_PRIORS: Dict[str, Dict[str, float]] = {}
        self.SPATIAL_CO_OCCURRENCE: Dict[str, Dict[str, dict]] = {}
        self.SCENE_ADJACENCY: Dict[str, Dict[str, float]] = {}
        self.FORESIGHT_RULES: List[dict] = []
        self.memscenes: Dict[str, MemScene] = {}

        self._update_count = 0
        self._last_save_ts = 0.0
        # Foresight 去重索引
        self._foresight_keys: Set[Tuple[str, str]] = set()

        self._load_and_merge()

    # ---- 持久化 ----

    def _load_and_merge(self):
        default_priors = self._load_default_priors()
        self.SCENE_OBJECT_PRIORS = copy.deepcopy(default_priors.get('scene_object_priors', {}))
        self.SPATIAL_CO_OCCURRENCE = copy.deepcopy(default_priors.get('spatial_co_occurrence', {}))
        self.SCENE_ADJACENCY = copy.deepcopy(default_priors.get('scene_adjacency', {}))
        self.FORESIGHT_RULES = copy.deepcopy(default_priors.get('foresight_rules', []))
        self._foresight_keys = {(r['trigger'], r['prediction']) for r in self.FORESIGHT_RULES}

        disk_data = self._load_from_disk()
        if disk_data:
            self._merge_from_dict(disk_data)
            logging.info(f'[SemanticMemory] Loaded persisted data from {self._persist_path}')

        self._build_memscenes()

    def _load_default_priors(self) -> dict:
        empty = {'scene_object_priors': {}, 'spatial_co_occurrence': {},
                 'scene_adjacency': {}, 'foresight_rules': []}
        if not os.path.isfile(self._default_priors_path):
            logging.warning(f'[SemanticMemory] Default priors not found: {self._default_priors_path}')
            return empty
        try:
            with open(self._default_priors_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f'[SemanticMemory] Failed to load default priors: {e}')
            return empty

    def _load_from_disk(self) -> dict:
        if not os.path.isfile(self._persist_path):
            return {}
        try:
            with open(self._persist_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f'[SemanticMemory] Failed to load {self._persist_path}: {e}')
            return {}

    def save_to_disk(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        data = {
            'scene_object_priors': self.SCENE_OBJECT_PRIORS,
            'spatial_co_occurrence': self.SPATIAL_CO_OCCURRENCE,
            'scene_adjacency': self.SCENE_ADJACENCY,
            'foresight_rules': self.FORESIGHT_RULES,
            'memscenes': {k: v.to_dict() for k, v in self.memscenes.items()},
            'meta': {'update_count': self._update_count, 'last_save_ts': time.time()}
        }
        try:
            with open(self._persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._last_save_ts = time.time()
        except IOError as e:
            logging.error(f'[SemanticMemory] Failed to save: {e}')

    def _merge_from_dict(self, data: dict):
        for scene, objs in data.get('scene_object_priors', {}).items():
            self.SCENE_OBJECT_PRIORS.setdefault(scene, {}).update(objs)

        for obj, co_objs in data.get('spatial_co_occurrence', {}).items():
            self.SPATIAL_CO_OCCURRENCE.setdefault(obj, {}).update(co_objs)

        for scene, adj in data.get('scene_adjacency', {}).items():
            self.SCENE_ADJACENCY.setdefault(scene, {}).update(adj)

        for rule in data.get('foresight_rules', []):
            key = (rule.get('trigger', ''), rule.get('prediction', ''))
            if key not in self._foresight_keys:
                self.FORESIGHT_RULES.append(rule)
                self._foresight_keys.add(key)

    # ---- 在线学习 ----

    def learn_scene_object(self, scene_type: str, object_name: str,
                           observed_prob: float = None, alpha: float = 0.1):
        if scene_type == 'unknown':
            return
        if scene_type not in self.SCENE_OBJECT_PRIORS:
            self.SCENE_OBJECT_PRIORS[scene_type] = {}

        old = self.SCENE_OBJECT_PRIORS[scene_type].get(object_name)
        new = observed_prob if observed_prob is not None else 0.5

        if old is None:
            self.SCENE_OBJECT_PRIORS[scene_type][object_name] = new
        else:
            self.SCENE_OBJECT_PRIORS[scene_type][object_name] = round(
                (1 - alpha) * old + alpha * new, 4
            )
        self._update_count += 1
        self._rebuild_memscene(scene_type)

    def learn_co_occurrence(self, obj_a: str, obj_b: str, distance: float,
                            alpha: float = 0.1):
        bucket = self.SPATIAL_CO_OCCURRENCE.setdefault(obj_a, {})
        if obj_b in bucket:
            old = bucket[obj_b]
            old['prob'] = round((1 - alpha) * old['prob'] + alpha * 1.0, 4)
            old['typical_dist'] = round((1 - alpha) * old['typical_dist'] + alpha * distance, 2)
        else:
            bucket[obj_b] = {'prob': 0.5, 'typical_dist': round(distance, 2)}
        self._update_count += 1

    def learn_scene_adjacency(self, scene_a: str, scene_b: str, alpha: float = 0.1):
        if scene_a == 'unknown' or scene_b == 'unknown':
            return
        bucket = self.SCENE_ADJACENCY.setdefault(scene_a, {})
        old = bucket.get(scene_b, 0.0)
        bucket[scene_b] = round((1 - alpha) * old + alpha * 1.0, 4)
        self._update_count += 1

    def learn_foresight_rule(self, trigger: str, prediction: str, confidence: float):
        key = (trigger, prediction)
        if key not in self._foresight_keys:
            self.FORESIGHT_RULES.append({
                'trigger': trigger, 'prediction': prediction, 'confidence': confidence
            })
            self._foresight_keys.add(key)
            self._update_count += 1
            if trigger.startswith('scene:'):
                self._rebuild_memscene(trigger.split(':')[1])

    def _rebuild_memscene(self, scene_type: str):
        if scene_type not in self.SCENE_OBJECT_PRIORS:
            return
        foresights = [r for r in self.FORESIGHT_RULES if f'scene:{scene_type}' in r['trigger']]
        self.memscenes[scene_type] = MemScene(
            scene_type=scene_type,
            object_priors=self.SCENE_OBJECT_PRIORS[scene_type],
            foresight_rules=foresights
        )

    def _build_memscenes(self):
        for scene_type, obj_priors in self.SCENE_OBJECT_PRIORS.items():
            foresights = [r for r in self.FORESIGHT_RULES if f'scene:{scene_type}' in r['trigger']]
            self.memscenes[scene_type] = MemScene(
                scene_type=scene_type, object_priors=obj_priors, foresight_rules=foresights
            )

    # ---- 查询接口 ----

    def get_likely_scenes(self, goal_object: str, threshold: float = 0.3) -> list:
        scenes = []
        for scene, objects in self.SCENE_OBJECT_PRIORS.items():
            prob = objects.get(goal_object, 0.0)
            if prob >= threshold:
                scenes.append((scene, prob))
        scenes.sort(key=lambda x: x[1], reverse=True)
        return scenes

    def get_co_occurrence_clues(self, goal_object: str) -> list:
        clues = []
        for obj, co_objs in self.SPATIAL_CO_OCCURRENCE.items():
            if goal_object in co_objs:
                clues.append({
                    'clue_object': obj,
                    'prob': co_objs[goal_object]['prob'],
                    'typical_dist': co_objs[goal_object]['typical_dist']
                })
        if goal_object in self.SPATIAL_CO_OCCURRENCE:
            for co_obj, info in self.SPATIAL_CO_OCCURRENCE[goal_object].items():
                clues.append({
                    'clue_object': co_obj,
                    'prob': info['prob'],
                    'typical_dist': info['typical_dist']
                })
        return clues

    def get_adjacent_scene_prob(self, current_scene: str, target_scene: str) -> float:
        return self.SCENE_ADJACENCY.get(current_scene, {}).get(target_scene, 0.05)

    def match_scene_type(self, detected_objects: list) -> Tuple[str, float]:
        best_scene, best_score = 'unknown', 0.0
        n = max(len(detected_objects), 1)
        for scene, obj_probs in self.SCENE_OBJECT_PRIORS.items():
            score = sum(obj_probs.get(obj, 0.0) for obj in detected_objects) / n
            if score > best_score:
                best_score = score
                best_scene = scene
        return best_scene, best_score

    def compute_scene_prior_bonus(self, goal_object: str, adjacent_scene_type: str) -> float:
        return self.SCENE_OBJECT_PRIORS.get(adjacent_scene_type, {}).get(goal_object, 0.0)


class TopologicalSpatialMemory:
    """
    拓扑空间记忆 (LTM)。
    
    优化点:
    - 量化精度参数化 (quantize_factor)
    - update_semantic_map 使用 defaultdict 减少分支
    """

    def __init__(self, map_size: int = 5000, quantize_factor: int = 10):
        self.map_size = map_size
        self.quantize_factor = quantize_factor
        self.global_semantic_map: Dict[Tuple, dict] = {}
        self.frontier_nodes: List[dict] = []
        self.trajectory_history: List[dict] = []
        self.discovered_objects: Dict[str, List[dict]] = {}
        self.visited_scene_regions: Set[Tuple] = set()
        self.last_detected_objects: Set[str] = set()

    def reset(self):
        self.global_semantic_map.clear()
        self.frontier_nodes.clear()
        self.trajectory_history.clear()
        self.discovered_objects.clear()
        self.visited_scene_regions.clear()
        self.last_detected_objects.clear()

    def _quantize(self, grid_coords: tuple) -> tuple:
        q = self.quantize_factor
        return (grid_coords[0] // q, grid_coords[1] // q)

    def add_trajectory_point(self, position: np.ndarray, rotation, step: int):
        self.trajectory_history.append({
            'position': position.copy(), 'rotation': rotation, 'step': step
        })

    def update_semantic_map(self, grid_coords: tuple, objects: list,
                            scene_type: str, step: int):
        key = self._quantize(grid_coords)
        if key not in self.global_semantic_map:
            self.global_semantic_map[key] = {
                'objects': set(), 'scene_type': scene_type,
                'step_discovered': step, 'visit_count': 0
            }
        entry = self.global_semantic_map[key]
        entry['objects'].update(objects)
        entry['visit_count'] += 1
        self.visited_scene_regions.add(key)

    def add_discovered_object(self, obj_name: str, position: np.ndarray,
                              grid_coords: tuple, step: int):
        self.discovered_objects.setdefault(obj_name, []).append({
            'position': position.copy() if isinstance(position, np.ndarray) else position,
            'grid_coords': grid_coords,
            'step': step
        })

    def update_frontiers(self, new_frontiers: list, max_nodes: int = 50):
        # 移除已访问的
        self.frontier_nodes = [
            f for f in self.frontier_nodes
            if self._quantize(f['grid_coords']) not in self.visited_scene_regions
        ]
        existing = {self._quantize(f['grid_coords']) for f in self.frontier_nodes}
        for nf in new_frontiers:
            key = self._quantize(nf['grid_coords'])
            if key not in existing and key not in self.visited_scene_regions:
                self.frontier_nodes.append(nf)
                existing.add(key)
        if len(self.frontier_nodes) > max_nodes:
            self.frontier_nodes.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
            self.frontier_nodes = self.frontier_nodes[:max_nodes]

    def check_new_discovery(self, current_objects: set) -> bool:
        has_new = bool(current_objects - self.last_detected_objects)
        self.last_detected_objects = current_objects.copy()
        return has_new

    def is_new_region(self, grid_coords: tuple) -> bool:
        return self._quantize(grid_coords) not in self.visited_scene_regions

    # ---- 持久化 ----

    def to_dict(self) -> dict:
        serialized_map = {}
        for k, v in self.global_semantic_map.items():
            serialized_map[f"{k[0]},{k[1]}"] = {
                'objects': list(v.get('objects', set())),
                'scene_type': v.get('scene_type', 'unknown'),
                'step_discovered': v.get('step_discovered', 0),
                'visit_count': v.get('visit_count', 0),
            }
        serialized_objects = {}
        for obj_name, locations in self.discovered_objects.items():
            serialized_objects[obj_name] = []
            for loc in locations:
                pos = loc.get('position')
                if isinstance(pos, np.ndarray):
                    pos = pos.tolist()
                serialized_objects[obj_name].append({
                    'position': pos,
                    'grid_coords': list(loc.get('grid_coords', (0, 0))),
                    'step': loc.get('step', 0),
                })
        serialized_traj = []
        for pt in self.trajectory_history:
            pos = pt.get('position')
            if isinstance(pos, np.ndarray):
                pos = pos.tolist()
            rot = pt.get('rotation')
            if hasattr(rot, 'tolist'):
                rot = rot.tolist()
            elif hasattr(rot, 'components'):
                rot = list(rot.components)
            serialized_traj.append({
                'position': pos, 'rotation': rot, 'step': pt.get('step', 0)
            })
        return {
            'map_size': self.map_size,
            'global_semantic_map': serialized_map,
            'discovered_objects': serialized_objects,
            'trajectory_history': serialized_traj,
            'visited_scene_regions': [list(r) for r in self.visited_scene_regions],
        }

    def save_to_disk(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = self.to_dict()
        data['meta'] = {'save_ts': time.time()}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logging.error(f'[TopologicalMemory] Failed to save: {e}')

    @staticmethod
    def load_from_disk(filepath: str) -> 'TopologicalSpatialMemory':
        mem = TopologicalSpatialMemory()
        if not os.path.isfile(filepath):
            return mem
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f'[TopologicalMemory] Failed to load {filepath}: {e}')
            return mem
        mem.map_size = data.get('map_size', 5000)
        for key_str, v in data.get('global_semantic_map', {}).items():
            parts = key_str.split(',')
            key = (int(parts[0]), int(parts[1]))
            mem.global_semantic_map[key] = {
                'objects': set(v.get('objects', [])),
                'scene_type': v.get('scene_type', 'unknown'),
                'step_discovered': v.get('step_discovered', 0),
                'visit_count': v.get('visit_count', 0),
            }
        for obj_name, locations in data.get('discovered_objects', {}).items():
            mem.discovered_objects[obj_name] = []
            for loc in locations:
                pos = loc.get('position')
                if pos is not None:
                    pos = np.array(pos)
                mem.discovered_objects[obj_name].append({
                    'position': pos,
                    'grid_coords': tuple(loc.get('grid_coords', [0, 0])),
                    'step': loc.get('step', 0),
                })
        for pt in data.get('trajectory_history', []):
            pos = pt.get('position')
            if pos is not None:
                pos = np.array(pos)
            mem.trajectory_history.append({
                'position': pos, 'rotation': pt.get('rotation'), 'step': pt.get('step', 0)
            })
        for r in data.get('visited_scene_regions', []):
            mem.visited_scene_regions.add(tuple(r))
        return mem


class MemCell:
    """
    工作记忆单元 (STM)。
    
    优化点:
    - snapshot() 工厂方法替代手动逐字段复制
    - add_foresight 去重内置
    """

    def __init__(self):
        self.current_observation = None
        self.depth_observation = None
        self.agent_pose = None
        self.detected_objects: List[str] = []
        self.episode: str = ''
        self.local_scene_type: str = 'unknown'
        self.scene_confidence: float = 0.0
        self.spatial_data: dict = {
            'position': None, 'grid_coords': None,
            'voxel_updates': [], 'navigable_directions': []
        }
        self.foresight: List[dict] = []
        self._foresight_predictions: Set[str] = set()  # 去重索引
        self.goal_detected: bool = False
        self.goal_distance: Optional[float] = None
        self.step: int = 0
        self.timestamp = None

    def update(self, obs: dict, step: int):
        self.current_observation = obs.get('color_sensor')
        self.depth_observation = obs.get('depth_sensor')
        self.agent_pose = obs.get('agent_state')
        self.step = step
        if self.agent_pose:
            self.spatial_data['position'] = self.agent_pose.position.copy()

    def add_foresight(self, prediction: str, confidence: float,
                      source: str = 'semantic_memory'):
        if prediction in self._foresight_predictions:
            # 已存在，更新置信度
            for fs in self.foresight:
                if isinstance(fs, dict) and fs.get('prediction') == prediction:
                    fs['confidence'] = max(fs['confidence'], confidence)
                    return
            return

        self._foresight_predictions.add(prediction)
        self.foresight.append({
            'prediction': prediction, 'confidence': confidence,
            'source': source, 'step': self.step
        })

    def snapshot(self) -> 'MemCell':
        """创建当前 MemCell 的轻量快照（不复制大型图像数据）。"""
        snap = MemCell()
        # 不复制 current_observation 和 depth_observation（太大）
        snap.agent_pose = self.agent_pose
        snap.detected_objects = self.detected_objects.copy()
        snap.episode = self.episode
        snap.local_scene_type = self.local_scene_type
        snap.scene_confidence = self.scene_confidence
        snap.spatial_data = copy.deepcopy(self.spatial_data)
        snap.foresight = copy.deepcopy(self.foresight)
        snap._foresight_predictions = self._foresight_predictions.copy()
        snap.goal_detected = self.goal_detected
        snap.goal_distance = self.goal_distance
        snap.step = self.step
        snap.timestamp = self.timestamp
        return snap

    def to_dict(self) -> dict:
        return {
            'episode': self.episode,
            'atomic_facts': self.detected_objects,
            'scene_type': self.local_scene_type,
            'scene_confidence': self.scene_confidence,
            'spatial_data': self.spatial_data,
            'foresight': self.foresight,
            'step': self.step,
        }


class StagnationDetector:
    """停滞检测器。"""

    def __init__(self, cfg: dict):
        self.stagnation_threshold = cfg.get('stagnation_threshold', 3)
        self.physical_stuck_threshold = cfg.get('physical_stuck_threshold', 0.1)
        self.physical_stuck_steps = cfg.get('physical_stuck_steps', 2)
        self.no_progress_counter = 0
        self.physical_stuck_counter = 0
        self.prev_position: Optional[np.ndarray] = None

    def reset(self):
        self.no_progress_counter = 0
        self.physical_stuck_counter = 0
        self.prev_position = None

    def check_physical_stuck(self, current_position: np.ndarray) -> bool:
        if self.prev_position is not None:
            displacement = np.linalg.norm(current_position - self.prev_position)
            if displacement < self.physical_stuck_threshold:
                self.physical_stuck_counter += 1
            else:
                self.physical_stuck_counter = 0
        self.prev_position = current_position.copy()
        return self.physical_stuck_counter >= self.physical_stuck_steps

    def check_cognitive_stagnation(self, has_new_objects: bool, is_new_region: bool) -> str:
        if has_new_objects or is_new_region:
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1
        if self.no_progress_counter >= self.stagnation_threshold:
            return "GLOBAL"
        return "LOCAL"

    def get_status(self) -> dict:
        return {
            'no_progress_counter': self.no_progress_counter,
            'physical_stuck_counter': self.physical_stuck_counter,
            'planning_mode': "GLOBAL" if self.no_progress_counter >= self.stagnation_threshold else "LOCAL"
        }


# ============================================================
# Agent 基类
# ============================================================

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
        agent_action = PolarAction(random.uniform(0, 1), random.uniform(-0.2, 0.2))
        return agent_action, {
            'step_metadata': {'success': 1},
            'logging_data': {},
            'images': {'color_sensor': obs['color_sensor']}
        }


# ============================================================
# VLMNavAgent 基类
# ============================================================

class VLMNavAgent(Agent):
    """VLMNav agent 基类。保留原有四大组件。"""

    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (cfg['sensor_cfg']['img_height'], cfg['sensor_cfg']['img_width'])
        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])
        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        agent_action, metadata = self._choose_action(obs)
        metadata['step_metadata'].update(self.cfg)
        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'],
            chosen_action=metadata['step_metadata']['action_number'],
            step=self.step_ndx, goal=obs['goal']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image
        metadata['images']['voxel_map_chosen'] = self._generate_voxel(
            metadata['a_final'], agent_state=agent_state,
            chosen_action=metadata['step_metadata']['action_number'], step=self.step_ndx
        )
        self.step_ndx += 1
        return agent_action, metadata

    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()

    def _construct_prompt(self, **kwargs):
        raise NotImplementedError

    def _choose_action(self, obs):
        raise NotImplementedError

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    # ---- 以下方法保持原实现，略去注释以节省空间 ----

    def _run_threads(self, obs, stopping_images, goal):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)
            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx, goal=obs['goal']
                )
                images['color_sensor'] = new_image
        step_metadata = {
            'action_number': -10, 'success': 1,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs):
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)
        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images

    def _stopping_module(self, stopping_images, goal):
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        if 'done' in dct and int(dct['done']) == 1:
            return True, stopping_response
        return False, stopping_response

    def _navigability(self, obs):
        agent_state = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs['depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None
        navigability_mask = self._get_navigability_mask(rgb_image, depth_image, agent_state, sensor_state)
        sensor_range = np.deg2rad(self.fov / 2) * 1.5
        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(
                start, theta_i, navigability_mask, agent_state, sensor_state, depth_image
            )
            if r_i is not None:
                self._update_voxel(r_i, theta_i, agent_state,
                                   clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
                a_initial.append((r_i, theta_i))
        return a_initial

    def _action_proposer(self, a_initial, agent_state):
        min_angle = self.fov / self.cfg['spacing_ratio']
        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']
        explore = explore_bias > 0

        unique = {}
        for mag, theta in a_initial:
            unique.setdefault(theta, []).append(mag)

        arrowData = []
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        for theta, mags in unique.items():
            mag = min(mags)
            cart = [self.e_i_scaling * mag * np.sin(theta), 0, -self.e_i_scaling * mag * np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (
                sum(np.all(topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color, axis=-1)) +
                sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1))
            )
            arrowData.append([clip_frac * mag, theta, score < 3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = [x for x in arrowData if x[0] > filter_thresh]
        filtered.sort(key=lambda x: x[1])
        if not filtered:
            return []

        if explore:
            f = [x for x in filtered if x[2]]
            if f:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx + 1, len(f)):
                    if f[i][1] - longest_theta > (min_angle * 0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx - 1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle * 0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]
                # 修复原始 bug: theta -> theta_i
                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min(abs(theta_i - t) for t in thetas) > min_angle * explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta_i)

        if not out:
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            for i in range(longest_ndx + 1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]

        if (not out or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and \
                (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final, images, agent_state, goal, candidate_flag=False):
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor'],
            step=self.step_ndx, goal=goal, candidate_flag=candidate_flag
        )
        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']) and not candidate_flag:
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor'],
                step=self.step_ndx, goal=goal
            )
        return a_final_projected

    def _prompting(self, goal, a_final, images, step_metadata):
        action_prompt = self._construct_prompt(goal, 'action', num_actions=len(a_final))
        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])
        response = self.actionVLM.call_chat(prompt_images, action_prompt)
        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response
        return step_metadata, logging_data, response

    def _get_navigability_mask(self, rgb_image, depth_image, agent_state, sensor_state):
        if self.cfg['navigability_mode'] == 'segmentation':
            return self.segmentor.get_navigability_mask(rgb_image)
        thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
        height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
        return abs(height_map - (agent_state.position[1] - 0.04)) < thresh

    def _get_default_arrows(self):
        angle = np.deg2rad(self.fov / 2) * 0.7
        return sorted([
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ], key=lambda x: x[1])

    def _get_radial_distance(self, start_pxl, theta_i, navigability_mask, agent_state, sensor_state, depth_image):
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None
        H, W = navigability_mask.shape
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None
        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        if num_points < 5:
            return None, None
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)
        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i
        last_valid_i = 0
        for i in range(num_points - 4):
            if sum(navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)) <= 2:
                out = (int(x_coords[i]), int(y_coords[i]))
                last_valid_i = i
                break
            last_valid_i = i
        if last_valid_i < 5:
            return 0, theta_i
        if self.cfg['navigability_mode'] == 'segmentation':
            r_i = 0.0794 * np.exp(0.006590 * last_valid_i) + 0.616
        else:
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]],
                resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])
        return r_i, theta_i

    def _can_project(self, r_i, theta_i, agent_state, sensor_state):
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None
        edge = self.cfg['image_edge_threshold']
        if (edge * self.resolution[1] <= end_px[0] <= (1 - edge) * self.resolution[1] and
                edge * self.resolution[0] <= end_px[1] <= (1 - edge) * self.resolution[0]):
            return end_px
        return None

    def _project_onto_image(self, a_final, rgb_image, agent_state, sensor_state,
                            chosen_action=None, step=None, goal='', candidate_flag=False):
        scale_factor = rgb_image.shape[0] / 1080
        font = cv2.FONT_HERSHEY_SIMPLEX
        projected = {}
        if chosen_action == -1:
            put_text_on_image(rgb_image, 'TERMINATING EPISODE', text_color=GREEN,
                              text_size=4 * scale_factor, location='center',
                              text_thickness=math.ceil(3 * scale_factor), highlight=False)
            return projected
        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        for r_i, theta_i in a_final:
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name
                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                cc = (end_px[0], end_px[1])
                cr = max(tw, th) // 2 + math.ceil(15 * scale_factor)
                fill_color = GREEN if (chosen_action is not None and action_name == chosen_action) else WHITE
                cv2.circle(rgb_image, cc, cr, fill_color, -1)
                cv2.circle(rgb_image, cc, cr, RED, math.ceil(2 * scale_factor))
                cv2.putText(rgb_image, text, (cc[0] - tw // 2, cc[1] + th // 2), font, text_size, BLACK, text_thickness)

        if not candidate_flag and ((self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or
                                   self.step_ndx == self.turned or chosen_action == 0):
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (tw, th), _ = cv2.getTextSize('0', font, text_size, text_thickness)
            cc = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            cr = max(tw, th) // 2 + math.ceil(15 * scale_factor)
            fill_color = GREEN if (chosen_action is not None and chosen_action == 0) else WHITE
            cv2.circle(rgb_image, cc, cr, fill_color, -1)
            cv2.circle(rgb_image, cc, cr, RED, math.ceil(2 * scale_factor))
            cv2.putText(rgb_image, '0', (cc[0] - tw // 2, cc[1] + th // 2), font, text_size, BLACK, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND',
                        (cc[0] - tw, cc[1] + th + math.ceil(40 * scale_factor)),
                        font, text_size * 0.75, RED, text_thickness)
        if step is not None:
            cv2.putText(rgb_image, f'step {step}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if goal:
            padding = 20
            ts = 2.5 * scale_factor
            (tw, th), _ = cv2.getTextSize(f"goal:{goal}", font, ts, 2)
            cv2.putText(rgb_image, f"goal:{goal}",
                        (rgb_image.shape[1] - tw - padding, padding + th),
                        font, ts, (255, 0, 0), 2, cv2.LINE_AA)
        return projected

    def _update_voxel(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)
        unclipped = max(r - 0.5, 0)
        local_c = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_c = local_to_global(agent_state.position, agent_state.rotation, local_c)
        point = self._global_to_grid(global_c)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)
        clipped = min(clip_frac * r, clip_dist)
        local_c = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_c = local_to_global(agent_state.position, agent_state.rotation, local_c)
        point = self._global_to_grid(global_c)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position, rotation=None):
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        res = self.voxel_map.shape
        x = int(res[1] // 2 + dx * self.scale)
        y = int(res[0] // 2 + dz * self.scale)
        if rotation is not None:
            coords = np.dot(rotation, np.array([x, y, 1]))
            return (int(coords[0]), int(coords[1]))
        return (x, y)

    def _generate_voxel(self, a_final, zoom=9, agent_state=None, chosen_action=None, step=None):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        delta = abs(agent_coords[0] - self._global_to_grid(right)[0])
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt)
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (tw, th), _ = cv2.getTextSize(text, font, 1.25, 1)
            cc = (act_coords[0], act_coords[1])
            cr = max(tw, th) // 2 + 15
            fill = GREEN if (chosen_action is not None and action == chosen_action) else WHITE
            cv2.circle(topdown_map, cc, cr, fill, -1)
            cv2.circle(topdown_map, cc, cr, RED, 1)
            cv2.putText(topdown_map, text, (cc[0] - tw // 2, cc[1] + th // 2), font, 1.25, RED, 2)

        cv2.circle(topdown_map, agent_coords, 15, RED, -1)
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        zoomed = topdown_map[max(0, y-delta):min(max_y, y+delta), max(0, x-delta):min(max_x, x+delta)]
        if step is not None:
            cv2.putText(zoomed, f'step {step}', (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
        return zoomed

    def _action_number_to_polar(self, action_number, a_final):
        try:
            action_number = int(action_number)
            if 0 < action_number <= len(a_final):
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except (ValueError, TypeError):
            pass
        logging.info(f"Bad action number: {action_number}")
        return PolarAction.default

    def _eval_response(self, response: str) -> dict:
        """解析VLM响应为字典。使用正则预处理后逐级尝试解析。"""
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        # 尝试策略: {{}} → {} → 首{到末}
        strategies = [
            lambda s: s[s.index('{') + 1:s.rindex('}')],
            lambda s: s[s.rindex('{'):s.rindex('}') + 1],
            lambda s: s[s.index('{'):s.rindex('}') + 1],
        ]
        for strategy in strategies:
            try:
                extracted = strategy(result)
                parsed = ast.literal_eval(extracted)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError, IndexError):
                continue
        logging.error(f'Failed to parse VLM response: {response[:200]}...')
        return {}


# ============================================================
# MerNavAgent — 三级记忆系统
# ============================================================

class MerNavAgent(VLMNavAgent):
    """
    优化点:
    1. 用 _initialized 标志替代 hasattr 泛滥
    2. cvalue_map 从 3 通道 → 1 通道
    3. _extract_frontiers 向量化
    4. reset() 不执行 I/O
    5. 学习逻辑独立为 _online_learning
    6. 目标距离估算改进
    """

    def __init__(self, cfg: dict):
        self.memory_cfg = cfg.get('memory_cfg', {})
        self._persist_dir = self.memory_cfg.get('persist_dir') or MEMORY_PERSIST_DIR
        self._auto_update = self.memory_cfg.get('auto_update', True)
        self._episode_id: Optional[str] = None
        self._initialized = False  # 关键标志

        # 初始化三级记忆（在 super().__init__ 之前）
        self.semantic_memory = UniversalSemanticMemory(persist_dir=self._persist_dir)
        self.spatial_memory = TopologicalSpatialMemory(map_size=self.map_size)
        self.working_memory = MemCell()
        self.stm_history: List[MemCell] = []
        self.stm_history_length = self.memory_cfg.get('stm_history_length', 5)
        self.stagnation_detector = StagnationDetector(self.memory_cfg)
        self.enriched_context: dict = {}
        self.current_planning_mode: str = "LOCAL"

        # 初始化 VLM 占位（super 会覆盖）
        self.ActionVLM = None
        self.PlanVLM = None
        self.GoalVLM = None
        self.PredictVLM = None

        super().__init__(cfg)
        self._initialized = True

    def reset(self):
        """
        重置 agent 状态。
        
        优化: 不再在 reset 中执行 I/O。
        持久化应由 save_memories_to_disk() 显式调用。
        """
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        # 优化: 单通道 cvalue_map，节省 2/3 内存
        self.cvalue_map = 10.0 * np.ones((self.map_size, self.map_size), dtype=np.float16)
        self.goal_position = []
        self.goal_mask = None
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']

        # VLM 重置（使用 _initialized 标志而非 hasattr）
        if self._initialized:
            self.ActionVLM.reset()
            self.PlanVLM.reset()
            self.PredictVLM.reset()
            self.GoalVLM.reset()

        # 记忆重置
        self.spatial_memory.reset()
        self.working_memory = MemCell()
        self.stm_history.clear()
        self.stagnation_detector.reset()
        self.enriched_context = {}
        self.current_planning_mode = "LOCAL"
        self._episode_id = None

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.GoalVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    # ================================================================
    # 持久化接口（不再在 reset 中调用）
    # ================================================================

    def set_episode_id(self, episode_id: str):
        self._episode_id = episode_id

    def save_memories_to_disk(self, episode_id: str = None):
        """显式保存所有记忆。由 Env._pre_reset_hook 调用。"""
        if self.semantic_memory._update_count > 0:
            self.semantic_memory.save_to_disk()
        ep_id = episode_id or self._episode_id
        if ep_id and self.spatial_memory.trajectory_history:
            topo_dir = os.path.join(self._persist_dir, 'topological')
            filepath = os.path.join(topo_dir, f'topo_{ep_id}.json')
            self.spatial_memory.save_to_disk(filepath)

    # ================================================================
    # Phase 1: 记忆预加载
    # ================================================================

    def preload_memory(self, goal_object: str):
        likely_scenes = self.semantic_memory.get_likely_scenes(goal_object)
        co_clues = self.semantic_memory.get_co_occurrence_clues(goal_object)
        self.enriched_context = {
            'goal_object': goal_object,
            'likely_scenes': likely_scenes,
            'co_occurrence_clues': co_clues,
            'foresight_rules': self._generate_foresight_text(goal_object, likely_scenes, co_clues),
            'activated_memscenes': [
                {'scene_type': st, 'probability': p, 'memscene': self.semantic_memory.memscenes[st]}
                for st, p in likely_scenes if st in self.semantic_memory.memscenes
            ]
        }
        return self.enriched_context

    def _generate_foresight_text(self, goal: str, likely_scenes: list, co_clues: list) -> str:
        """生成记忆上下文文本（带长度控制）。"""
        parts = []
        if likely_scenes:
            top = ', '.join(f"{s[0]}({s[1]:.0%})" for s in likely_scenes[:3])
            parts.append(f"The {goal} is most commonly found in: {top}.")
        if co_clues:
            cs = [f"{c['clue_object']}(~{c['typical_dist']:.1f}m)" for c in co_clues[:3]]
            parts.append(f"If you see {', '.join(cs)}, the {goal} is likely nearby.")
        for scene, _ in likely_scenes[:2]:
            adj = self.semantic_memory.SCENE_ADJACENCY.get(scene, {})
            if adj:
                top_adj = sorted(adj.items(), key=lambda x: x[1], reverse=True)[:2]
                parts.append(f"Rooms adjacent to {scene}: {', '.join(a[0] for a in top_adj)}.")
        # 长度控制: 限制总长度不超过 500 字符
        result = ' '.join(parts)
        if len(result) > 500:
            result = result[:497] + '...'
        return result

    # ================================================================
    # Phase 2: 感知与 MemCell 生成
    # ================================================================

    def perception_and_memcell(self, obs: dict, detected_objects: list = None):
        agent_state = obs['agent_state']
        self.working_memory.update(obs, self.step_ndx)

        if detected_objects:
            self.working_memory.detected_objects = detected_objects

        grid_coords = self._global_to_grid(agent_state.position)
        self.working_memory.spatial_data['grid_coords'] = grid_coords

        # Episode 描述
        self.working_memory.episode = self._generate_episode_description(detected_objects)

        # 场景匹配
        if detected_objects:
            scene_type, confidence = self.semantic_memory.match_scene_type(detected_objects)
            self.working_memory.local_scene_type = scene_type
            self.working_memory.scene_confidence = confidence
            threshold = self.memory_cfg.get('clustering_threshold', 0.5)
            if confidence > threshold and scene_type in self.semantic_memory.memscenes:
                self._trigger_foresight(self.semantic_memory.memscenes[scene_type], detected_objects)

        # 更新 LTM
        self.spatial_memory.add_trajectory_point(agent_state.position, agent_state.rotation, self.step_ndx)
        self.spatial_memory.update_semantic_map(
            grid_coords, detected_objects or [],
            self.working_memory.local_scene_type, self.step_ndx
        )
        if detected_objects:
            for obj in detected_objects:
                self.spatial_memory.add_discovered_object(obj, agent_state.position, grid_coords, self.step_ndx)

        # STM 快照（使用 snapshot() 方法）
        self.stm_history.append(self.working_memory.snapshot())
        if len(self.stm_history) > self.stm_history_length:
            self.stm_history.pop(0)

        return self.working_memory

    def _generate_episode_description(self, detected_objects: list) -> str:
        if not detected_objects:
            return f"Step {self.step_ndx}: No significant objects detected."
        obj_list = ', '.join(detected_objects[:3])
        scene = self.working_memory.local_scene_type
        if scene != 'unknown':
            return f"Step {self.step_ndx}: In {scene}, detected: {obj_list}"
        return f"Step {self.step_ndx}: Detected: {obj_list}"

    def _trigger_foresight(self, memscene: MemScene, detected_objects: list):
        for rule in memscene.foresight_rules:
            if rule['trigger'].startswith('scene:'):
                self.working_memory.add_foresight(
                    rule['prediction'], rule['confidence'], source='memscene'
                )
        for obj in detected_objects:
            for clue in self.semantic_memory.get_co_occurrence_clues(obj):
                self.working_memory.add_foresight(
                    f"{clue['clue_object']} likely within {clue['typical_dist']:.1f}m",
                    clue['prob'], source='co_occurrence'
                )

    # ================================================================
    # Phase 3: 自省与停滞检测
    # ================================================================

    def review_and_stagnation_check(self, obs: dict, detected_objects: list = None) -> tuple:
        agent_state = obs['agent_state']
        goal_object = obs.get('goal', '')

        # 目标检测
        found_goal = False
        goal_close_enough = False
        if goal_object and detected_objects and goal_object in detected_objects:
            found_goal = True
            estimated_distance = self._estimate_goal_distance(obs)
            goal_close_enough = estimated_distance < 1.5
            self.working_memory.goal_detected = True
            self.working_memory.goal_distance = estimated_distance

        # 物理层
        physical_stuck = self.stagnation_detector.check_physical_stuck(agent_state.position)

        # 认知层
        current_objects = set(detected_objects) if detected_objects else set()
        has_new = self.spatial_memory.check_new_discovery(current_objects)
        grid_coords = self._global_to_grid(agent_state.position)
        is_new = self.spatial_memory.is_new_region(grid_coords)
        planning_mode = self.stagnation_detector.check_cognitive_stagnation(has_new, is_new)
        self.current_planning_mode = planning_mode

        return planning_mode, physical_stuck, found_goal, goal_close_enough

    def _estimate_goal_distance(self, obs: dict) -> float:
        """
        改进版: 在整个深度图中搜索最近的有效深度，
        而非仅看中心区域。
        """
        depth = obs.get('depth_sensor')
        if depth is None:
            return 2.0

        h, w = depth.shape[:2]
        # 使用下半部分（目标物体通常在视野下方）
        lower_half = depth[h // 3:, :]
        valid = lower_half[(lower_half > 0.1) & (lower_half < 10.0)]

        if len(valid) > 0:
            # 使用 25th 百分位数（偏近的估算，因为我们关心最近距离）
            return float(np.clip(np.percentile(valid, 25), 0.3, 5.0))
        return 2.0

    # ================================================================
    # Phase 4: 价值评估与规划（统一版）
    # ================================================================

    def update_curiosity_value(self, explorable_value, reason, goal=None):
        """统一入口: 根据 planning_mode 选择评估策略。"""
        if explorable_value is None or reason is None:
            return np.random.randint(0, 12), ''

        if goal and self.current_planning_mode == "GLOBAL":
            return self._global_valuation(explorable_value, reason, goal)

        # LOCAL 模式 或 无 goal 时使用统一的 cvalue 更新逻辑
        return self._update_cvalue_and_score(explorable_value, reason, goal)

    def _update_cvalue_and_score(self, explorable_value: dict, reason: dict,
                                  goal: Optional[str] = None) -> Tuple[int, str]:
        """
        统一的 cvalue 更新与评分逻辑（消除重复代码）。
        同时处理有/无 goal 的情况。
        """
        scene_prior_scale = self.memory_cfg.get('scene_prior_bonus_scale', 0.3) if goal else 0.0

        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i - 2) * 30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i + 2) * 30)) if i != 11 else '30'

                pano = self.panoramic_mask.get(angle)
                if pano is None or not np.any(pano):
                    continue

                eff = self.effective_mask.get(angle, np.zeros(1, dtype=bool))
                eff_last = self.effective_mask.get(last_angle, np.zeros_like(eff))
                eff_next = self.effective_mask.get(next_angle, np.zeros_like(eff))

                intersection1 = eff_last & eff
                intersection2 = eff & eff_next
                mask_exclusive = eff & ~intersection1 & ~intersection2

                ev = explorable_value.get(angle, 5)
                # 单通道 cvalue_map
                self.cvalue_map[mask_exclusive] = np.minimum(
                    self.cvalue_map[mask_exclusive], ev
                )
                if np.any(intersection2):
                    ev_next = explorable_value.get(next_angle, 5)
                    self.cvalue_map[intersection2] = np.minimum(
                        self.cvalue_map[intersection2], (ev + ev_next) / 2
                    )

            if self.goal_mask is not None:
                self.cvalue_map[self.goal_mask] = 10.0

            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                pano = self.panoramic_mask.get(angle)
                if pano is None or not np.any(pano):
                    base_score = explorable_value.get(angle, 5)
                else:
                    base_score = float(np.mean(self.cvalue_map[pano]))

                # 场景先验加成
                if goal and scene_prior_scale > 0:
                    bonus = self._compute_direction_scene_bonus(i, goal)
                    base_score *= (1 + scene_prior_scale * bonus)

                final_score[i] = base_score

            idx = max(final_score, key=final_score.get)
            final_reason = reason.get(str(int(idx * 30)), '')

        except (KeyError, TypeError, ValueError) as e:
            logging.error(f'[Phase4] Valuation error: {e}')
            idx = np.random.randint(0, 12)
            final_reason = ''

        return idx, final_reason

    def _global_valuation(self, explorable_value: dict, reason: dict, goal: str) -> Tuple[int, str]:
        """全局扩张模式: 结合 frontier 信息。"""
        w1 = self.memory_cfg.get('frontier_exploration_weight', 0.4)
        w2 = self.memory_cfg.get('frontier_semantic_weight', 0.3)
        w3 = self.memory_cfg.get('frontier_distance_weight', 0.2)
        w4 = self.memory_cfg.get('frontier_recency_weight', 0.1)

        try:
            frontier_scores = {}
            agent_pos = self.working_memory.agent_pose.position if self.working_memory.agent_pose else None

            if self.spatial_memory.frontier_nodes and agent_pos is not None:
                for fnode in self.spatial_memory.frontier_nodes:
                    fpos = fnode.get('position')
                    if fpos is None:
                        continue
                    dx = fpos[0] - agent_pos[0]
                    dz = fpos[2] - agent_pos[2]
                    angle_deg = np.degrees(np.arctan2(dx, -dz)) % 360
                    dist = np.sqrt(dx ** 2 + dz ** 2)

                    score = (
                        w1 * fnode.get('exploration_area', 1.0) +
                        w2 * self.semantic_memory.compute_scene_prior_bonus(goal, fnode.get('adjacent_scene', 'unknown')) +
                        w3 / (1.0 + dist) +
                        w4 / (1.0 + self.step_ndx - fnode.get('step_discovered', 0))
                    )
                    dir_idx = int(round(angle_deg / 30)) % 12
                    if dir_idx not in frontier_scores or score > frontier_scores[dir_idx]:
                        frontier_scores[dir_idx] = score

            if frontier_scores:
                combined = {}
                for i in range(12):
                    if i % 2 == 0:
                        continue
                    angle = str(int(i * 30))
                    base = explorable_value.get(angle, 5) / 10.0
                    combined[i] = base + frontier_scores.get(i, 0) * 2.0
                idx = max(combined, key=combined.get)
                return idx, reason.get(str(int(idx * 30)), '') + ' [GLOBAL: frontier-guided]'

            # 回退到局部
            idx, r = self._update_cvalue_and_score(explorable_value, reason, goal)
            return idx, r + ' [GLOBAL: no frontiers, fallback]'

        except Exception as e:
            logging.error(f'[Phase4-GLOBAL] Error: {e}')
            return np.random.randint(0, 12), ''

    def _compute_direction_scene_bonus(self, direction_idx: int, goal: str) -> float:
        current_scene = self.working_memory.local_scene_type
        if current_scene == 'unknown':
            return 0.0
        likely_scenes = self.semantic_memory.get_likely_scenes(goal)
        if not likely_scenes:
            return 0.0
        return sum(
            tp * self.semantic_memory.get_adjacent_scene_prob(current_scene, ts)
            for ts, tp in likely_scenes[:3]
        )

    # ================================================================
    # Phase 5: 执行与动态更新（学习逻辑独立）
    # ================================================================

    def execute_and_update(self, obs: dict, agent_action: PolarAction,
                           detected_objects: list = None):
        agent_state = obs['agent_state']
        grid_coords = self._global_to_grid(agent_state.position)

        # LTM 更新
        self.spatial_memory.update_semantic_map(
            grid_coords, detected_objects or [],
            self.working_memory.local_scene_type, self.step_ndx
        )

        # Frontier 提取（向量化版）
        new_frontiers = self._extract_frontiers_vectorized(agent_state)
        self.spatial_memory.update_frontiers(
            new_frontiers, max_nodes=self.memory_cfg.get('max_frontier_nodes', 50)
        )

        # 在线学习（独立方法）
        if self._auto_update and detected_objects:
            self._online_learning(obs, detected_objects)

    def _online_learning(self, obs: dict, detected_objects: list):
        """
        独立的在线学习方法。
        
        优化: 从 execute_and_update 中分离，职责清晰。
        """
        current_scene = self.working_memory.local_scene_type
        scene_confidence = self.working_memory.scene_confidence

        if current_scene == 'unknown' or scene_confidence < SCENE_CONFIDENCE_THRESHOLD:
            return

        known = self.semantic_memory.SCENE_OBJECT_PRIORS.get(current_scene, {})
        for obj in detected_objects:
            if obj in GENERIC_OBJECTS:
                continue
            if obj not in known:
                self.semantic_memory.learn_scene_object(
                    current_scene, obj, observed_prob=0.3
                )
            else:
                self.semantic_memory.learn_scene_object(
                    current_scene, obj, observed_prob=1.0, alpha=0.05
                )

        # 共现学习
        if len(detected_objects) >= 2:
            est_dist = self._estimate_co_occurrence_distance(obs, detected_objects)
            for i, a in enumerate(detected_objects):
                for b in detected_objects[i + 1:]:
                    if a not in GENERIC_OBJECTS and b not in GENERIC_OBJECTS:
                        self.semantic_memory.learn_co_occurrence(a, b, est_dist)

        # 场景邻接
        if (len(self.stm_history) >= 2 and
                self.stm_history[-1].local_scene_type != 'unknown' and
                self.stm_history[-1].local_scene_type != current_scene):
            self.semantic_memory.learn_scene_adjacency(
                self.stm_history[-1].local_scene_type, current_scene
            )

    def _estimate_co_occurrence_distance(self, obs: dict, detected_objects: list) -> float:
        """估算共现物体间距。"""
        max_dist = self.cfg.get('max_action_dist', 3.0)
        fov_rad = np.deg2rad(self.cfg.get('fov', 79) / 2)
        view_width = 2 * max_dist * np.tan(fov_rad)
        n = len(detected_objects)
        dist = view_width / max(n - 1, 1)
        return round(float(np.clip(dist, 0.5, 5.0)), 1)

    def _extract_frontiers_vectorized(self, agent_state) -> list:
        """
        向量化版 frontier 提取。
        
        优化: 用 NumPy 切片替代嵌套 for 循环。
        """
        agent_coords = self._global_to_grid(agent_state.position)
        search_radius = int(self.cfg.get('max_action_dist', 3) * self.scale * 2)
        x_c, y_c = agent_coords
        step = 20  # 采样步长

        # 计算搜索范围
        x_min = max(0, x_c - search_radius)
        x_max = min(self.map_size, x_c + search_radius + 1)
        y_min = max(0, y_c - search_radius)
        y_max = min(self.map_size, y_c + search_radius + 1)

        # 采样点
        xs = np.arange(x_min, x_max, step)
        ys = np.arange(y_min, y_max, step)
        if len(xs) == 0 or len(ys) == 0:
            return []

        # 向量化检查: voxel 有颜色但 explored_map 未标记
        # 构建采样网格
        gx, gy = np.meshgrid(xs, ys)
        gx_flat = gx.ravel()
        gy_flat = gy.ravel()

        is_visible = np.any(self.voxel_map[gy_flat, gx_flat] != 0, axis=1)
        is_explored = np.all(self.explored_map[gy_flat, gx_flat] == self.explored_color, axis=1)
        frontier_mask = is_visible & ~is_explored

        frontier_indices = np.where(frontier_mask)[0]
        if len(frontier_indices) == 0:
            return []

        # 限制 frontier 数量
        max_new = 20
        if len(frontier_indices) > max_new:
            frontier_indices = np.random.choice(frontier_indices, max_new, replace=False)

        frontiers = []
        inv_scale = 1.0 / self.scale
        half_map = self.map_size // 2
        for idx in frontier_indices:
            gx_i, gy_i = int(gx_flat[idx]), int(gy_flat[idx])
            global_x = self.init_pos[0] + (gx_i - half_map) * inv_scale
            global_z = self.init_pos[2] + (gy_i - half_map) * inv_scale
            frontiers.append({
                'position': np.array([global_x, agent_state.position[1], global_z]),
                'grid_coords': (gx_i, gy_i),
                'adjacent_scene': self.working_memory.local_scene_type,
                'priority_score': 0.5,
                'step_discovered': self.step_ndx,
                'exploration_area': 1.0
            })
        return frontiers

    # ================================================================
    # 保留原有方法
    # ================================================================

    def _update_panoramic_voxel(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)
        clipped = min(clip_frac * r, clip_dist)
        local_c = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_c = local_to_global(agent_state.position, agent_state.rotation, local_c)
        point = self._global_to_grid(global_c)
        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)

    def _stopping_module(self, obs, threshold_dist=0.8):
        if self.goal_position:
            avg = np.mean(np.array(self.goal_position), axis=0)
            agent_pos = obs['agent_state'].position
            dist = np.linalg.norm(
                np.array([agent_pos[0], agent_pos[2]]) - np.array([avg[0], avg[2]])
            )
            return dist < threshold_dist
        return False

    def _run_threads(self, obs, stopping_images, goal):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, obs)
            a_final, images, a_goal, candidate_images = preprocessing_thread.result()
            called_stop = stopping_thread.result()
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx, goal=obs['goal']
                )
                images['color_sensor'] = new_image
        step_metadata = {
            'action_number': -10, 'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, a_goal, candidate_images

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords,
                              angle_text='', arrow_length=1):
        arrow_end = np.array([
            position[0] + direction_vector[0] * arrow_length,
            position[1],
            position[2] + direction_vector[2] * arrow_length
        ])
        arrow_end_coords = self._global_to_grid(arrow_end)
        cv2.arrowedLine(roomtrack_map, tuple(coords), tuple(arrow_end_coords), WHITE, 4, tipLength=0.1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(angle_text, font, 1, 2)
        text_end = self._global_to_grid(np.array([
            position[0] + direction_vector[0] * arrow_length * 1.5,
            position[1],
            position[2] + direction_vector[2] * arrow_length * 1.5
        ]))
        cv2.putText(roomtrack_map, angle_text,
                     (text_end[0] - tw // 2, text_end[1] + th // 2),
                     font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def generate_voxel(self, agent_state=None, zoom=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        delta = abs(agent_coords[0] - self._global_to_grid(right)[0])
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)
        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords, "0")

        y_axis = np.array([0, 1, 0])
        quat_30 = habitat_sim.utils.quat_from_angle_axis(-np.pi / 6, y_axis)
        quat_60 = habitat_sim.utils.quat_from_angle_axis(-np.pi / 3, y_axis)

        dir_30 = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, dir_30, agent_state.position, agent_coords, "30")

        current_dir = dir_30.copy()
        for i in range(5):
            current_dir = habitat_sim.utils.quat_rotate_vector(quat_60, current_dir)
            self._draw_direction_arrow(topdown_map, current_dir, agent_state.position,
                                       agent_coords, str((i + 1) * 60 + 30))

        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(self.step_ndx)
        (tw, th), _ = cv2.getTextSize(text, font, 1.25, 1)
        cr = max(tw, th) // 2 + 15
        cv2.circle(topdown_map, agent_coords, cr, WHITE, -1)
        cv2.circle(topdown_map, agent_coords, cr, RED, 1)
        cv2.putText(topdown_map, text, (agent_coords[0] - tw // 2, agent_coords[1] + th // 2),
                     font, 1.25, RED, 2)

        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        zoomed = topdown_map[max(0, y-delta):min(max_y, y+delta), max(0, x-delta):min(max_x, x+delta)]
        cv2.putText(zoomed, f'step {self.step_ndx}', (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
        return zoomed

    def update_voxel(self, r, theta, agent_state, temp_map, effective_dist=3):
        agent_coords = self._global_to_grid(agent_state.position)
        unclipped = max(r, 0)
        local_c = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_c = local_to_global(agent_state.position, agent_state.rotation, local_c)
        point = self._global_to_grid(global_c)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)
        cv2.line(temp_map, agent_coords, point, WHITE, 40)
        unclipped = min(r, effective_dist)
        local_c = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_c = local_to_global(agent_state.position, agent_state.rotation, local_c)
        point = self._global_to_grid(global_c)
        cv2.line(temp_map, agent_coords, point, GREEN, 40)

    def _goal_proposer(self, a_initial, agent_state):
        min_angle = self.fov / self.cfg['spacing_ratio']
        unique = {}
        for mag, theta in a_initial:
            unique.setdefault(theta, []).append(mag)
        arrowData = sorted([[min(mags), theta] for theta, mags in unique.items()], key=lambda x: x[1])
        f = [x for x in arrowData if x[0] > 0]
        if not f:
            return []
        f.sort(key=lambda x: x[1])
        longest = max(f, key=lambda x: x[0])
        longest_ndx = f.index(longest)
        out = [[longest[0], longest[1]]]
        thetas = {longest[1]}
        lt, st = longest[1], longest[1]
        for i in range(longest_ndx + 1, len(f)):
            if f[i][1] - lt > min_angle * 0.45:
                out.append([f[i][0], f[i][1]])
                thetas.add(f[i][1])
                lt = f[i][1]
        for i in range(longest_ndx - 1, -1, -1):
            if st - f[i][1] > min_angle * 0.45:
                out.append([f[i][0], f[i][1]])
                thetas.add(f[i][1])
                st = f[i][1]
        out.sort(key=lambda x: x[1])
        return [(m, t) for m, t in out]

    def _preprocessing_module(self, obs):
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        candidate_images = {'color_sensor': obs['color_sensor'].copy()}
        a_goal_projected = None
        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)
            if obs.get('goal_flag', False):
                a_goal = self._goal_proposer(a_initial, agent_state)
        a_final_projected = self._projection(a_final, images, agent_state, obs.get('goal', ''))
        if obs.get('goal_flag', False):
            a_goal_projected = self._projection(a_goal, candidate_images, agent_state,
                                                 obs.get('goal', ''), candidate_flag=True)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images, a_goal_projected, candidate_images

    def navigability(self, obs, direction_idx):
        agent_state = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs['depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None
        navigability_mask = self._get_navigability_mask(rgb_image, depth_image, agent_state, sensor_state)
        sensor_range = np.deg2rad(self.fov / 2) * 1.5
        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(
                start, theta_i, navigability_mask, agent_state, sensor_state, depth_image
            )
            if r_i is not None:
                self.update_voxel(r_i, theta_i, agent_state, temp_map)
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)
        clipped = min(r, clip_dist)
        local_c = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_c = local_to_global(agent_state.position, agent_state.rotation, local_c)
        point = self._global_to_grid(global_c)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs):
        agent_state = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs['depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None
        navigability_mask = self._get_navigability_mask(rgb_image, depth_image, agent_state, sensor_state)
        sensor_range = np.deg2rad(self.fov / 2) * 1.5
        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )
        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(
                start, theta_i, navigability_mask, agent_state, sensor_state, depth_image
            )
            if r_i is not None:
                self._update_voxel(r_i, theta_i, agent_state,
                                   clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
                a_initial.append((r_i, theta_i))
        if self.cfg.get('panoramic_padding', False) and a_initial:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                                         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return (self.ActionVLM.get_spend() + self.PlanVLM.get_spend() +
                self.PredictVLM.get_spend() + self.GoalVLM.get_spend())

    def _prompting(self, goal, a_final, images, step_metadata, subtask):
        action_prompt = self._construct_prompt(goal, 'action', subtask, num_actions=len(a_final))
        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])
        response = self.ActionVLM.call_chat(prompt_images, action_prompt)
        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response
        return step_metadata, logging_data, response

    def _goal_module(self, goal_image, a_goal, goal):
        location_prompt = self._construct_prompt(goal, 'goal', num_actions=len(a_goal))
        location_response = self.GoalVLM.call([goal_image], location_prompt)
        dct = self._eval_response(location_response)
        try:
            number = int(dct['Number'])
        except (KeyError, TypeError, ValueError):
            number = None
        return number, location_response

    def _get_goal_position(self, action_goal, idx, agent_state):
        r, theta = None, None
        for key, value in action_goal.items():
            if value == idx:
                r, theta = key
                break
        if r is None:
            return None, None
        local_goal = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        agent_coords = self._global_to_grid(agent_state.position)
        point = self._global_to_grid(global_goal)
        local_r = np.array([0, 0, -1])
        global_r = local_to_global(agent_state.position, agent_state.rotation, local_r)
        radius_point = self._global_to_grid(global_r)
        td_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))
        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, td_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)
        return global_goal, goal_mask

    def _choose_action(self, obs):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(
            obs, [obs['color_sensor']], goal
        )

        goal_number = None
        location_response = None
        goal_image = candidate_images['color_sensor'].copy()
        if a_goal is not None:
            goal_number, location_response = self._goal_module(goal_image, a_goal, goal)
            images['goal_image'] = goal_image
            if goal_number is not None and goal_number != 0:
                result = self._get_goal_position(a_goal, goal_number, agent_state)
                if result[0] is not None:
                    goal_position, self.goal_mask = result
                    self.goal_position.append(goal_position)

        step_metadata['object'] = goal

        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if a_goal is not None and goal_number is not None and goal_number != 0:
                logging_data = {'ACTION_NUMBER': int(goal_number)}
                step_metadata['action_number'] = goal_number
                a_final = a_goal
            else:
                step_metadata, logging_data, _ = self._prompting(
                    goal, a_final, images, step_metadata, obs.get('subtask', '{}')
                )
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        # Phase 5
        self.execute_and_update(obs, agent_action, self.working_memory.detected_objects)

        if a_goal is not None and location_response is not None:
            logging_data['LOCATOR_RESPONSE'] = location_response

        logging_data['MEMORY_STATUS'] = str({
            'planning_mode': self.current_planning_mode,
            'stagnation': self.stagnation_detector.get_status(),
            'scene_type': self.working_memory.local_scene_type,
            'frontier_count': len(self.spatial_memory.frontier_nodes)
        })

        return agent_action, {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[:2]
        except (IndexError, AttributeError):
            height, width = 480, 640
        bg = np.zeros((2 * height + 30, 3 * width + 40, 3), np.uint8)
        arr = np.array(images, dtype=np.uint8)
        for i in range(len(arr)):
            if i % 2 == 0:
                continue
            cv2.putText(arr[i], f"Angle {angles[i]}", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            y0 = 10 * (row + 1) + row * height
            x0 = 10 * (col + 1) + col * width
            bg[y0:y0 + height, x0:x0 + width, :] = arr[i]
        return bg

    def make_curiosity_value(self, pano_images, goal):
        angles = np.arange(len(pano_images)) * 30
        inference_image = self._concat_panoramic(pano_images, angles)
        response = self._predicting_module(inference_image, goal)
        explorable_value, reason = {}, {}
        try:
            for angle, values in response.items():
                if isinstance(values, dict):
                    explorable_value[angle] = values.get('Score', 5)
                    reason[angle] = values.get('Explanation', '')
        except (AttributeError, TypeError) as e:
            logging.error(f'[CuriosityValue] Parse error: {e}')
            return inference_image, None, None
        return inference_image, explorable_value or None, reason or None

    def draw_cvalue_map(self, agent_state=None, zoom=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        delta = abs(agent_coords[0] - self._global_to_grid(right)[0])
        # 单通道 → 3通道用于显示
        cvalue_vis = (self.cvalue_map / 10 * 255).astype(np.uint8)
        cvalue_vis = np.stack([cvalue_vis] * 3, axis=-1)
        x, y = agent_coords
        max_x, max_y = cvalue_vis.shape[1], cvalue_vis.shape[0]
        zoomed = cvalue_vis[max(0, y-delta):min(max_y, y+delta), max(0, x-delta):min(max_x, x+delta)]
        cv2.putText(zoomed, f'step {self.step_ndx}', (30, 90),
                     cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
        return zoomed

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)
        try:
            return response['Flag'], response['Subtask']
        except (KeyError, TypeError):
            logging.warning(f'Planning parse failed: {response}')
            return False, '{}'

    def _planning_module(self, planning_image, previous_subtask, goal_reason, goal):
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        return self._eval_response(planning_response)

    def _predicting_module(self, evaluator_image, goal):
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        return self._eval_response(evaluator_response)

    # ================================================================
    # Prompt 构建（模板化 + 长度控制）
    # ================================================================

    # 共享的物体识别注意事项
    _OBJECT_NOTES = (
        "Note a chair must have a backrest and a chair is not a stool. "
        "Note a chair is NOT sofa(couch) which is NOT a bed. "
    )

    def _build_memory_context(self, goal: str) -> str:
        """构建记忆上下文（带长度控制）。"""
        parts = []
        foresight = self.enriched_context.get('foresight_rules', '')
        if foresight:
            parts.append(f"[SPATIAL KNOWLEDGE] {foresight}")
        if self.spatial_memory.discovered_objects:
            objs = list(self.spatial_memory.discovered_objects.keys())[:5]
            parts.append(f"[EXPLORED] Found: {', '.join(objs)}.")
        if self.working_memory.local_scene_type != 'unknown':
            parts.append(f"[SCENE] Currently in: {self.working_memory.local_scene_type}.")
        if self.current_planning_mode == "GLOBAL":
            parts.append("[STRATEGY] Prioritize unexplored areas.")

        context = ' '.join(parts)
        # 长度控制
        max_len = self.memory_cfg.get('max_context_length', 600)
        if len(context) > max_len:
            context = context[:max_len - 3] + '...'
        return (context + ' ') if context else ''

    def _construct_prompt(self, goal, prompt_type, subtask='{}', reason='{}', num_actions=0):
        ctx = self._build_memory_context(goal)

        if prompt_type == 'goal':
            return (
                f"{ctx}"
                f"The agent has been tasked with navigating to a {goal.upper()}. "
                f"There are {num_actions} red arrows representing potential positions, labeled with numbers. "
                f"First, determine if the {goal} is actually in the image. Return 0 if unsure. "
                f"{self._OBJECT_NOTES}"
                f"If found, return the number closest to the {goal}. "
                "Format: {{'Number': <number>}}"
            )

        if prompt_type == 'predicting':
            return (
                f"{ctx}"
                f"The agent is navigating to a {goal.upper()}. Panoramic images show 6 directions "
                f"(30, 90, 150, 210, 270, 330 degrees). "
                f"Score each direction 0-10: 0=dead end, 10={goal} found. "
                f"{self._OBJECT_NOTES}"
                f"Consider open doors, hallways, room types. Cannot go through CLOSED doors. "
                "Also list detected objects. "
                "Format: {{'DetectedObjects': [...], '30': {{'Score': N, 'Explanation': '...'}}, ...}}"
            )

        if prompt_type == 'planning':
            goal_hint = ""
            if self.working_memory.goal_detected:
                d = self.working_memory.goal_distance
                if d is not None and d < 1.5:
                    goal_hint = f"🎯 CRITICAL: {goal} detected at ~{d:.1f}m! Move directly toward it. "
                elif d is not None:
                    goal_hint = f"🎯 {goal} detected at ~{d:.1f}m. Approach it. "

            stag_hint = ""
            if self.current_planning_mode == "GLOBAL":
                stag_hint = (
                    f"No progress for several steps. {len(self.spatial_memory.frontier_nodes)} "
                    f"unexplored frontiers remain. Try a new direction. "
                )

            base = (
                f"{ctx}{goal_hint}{stag_hint}"
                f"Task: navigate to {goal.upper()}. "
                f"{self._OBJECT_NOTES}"
            )

            if reason != '' and subtask != '{}':
                return (
                    f"{base}"
                    f"Direction reason: {reason}. Previous subtask: {subtask}. "
                    f"If {goal} is visible, go to it. If previous subtask incomplete, continue it. "
                    f"Otherwise, choose a new subtask toward the {goal}. "
                    "Format: {{'Subtask': '...', 'Flag': True/False}}"
                )
            return (
                f"{base}"
                f"Describe where to go next to find the {goal}. "
                f"Pay attention to open doors, hallways, stairs. "
                "Format: {{'Subtask': '...', 'Flag': True/False}}"
            )

        if prompt_type == 'action':
            turn_note = (
                "Choose action 0 to TURN AROUND if no good actions. "
                if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''
            )
            if subtask != '{}':
                return (
                    f"{ctx}"
                    f"TASK: {subtask}. Final goal: reach the nearest {goal.upper()}. "
                    f"{num_actions - 1} arrows show possible moves. {turn_note}"
                    f"Which action best achieves the subtask? "
                    "Return: {{'action': <number>}}. Cannot go through closed doors."
                )
            return (
                f"{ctx}"
                f"TASK: Navigate to nearest {goal.upper()}. "
                f"{num_actions - 1} arrows show possible moves. {turn_note}"
                f"Describe what you see, your leads, direction, and best action. "
                "Return: {{'action': <number>}}. Cannot go through closed doors."
            )

        raise ValueError(f'Unknown prompt type: {prompt_type}')