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
from collections import defaultdict

from simWrapper import PolarAction
from utils import *
from api import *


# ============================================================
# 记忆持久化默认路径
# ============================================================
MEMORY_PERSIST_DIR = os.path.join(
    os.environ.get("LOG_DIR", "/tmp/wmnav_logs"), "memory_db"
)


# ============================================================
# 三级记忆数据结构 (Tri-Level Memory Data Structures)
# 参考 EverMemOS 架构设计
# ============================================================

class MemScene:
    """
    MemScene — 单个场景的记忆模板,类似 EverMemOS 的 MemCell 但用于场景级别。
    包含场景的先验知识、预测性规则(Foresight)和空间特征。
    """
    def __init__(self, scene_type: str, object_priors: dict, foresight_rules: list = None):
        self.scene_type = scene_type
        self.object_priors = object_priors  # P(object | scene)
        self.foresight_rules = foresight_rules or []  # 预测性规则列表
        self.spatial_features = {}  # 空间特征（如典型布局）
        
    def to_dict(self):
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
    通用语义记忆 (MemScenes) — 跨任务共享的知识库（持久化版本）。

    参考 EverMemOS 架构：
    - 类似 BaseMemory 的层次结构，支持磁盘持久化（JSON）
    - 包含 Foresight 预测性知识
    - 支持聚类匹配（cluster_manager）
    - **自动学习**：遇到新数据时自动补充到数据库

    持久化策略（参考 EverMemOS 的 write-through）：
    - 初始化时从磁盘加载已有数据，与默认先验合并
    - 每次 update 后立刻写回磁盘（write-through）
    - 使用 JSON 文件代替 MongoDB（轻量化）
    """

    # ---- 持久化文件名 ----
    _PERSIST_FILENAME = "universal_semantic_memory.json"
    _DEFAULT_PRIORS_FILENAME = "default_semantic_priors.json"
    
    def __init__(self, persist_dir: str = None, config_dir: str = None):
        """
        初始化通用语义记忆（参考 EverMemOS write-through 模式）。

        Args:
            persist_dir: 持久化目录。为 None 时使用 MEMORY_PERSIST_DIR 默认值。
            config_dir: 配置文件目录。为 None 时自动查找 config/ 目录。
        """
        self.persist_dir = persist_dir or MEMORY_PERSIST_DIR
        self._persist_path = os.path.join(self.persist_dir, self._PERSIST_FILENAME)
        
        # 配置文件路径（默认先验JSON）
        if config_dir is None:
            # 自动查找 config 目录（假设在 src 的上级目录）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(os.path.dirname(current_dir), 'config')
        self._default_priors_path = os.path.join(config_dir, self._DEFAULT_PRIORS_FILENAME)

        # ---- 运行时可变数据（从配置文件 + 磁盘加载合并） ----
        self.SCENE_OBJECT_PRIORS = {}
        self.SPATIAL_CO_OCCURRENCE = {}
        self.SCENE_ADJACENCY = {}
        self.FORESIGHT_RULES = []
        self.memscenes = {}              # scene_type → MemScene

        # 统计信息
        self._update_count = 0           # 本次运行期间的更新次数
        self._last_save_ts = 0.0

        self._load_and_merge()           # 加载配置 → 加载磁盘 → 合并 → 构建 MemScene
        
    # ================================================================
    #  持久化: 加载 / 保存 / 合并
    # ================================================================

    def _load_and_merge(self):
        """从配置文件加载默认先验，再从磁盘加载学习数据，合并后构建 MemScene。"""
        # Step 1: 从配置文件加载默认先验
        default_priors = self._load_default_priors()
        
        self.SCENE_OBJECT_PRIORS = copy.deepcopy(default_priors.get('scene_object_priors', {}))
        self.SPATIAL_CO_OCCURRENCE = copy.deepcopy(default_priors.get('spatial_co_occurrence', {}))
        self.SCENE_ADJACENCY = copy.deepcopy(default_priors.get('scene_adjacency', {}))
        self.FORESIGHT_RULES = copy.deepcopy(default_priors.get('foresight_rules', []))
        
        # Step 2: 从磁盘加载学习到的数据并合并
        disk_data = self._load_from_disk()
        if disk_data:
            self._merge_from_dict(disk_data)
            logging.info(f'[SemanticMemory] Loaded persisted data from {self._persist_path}')
        else:
            logging.info('[SemanticMemory] No persisted data found, using defaults only')

        self._build_memscenes()

    def _load_default_priors(self) -> dict:
        """从 JSON 配置文件加载默认先验数据。"""
        if not os.path.isfile(self._default_priors_path):
            logging.warning(f'[SemanticMemory] Default priors file not found: {self._default_priors_path}')
            logging.warning('[SemanticMemory] Using empty defaults (no prior knowledge)')
            return {
                'scene_object_priors': {},
                'spatial_co_occurrence': {},
                'scene_adjacency': {},
                'foresight_rules': []
            }
        try:
            with open(self._default_priors_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f'[SemanticMemory] Loaded default priors from {self._default_priors_path}')
            return data
        except Exception as e:
            logging.error(f'[SemanticMemory] Failed to load default priors: {e}')
            return {
                'scene_object_priors': {},
                'spatial_co_occurrence': {},
                'scene_adjacency': {},
                'foresight_rules': []
            }

    def _load_from_disk(self) -> dict:
        """从 JSON 文件加载持久化数据。"""
        if not os.path.isfile(self._persist_path):
            return {}
        try:
            with open(self._persist_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f'[SemanticMemory] Failed to load {self._persist_path}: {e}')
            return {}

    def save_to_disk(self):
        """
        将当前完整状态写入磁盘（write-through）。
        参考 EverMemOS 的即时写入策略。
        """
        os.makedirs(self.persist_dir, exist_ok=True)
        data = {
            'scene_object_priors': self.SCENE_OBJECT_PRIORS,
            'spatial_co_occurrence': self.SPATIAL_CO_OCCURRENCE,
            'scene_adjacency': self.SCENE_ADJACENCY,
            'foresight_rules': self.FORESIGHT_RULES,
            'memscenes': {k: v.to_dict() for k, v in self.memscenes.items()},
            'meta': {
                'update_count': self._update_count,
                'last_save_ts': time.time(),
            }
        }
        try:
            with open(self._persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._last_save_ts = time.time()
            logging.info(f'[SemanticMemory] Saved to {self._persist_path} (updates={self._update_count})')
        except Exception as e:
            logging.error(f'[SemanticMemory] Failed to save: {e}')

    def _merge_from_dict(self, data: dict):
        """将磁盘数据合并到运行时数据（磁盘数据优先级更高，因为包含学习到的新知识）。"""
        # 合并 scene_object_priors
        disk_priors = data.get('scene_object_priors', {})
        for scene, objs in disk_priors.items():
            if scene not in self.SCENE_OBJECT_PRIORS:
                self.SCENE_OBJECT_PRIORS[scene] = {}
            for obj, prob in objs.items():
                self.SCENE_OBJECT_PRIORS[scene][obj] = prob

        # 合并 spatial_co_occurrence
        disk_co = data.get('spatial_co_occurrence', {})
        for obj, co_objs in disk_co.items():
            if obj not in self.SPATIAL_CO_OCCURRENCE:
                self.SPATIAL_CO_OCCURRENCE[obj] = {}
            for co_obj, info in co_objs.items():
                self.SPATIAL_CO_OCCURRENCE[obj][co_obj] = info

        # 合并 scene_adjacency
        disk_adj = data.get('scene_adjacency', {})
        for scene, adj_scenes in disk_adj.items():
            if scene not in self.SCENE_ADJACENCY:
                self.SCENE_ADJACENCY[scene] = {}
            for adj_scene, prob in adj_scenes.items():
                self.SCENE_ADJACENCY[scene][adj_scene] = prob

        # 合并 foresight_rules（去重：按 trigger+prediction 去重）
        disk_rules = data.get('foresight_rules', [])
        existing_keys = {(r['trigger'], r['prediction']) for r in self.FORESIGHT_RULES}
        for rule in disk_rules:
            key = (rule.get('trigger', ''), rule.get('prediction', ''))
            if key not in existing_keys:
                self.FORESIGHT_RULES.append(rule)
                existing_keys.add(key)

        # 合并 memscenes 的 spatial_features
        disk_memscenes = data.get('memscenes', {})
        for st, ms_data in disk_memscenes.items():
            if st in self.memscenes:
                self.memscenes[st].spatial_features.update(ms_data.get('spatial_features', {}))

    # ================================================================
    #  在线学习: 遇到新数据自动补充
    # ================================================================

    def learn_scene_object(self, scene_type: str, object_name: str,
                           observed_prob: float = None, alpha: float = 0.1):
        """
        从观测中学习新的 场景-物体 关系。
        如果该 (scene, object) 对已存在，用 EMA 更新概率；
        如果不存在，自动添加到数据库中。

        Args:
            scene_type: 场景类型（如 'kitchen'）
            object_name: 检测到的物体
            observed_prob: 观测概率（None 则使用默认 0.5）
            alpha: EMA 平滑系数
        """
        if scene_type == 'unknown':
            return

        if scene_type not in self.SCENE_OBJECT_PRIORS:
            # 全新场景类型！创建
            self.SCENE_OBJECT_PRIORS[scene_type] = {}
            logging.info(f'[SemanticMemory-Learn] New scene type discovered: {scene_type}')

        old_prob = self.SCENE_OBJECT_PRIORS[scene_type].get(object_name)
        new_prob = observed_prob if observed_prob is not None else 0.5

        if old_prob is None:
            # 全新物体-场景关联
            self.SCENE_OBJECT_PRIORS[scene_type][object_name] = new_prob
            logging.info(f'[SemanticMemory-Learn] New association: {object_name} in {scene_type} = {new_prob:.2f}')
        else:
            # EMA 更新
            updated = (1 - alpha) * old_prob + alpha * new_prob
            self.SCENE_OBJECT_PRIORS[scene_type][object_name] = round(updated, 4)

        self._update_count += 1
        # 重建该场景的 MemScene
        self._rebuild_memscene(scene_type)

    def learn_co_occurrence(self, obj_a: str, obj_b: str, distance: float,
                            alpha: float = 0.1):
        """
        从共现观测中学习 物体-物体 空间关系。
        """
        if obj_a not in self.SPATIAL_CO_OCCURRENCE:
            self.SPATIAL_CO_OCCURRENCE[obj_a] = {}

        if obj_b in self.SPATIAL_CO_OCCURRENCE[obj_a]:
            old = self.SPATIAL_CO_OCCURRENCE[obj_a][obj_b]
            old['prob'] = round((1 - alpha) * old['prob'] + alpha * 1.0, 4)
            old['typical_dist'] = round((1 - alpha) * old['typical_dist'] + alpha * distance, 2)
        else:
            self.SPATIAL_CO_OCCURRENCE[obj_a][obj_b] = {
                'prob': 0.5, 'typical_dist': round(distance, 2)
            }
            logging.info(f'[SemanticMemory-Learn] New co-occurrence: {obj_a} ↔ {obj_b} @ {distance:.1f}m')

        self._update_count += 1

    def learn_scene_adjacency(self, scene_a: str, scene_b: str, alpha: float = 0.1):
        """从相邻观测中学习 场景-场景 邻接关系。"""
        if scene_a == 'unknown' or scene_b == 'unknown':
            return

        if scene_a not in self.SCENE_ADJACENCY:
            self.SCENE_ADJACENCY[scene_a] = {}

        old = self.SCENE_ADJACENCY[scene_a].get(scene_b, 0.0)
        updated = (1 - alpha) * old + alpha * 1.0
        self.SCENE_ADJACENCY[scene_a][scene_b] = round(updated, 4)
        self._update_count += 1

    def learn_foresight_rule(self, trigger: str, prediction: str, confidence: float):
        """添加新的 Foresight 规则（去重）。"""
        existing_keys = {(r['trigger'], r['prediction']) for r in self.FORESIGHT_RULES}
        if (trigger, prediction) not in existing_keys:
            self.FORESIGHT_RULES.append({
                'trigger': trigger,
                'prediction': prediction,
                'confidence': confidence
            })
            self._update_count += 1
            logging.info(f'[SemanticMemory-Learn] New foresight: {trigger} → {prediction}')
            # 更新对应场景的 MemScene
            if trigger.startswith('scene:'):
                scene = trigger.split(':')[1]
                self._rebuild_memscene(scene)

    def _rebuild_memscene(self, scene_type: str):
        """重建单个场景的 MemScene 对象。"""
        if scene_type not in self.SCENE_OBJECT_PRIORS:
            return
        scene_foresights = [
            r for r in self.FORESIGHT_RULES
            if f'scene:{scene_type}' in r['trigger']
        ]
        self.memscenes[scene_type] = MemScene(
            scene_type=scene_type,
            object_priors=self.SCENE_OBJECT_PRIORS[scene_type],
            foresight_rules=scene_foresights
        )

    # ================================================================
    #  构建 MemScene
    # ================================================================

    def _build_memscenes(self):
        """从先验数据构建 MemScene 对象库。"""
        for scene_type, obj_priors in self.SCENE_OBJECT_PRIORS.items():
            scene_foresights = [
                r for r in self.FORESIGHT_RULES 
                if f'scene:{scene_type}' in r['trigger']
            ]
            self.memscenes[scene_type] = MemScene(
                scene_type=scene_type,
                object_priors=obj_priors,
                foresight_rules=scene_foresights
            )

    # ================================================================
    #  查询接口（保持不变）
    # ================================================================

    def get_likely_scenes(self, goal_object: str, threshold: float = 0.3) -> list:
        """返回目标物体可能出现的场景列表，按概率排序。"""
        scenes = []
        for scene, objects in self.SCENE_OBJECT_PRIORS.items():
            prob = objects.get(goal_object, 0.0)
            if prob >= threshold:
                scenes.append((scene, prob))
        scenes.sort(key=lambda x: x[1], reverse=True)
        return scenes

    def get_co_occurrence_clues(self, goal_object: str) -> list:
        """返回与目标物体空间共现的线索物体列表。"""
        clues = []
        for obj, co_objs in self.SPATIAL_CO_OCCURRENCE.items():
            if goal_object in co_objs:
                clues.append({'clue_object': obj, 'prob': co_objs[goal_object]['prob'],
                              'typical_dist': co_objs[goal_object]['typical_dist']})
        # 也检查目标本身是否是其他物体的线索
        if goal_object in self.SPATIAL_CO_OCCURRENCE:
            for co_obj, info in self.SPATIAL_CO_OCCURRENCE[goal_object].items():
                clues.append({'clue_object': co_obj, 'prob': info['prob'],
                              'typical_dist': info['typical_dist']})
        return clues

    def get_adjacent_scene_prob(self, current_scene: str, target_scene: str) -> float:
        """获取从当前场景到目标场景的邻接概率。"""
        if current_scene in self.SCENE_ADJACENCY:
            return self.SCENE_ADJACENCY[current_scene].get(target_scene, 0.05)
        return 0.05

    def match_scene_type(self, detected_objects: list) -> tuple:
        """
        根据检测到的物体列表匹配最可能的场景类型。
        返回 (scene_type, confidence_score)。
        """
        best_scene = 'unknown'
        best_score = 0.0
        for scene, obj_probs in self.SCENE_OBJECT_PRIORS.items():
            score = sum(obj_probs.get(obj, 0.0) for obj in detected_objects)
            if len(detected_objects) > 0:
                score /= len(detected_objects)
            if score > best_score:
                best_score = score
                best_scene = scene
        return best_scene, best_score

    def compute_scene_prior_bonus(self, goal_object: str, adjacent_scene_type: str) -> float:
        """计算某方向的场景先验加成分数。"""
        # 检查该场景类型中目标物体的概率
        if adjacent_scene_type in self.SCENE_OBJECT_PRIORS:
            return self.SCENE_OBJECT_PRIORS[adjacent_scene_type].get(goal_object, 0.0)
        return 0.0


class TopologicalSpatialMemory:
    """
    拓扑空间记忆 (LTM) — 任务特定的长期记忆。
    包含全局语义地图、frontier 节点和轨迹历史。
    """

    def __init__(self, map_size: int = 5000):
        self.map_size = map_size
        # 全局语义地图: grid_coords → {objects, scene_type, step_discovered}
        self.global_semantic_map = {}
        # 未探索边界节点
        self.frontier_nodes = []
        # 轨迹历史
        self.trajectory_history = []
        # 已发现的物体全局位置
        self.discovered_objects = {}  # object_name → [{"position": ..., "grid_coords": ..., "step": ...}]
        # 已访问的场景区域
        self.visited_scene_regions = set()  # grid_coords 的集合
        # 上一步检测到的物体集合（用于停滞检测）
        self.last_detected_objects = set()

    def reset(self):
        self.global_semantic_map.clear()
        self.frontier_nodes.clear()
        self.trajectory_history.clear()
        self.discovered_objects.clear()
        self.visited_scene_regions.clear()
        self.last_detected_objects.clear()

    def add_trajectory_point(self, position: np.ndarray, rotation, step: int):
        self.trajectory_history.append({
            'position': position.copy(),
            'rotation': rotation,
            'step': step
        })

    def update_semantic_map(self, grid_coords: tuple, objects: list, scene_type: str, step: int):
        """更新全局语义地图中某个位置的信息。"""
        key = (grid_coords[0] // 10, grid_coords[1] // 10)  # 量化到10格精度
        if key not in self.global_semantic_map:
            self.global_semantic_map[key] = {
                'objects': set(),
                'scene_type': scene_type,
                'step_discovered': step,
                'visit_count': 0
            }
        self.global_semantic_map[key]['objects'].update(objects)
        self.global_semantic_map[key]['visit_count'] += 1
        self.visited_scene_regions.add(key)

    def add_discovered_object(self, obj_name: str, position: np.ndarray, grid_coords: tuple, step: int):
        """记录新发现的物体及其位置。"""
        if obj_name not in self.discovered_objects:
            self.discovered_objects[obj_name] = []
        self.discovered_objects[obj_name].append({
            'position': position.copy() if isinstance(position, np.ndarray) else position,
            'grid_coords': grid_coords,
            'step': step
        })

    def update_frontiers(self, new_frontiers: list, max_nodes: int = 50):
        """更新 frontier 节点列表，移除已探索的并添加新发现的。"""
        # 移除已访问区域中的 frontier
        self.frontier_nodes = [
            f for f in self.frontier_nodes
            if (f['grid_coords'][0] // 10, f['grid_coords'][1] // 10) not in self.visited_scene_regions
        ]
        # 添加新 frontier（去重）
        existing_keys = {(f['grid_coords'][0] // 10, f['grid_coords'][1] // 10) for f in self.frontier_nodes}
        for nf in new_frontiers:
            key = (nf['grid_coords'][0] // 10, nf['grid_coords'][1] // 10)
            if key not in existing_keys and key not in self.visited_scene_regions:
                self.frontier_nodes.append(nf)
                existing_keys.add(key)
        # 限制数量
        if len(self.frontier_nodes) > max_nodes:
            self.frontier_nodes.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
            self.frontier_nodes = self.frontier_nodes[:max_nodes]

    def check_new_discovery(self, current_objects: set) -> bool:
        """检查当前步是否有新物体发现（用于停滞检测）。"""
        has_new = bool(current_objects - self.last_detected_objects)
        self.last_detected_objects = current_objects.copy()
        return has_new

    def is_new_region(self, grid_coords: tuple) -> bool:
        """检查当前位置是否是新区域。"""
        key = (grid_coords[0] // 10, grid_coords[1] // 10)
        return key not in self.visited_scene_regions

    # ================================================================
    #  持久化: 保存 / 加载
    # ================================================================

    def to_dict(self) -> dict:
        """
        将拓扑记忆序列化为可 JSON 化的字典。
        numpy array 会自动转为 list。
        """
        # 全局语义地图：key 是 tuple，需要转为 str
        serialized_map = {}
        for k, v in self.global_semantic_map.items():
            key_str = f"{k[0]},{k[1]}"
            serialized_map[key_str] = {
                'objects': list(v.get('objects', set())),
                'scene_type': v.get('scene_type', 'unknown'),
                'step_discovered': v.get('step_discovered', 0),
                'visit_count': v.get('visit_count', 0),
            }

        # 已发现物体
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

        # 轨迹历史
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
                'position': pos,
                'rotation': rot,
                'step': pt.get('step', 0),
            })

        return {
            'map_size': self.map_size,
            'global_semantic_map': serialized_map,
            'discovered_objects': serialized_objects,
            'trajectory_history': serialized_traj,
            'visited_scene_regions': [list(r) for r in self.visited_scene_regions],
        }

    def save_to_disk(self, filepath: str):
        """将拓扑记忆保存到 JSON 文件。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = self.to_dict()
        data['meta'] = {'save_ts': time.time()}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f'[TopologicalMemory] Saved to {filepath} '
                         f'(map_cells={len(self.global_semantic_map)}, '
                         f'objects={len(self.discovered_objects)})')
        except Exception as e:
            logging.error(f'[TopologicalMemory] Failed to save: {e}')

    @staticmethod
    def load_from_disk(filepath: str) -> 'TopologicalSpatialMemory':
        """
        从 JSON 文件加载拓扑记忆，返回新的 TopologicalSpatialMemory 实例。
        如果文件不存在或加载失败，返回一个空的实例。
        """
        mem = TopologicalSpatialMemory()
        if not os.path.isfile(filepath):
            return mem
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f'[TopologicalMemory] Failed to load {filepath}: {e}')
            return mem

        mem.map_size = data.get('map_size', 5000)

        # 还原全局语义地图
        for key_str, v in data.get('global_semantic_map', {}).items():
            parts = key_str.split(',')
            key = (int(parts[0]), int(parts[1]))
            mem.global_semantic_map[key] = {
                'objects': set(v.get('objects', [])),
                'scene_type': v.get('scene_type', 'unknown'),
                'step_discovered': v.get('step_discovered', 0),
                'visit_count': v.get('visit_count', 0),
            }

        # 还原已发现物体
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

        # 还原轨迹
        for pt in data.get('trajectory_history', []):
            pos = pt.get('position')
            if pos is not None:
                pos = np.array(pos)
            mem.trajectory_history.append({
                'position': pos,
                'rotation': pt.get('rotation'),
                'step': pt.get('step', 0),
            })

        # 还原已访问区域
        for r in data.get('visited_scene_regions', []):
            mem.visited_scene_regions.add(tuple(r))

        logging.info(f'[TopologicalMemory] Loaded from {filepath} '
                     f'(map_cells={len(mem.global_semantic_map)}, '
                     f'objects={len(mem.discovered_objects)})')
        return mem


class MemCell:
    """
    工作记忆单元 (STM) — 封装当前时刻的局部感知信息。
    
    参考 EverMemOS 的 MemCell 设计:
    - episode: 语义描述（对应 EverMemOS 的 episode 字段）
    - atomic_facts: 原子事实列表（对应 EventLog）
    - spatial_data: 空间信息（体素更新、位置等）
    - foresight: 预测性线索（对应 EverMemOS 的 Foresight）
    """

    def __init__(self):
        # 基础观测
        self.current_observation = None   # RGB image
        self.depth_observation = None     # depth image
        self.agent_pose = None            # AgentState
        
        # 核心 MemCell 字段（参考 EverMemOS）
        self.detected_objects = []        # atomic_facts: 当前帧检测到的物体列表
        self.episode = ''                 # episode: 语义描述
        self.local_scene_type = 'unknown' # 当前局部场景类型
        self.scene_confidence = 0.0       # 场景匹配置信度
        
        # 空间数据（spatial_data）
        self.spatial_data = {
            'position': None,             # 当前位置
            'grid_coords': None,          # 栅格坐标
            'voxel_updates': [],          # 体素更新列表
            'navigable_directions': []    # 可导航方向
        }
        
        # Foresight 预测性线索
        self.foresight = []               # [{'prediction': ..., 'confidence': ..., 'source': ...}]
        
        # 目标检测状态
        self.goal_detected = False        # 是否检测到目标
        self.goal_distance = None         # 估算的目标距离（米）
        
        # 元信息
        self.step = 0
        self.timestamp = None

    def update(self, obs: dict, step: int):
        """更新基础观测信息。"""
        self.current_observation = obs.get('color_sensor')
        self.depth_observation = obs.get('depth_sensor')
        self.agent_pose = obs.get('agent_state')
        self.step = step
        
        # 更新空间数据
        if self.agent_pose:
            self.spatial_data['position'] = self.agent_pose.position.copy()
    
    def add_foresight(self, prediction: str, confidence: float, source: str = 'semantic_memory'):
        """添加预测性线索（去重）。"""
        # 检查是否已存在相同的预测（避免重复）
        for existing in self.foresight:
            if isinstance(existing, dict) and existing.get('prediction') == prediction:
                # 已存在，更新置信度（取最大值）
                existing['confidence'] = max(existing['confidence'], confidence)
                return
        
        # 新预测，添加
        self.foresight.append({
            'prediction': prediction,
            'confidence': confidence,
            'source': source,
            'step': self.step
        })

    def to_dict(self):
        """转换为字典（类似 EverMemOS 的序列化）。"""
        return {
            'episode': self.episode,
            'atomic_facts': self.detected_objects,  # 对应 detected_objects
            'scene_type': self.local_scene_type,
            'scene_confidence': self.scene_confidence,
            'spatial_data': self.spatial_data,
            'foresight': self.foresight,
            'step': self.step,
            'timestamp': self.timestamp
        }


class StagnationDetector:
    """
    停滞检测器 — 结合物理层与认知层的自省机制。
    """

    def __init__(self, cfg: dict):
        self.stagnation_threshold = cfg.get('stagnation_threshold', 3)
        self.physical_stuck_threshold = cfg.get('physical_stuck_threshold', 0.1)
        self.physical_stuck_steps = cfg.get('physical_stuck_steps', 2)

        self.no_progress_counter = 0
        self.physical_stuck_counter = 0
        self.prev_position = None
        self.position_history = []

    def reset(self):
        self.no_progress_counter = 0
        self.physical_stuck_counter = 0
        self.prev_position = None
        self.position_history.clear()

    def check_physical_stuck(self, current_position: np.ndarray) -> bool:
        """物理层自省：检查 agent 是否卡住。"""
        if self.prev_position is not None:
            displacement = np.linalg.norm(current_position - self.prev_position)
            if displacement < self.physical_stuck_threshold:
                self.physical_stuck_counter += 1
            else:
                self.physical_stuck_counter = 0
        self.prev_position = current_position.copy()
        self.position_history.append(current_position.copy())
        return self.physical_stuck_counter >= self.physical_stuck_steps

    def check_cognitive_stagnation(self, has_new_objects: bool, is_new_region: bool) -> str:
        """
        认知层自省：检查是否陷入认知停滞。
        返回 planning_mode: "LOCAL" 或 "GLOBAL"
        """
        if has_new_objects or is_new_region:
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1

        if self.no_progress_counter >= self.stagnation_threshold:
            logging.info(f'Cognitive stagnation detected (counter={self.no_progress_counter}), switching to GLOBAL mode')
            return "GLOBAL"
        return "LOCAL"

    def get_status(self) -> dict:
        return {
            'no_progress_counter': self.no_progress_counter,
            'physical_stuck_counter': self.physical_stuck_counter,
            'planning_mode': "GLOBAL" if self.no_progress_counter >= self.stagnation_threshold else "LOCAL"
        }


# ============================================================
# Agent 基类 (保持不变)
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
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)
        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1},
            'logging_data': {},
            'images': {'color_sensor': obs['color_sensor']}
        }
        return agent_action, metadata


# ============================================================
# VLMNavAgent 基类 (保持不变)
# ============================================================

class VLMNavAgent(Agent):
    """
    VLMNav agent 基类，保留原有的 navigability, action_proposer, projection, prompting 四大组件。
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
            cfg['sensor_cfg']['img_width']
        )
        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])
        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
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
            step=self.step_ndx,
            goal=obs['goal']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image
        metadata['images']['voxel_map_chosen'] = self._generate_voxel(
            metadata['a_final'], agent_state=agent_state,
            chosen_action=metadata['step_metadata']['action_number'],
            step=self.step_ndx
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
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
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
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            mag = min(mags)
            cart = [self.e_i_scaling * mag * np.sin(theta), 0, -self.e_i_scaling * mag * np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) +
                     sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac * mag, theta, score < 3])
        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))
        filtered.sort(key=lambda x: x[1])
        if not filtered:
            return []
        if explore:
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
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
                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle * explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta_i)
        if len(out) == 0:
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
        if (not out or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
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
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor'],
                step=self.step_ndx, goal=goal
            )
        return a_final_projected

    def _prompting(self, goal, a_final, images, step_metadata):
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, num_actions=len(a_final))
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
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            return abs(height_map - (agent_state.position[1] - 0.04)) < thresh

    def _get_default_arrows(self):
        angle = np.deg2rad(self.fov / 2) * 0.7
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        default_actions.sort(key=lambda x: x[1])
        return default_actions

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
        for i in range(num_points - 4):
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break
        if i < 5:
            return 0, theta_i
        if self.cfg['navigability_mode'] == 'segmentation':
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616
        else:
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(*out, depth_image[out[1], out[0]],
                                         resolution=self.resolution, focal_length=self.focal_length)
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
        if (self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]):
            return end_px
        return None

    def _project_onto_image(self, a_final, rgb_image, agent_state, sensor_state,
                            chosen_action=None, step=None, goal='', candidate_flag=False):
        scale_factor = rgb_image.shape[0] / 1080
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
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
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name
                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)
                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
        if not candidate_flag and ((self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0)):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)
            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)),
                        font, text_size * 0.75, RED, text_thickness)
        if step is not None:
            step_text = f'step {step}'
            cv2.putText(rgb_image, step_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if goal is not None:
            padding = 20
            text_size = 2.5 * scale_factor
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(f"goal:{goal}", font, text_size, text_thickness)
            text_position = (rgb_image.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(rgb_image, f"goal:{goal}", text_position, font, text_size, (255, 0, 0), text_thickness, cv2.LINE_AA)
        return projected

    def _update_voxel(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position, rotation=None):
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)
        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            return (int(new_coords[0]), int(new_coords[1]))
        return (x, y)

    def _generate_voxel(self, a_final, zoom=9, agent_state=None, chosen_action=None, step=None):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        text_size = 1.25
        text_thickness = 1
        agent_coords = self._global_to_grid(agent_state.position, rotation=None)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0
        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=None)
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15
            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1, x2 = max(0, x - delta), min(max_x, x + delta)
        y1, y2 = max(0, y - delta), min(max_y, y + delta)
        zoomed_map = topdown_map[y1:y2, x1:x2]
        if step is not None:
            cv2.putText(zoomed_map, f'step {step}', (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
        return zoomed_map

    def _action_number_to_polar(self, action_number, a_final):
        try:
            action_number = int(action_number)
            if 0 < action_number <= len(a_final):
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass
        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    def _eval_response(self, response: str):
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')])
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1])
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}') + 1])
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}


# ============================================================
# MerNavAgent — 集成三级记忆系统 (Tri-Level Memory)
# ============================================================

class MerNavAgent(VLMNavAgent):
    """
    WMNav Agent with Tri-Level Memory System:
    1. UniversalSemanticMemory (MemScenes) — 跨任务共享的常识先验
    2. TopologicalSpatialMemory (LTM) — 任务内的全局空间记忆
    3. MemCell (STM) — 当前时刻的局部感知
    + StagnationDetector — 物理层+认知层停滞检测
    """

    def __init__(self, cfg: dict):
        # 初始化记忆配置
        self.memory_cfg = cfg.get('memory_cfg', {})
        self._persist_dir = self.memory_cfg.get('persist_dir') or MEMORY_PERSIST_DIR
        self._auto_update = self.memory_cfg.get('auto_update', True)
        self._episode_id = None  # 当前 episode 标识（用于拓扑记忆文件名）
        
        # === 先初始化三级记忆系统（在调用 super().__init__() 之前） ===
        # 因为 super().__init__() 会调用 reset()，而 reset() 需要访问这些属性
        # 注意：此时 self.map_size 还未设置，使用默认值
        map_size = 5000  # VLMNavAgent.map_size 的默认值
        self.semantic_memory = UniversalSemanticMemory(persist_dir=self._persist_dir)
        self.spatial_memory = TopologicalSpatialMemory(map_size=map_size)
        self.working_memory = MemCell()
        self.stm_history = []  # 工作记忆历史（最近N步）
        self.stm_history_length = self.memory_cfg.get('stm_history_length', 5)
        self.stagnation_detector = StagnationDetector(self.memory_cfg)
        self.enriched_context = {}
        self.current_planning_mode = "LOCAL"
        
        # 调用父类初始化（会触发 reset()）
        super().__init__(cfg)

    def reset(self):
        # ---- 持久化：在重置之前保存拓扑记忆 ----
        if hasattr(self, '_episode_id') and self._episode_id and len(self.spatial_memory.trajectory_history) > 0:
            self.save_spatial_memory(self._episode_id)
        # ---- 持久化：通用语义记忆 write-through ----
        if hasattr(self, 'semantic_memory') and self.semantic_memory._update_count > 0:
            self.semantic_memory.save_to_disk()

        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.cvalue_map = 10 * np.ones((self.map_size, self.map_size, 3), dtype=np.float16)
        self.goal_position = []
        self.goal_mask = None
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        
        # 只在 VLM 已初始化时才调用 reset
        if hasattr(self, 'ActionVLM'):
            self.ActionVLM.reset()
        if hasattr(self, 'PlanVLM'):
            self.PlanVLM.reset()
        if hasattr(self, 'PredictVLM'):
            self.PredictVLM.reset()
        if hasattr(self, 'GoalVLM'):
            self.GoalVLM.reset()

        # 重置三级记忆
        if hasattr(self, 'spatial_memory'):
            self.spatial_memory.reset()
        if hasattr(self, 'working_memory'):
            self.working_memory = MemCell()
        if hasattr(self, 'stm_history'):
            self.stm_history.clear()
        if hasattr(self, 'stagnation_detector'):
            self.stagnation_detector.reset()
        self.enriched_context = {}
        self.current_planning_mode = "LOCAL"
        self._episode_id = None

    # ================================================================
    #  记忆持久化接口
    # ================================================================

    def set_episode_id(self, episode_id: str):
        """设置当前 episode 标识，用于拓扑记忆文件名。"""
        self._episode_id = episode_id

    def save_spatial_memory(self, episode_id: str):
        """将当前拓扑记忆保存到磁盘（按 episode 分文件）。"""
        topo_dir = os.path.join(self._persist_dir, 'topological')
        filepath = os.path.join(topo_dir, f'topo_{episode_id}.json')
        self.spatial_memory.save_to_disk(filepath)

    def save_memories_to_disk(self, episode_id: str = None):
        """
        一键保存所有记忆到磁盘。
        通常在 _post_episode() 中、agent.reset() 之前调用。
        """
        # 1. 通用语义记忆（write-through，只要有更新就写入）
        if self.semantic_memory._update_count > 0:
            self.semantic_memory.save_to_disk()
            logging.info(f'[Persistence] Semantic memory saved '
                         f'(updates={self.semantic_memory._update_count})')

        # 2. 拓扑空间记忆（按 episode 保存）
        ep_id = episode_id or self._episode_id
        if ep_id and len(self.spatial_memory.trajectory_history) > 0:
            self.save_spatial_memory(ep_id)
            logging.info(f'[Persistence] Topological memory saved for episode={ep_id}')

    def load_spatial_memory(self, episode_id: str):
        """从磁盘加载指定 episode 的拓扑记忆。"""
        topo_dir = os.path.join(self._persist_dir, 'topological')
        filepath = os.path.join(topo_dir, f'topo_{episode_id}.json')
        if os.path.isfile(filepath):
            self.spatial_memory = TopologicalSpatialMemory.load_from_disk(filepath)
            logging.info(f'[Persistence] Loaded topological memory for episode={episode_id}')
        else:
            logging.info(f'[Persistence] No topological memory found for episode={episode_id}')


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
    # Phase 1: 记忆预加载与常识注入
    # ================================================================

    def preload_memory(self, goal_object: str):
        """
        Phase 1: 从 MemScenes 加载与目标物体相关的先验知识,
        注入到后续 VLM prompt 的上下文中。
        
        参考 EverMemOS 的记忆加载流程:
        1. 查询 likely_scenes（最可能的场景）
        2. 获取 co_occurrence_clues（空间共现线索）
        3. 生成 foresight_rules（预测性知识）
        4. 激活相关的 MemScene 模板
        """
        likely_scenes = self.semantic_memory.get_likely_scenes(goal_object)
        co_clues = self.semantic_memory.get_co_occurrence_clues(goal_object)

        self.enriched_context = {
            'goal_object': goal_object,
            'likely_scenes': likely_scenes,          # [(scene_type, prob), ...]
            'co_occurrence_clues': co_clues,          # [{'clue_object': ..., 'prob': ..., 'typical_dist': ...}]
            'foresight_rules': self._generate_foresight_rules(goal_object, likely_scenes, co_clues),
            'activated_memscenes': []                 # 激活的 MemScene 列表
        }
        
        # 激活相关的 MemScene
        for scene_type, prob in likely_scenes:
            if scene_type in self.semantic_memory.memscenes:
                memscene = self.semantic_memory.memscenes[scene_type]
                self.enriched_context['activated_memscenes'].append({
                    'scene_type': scene_type,
                    'probability': prob,
                    'memscene': memscene
                })
        
        logging.info(f'[Phase1-Preload] Memory preloaded for {goal_object}:')
        logging.info(f'  - Likely scenes: {[s[0] for s in likely_scenes[:3]]}')
        logging.info(f'  - Clue objects: {[c["clue_object"] for c in co_clues[:3]]}')
        logging.info(f'  - Activated MemScenes: {len(self.enriched_context["activated_memscenes"])}')
        
        return self.enriched_context

    def _generate_foresight_rules(self, goal: str, likely_scenes: list, co_clues: list) -> str:
        """生成预测性知识的文本描述，用于注入 prompt。"""
        rules = []
        if likely_scenes:
            top_scenes = ', '.join([f"{s[0]}({s[1]:.0%})" for s in likely_scenes[:3]])
            rules.append(f"The {goal} is most commonly found in: {top_scenes}.")
        if co_clues:
            clue_strs = [f"{c['clue_object']}(within ~{c['typical_dist']:.1f}m)" for c in co_clues[:3]]
            rules.append(f"If you see {', '.join(clue_strs)}, the {goal} is likely nearby.")
        # 场景连接推理
        for scene, prob in likely_scenes[:2]:
            adj = self.semantic_memory.SCENE_ADJACENCY.get(scene, {})
            if adj:
                top_adj = sorted(adj.items(), key=lambda x: x[1], reverse=True)[:2]
                adj_str = ', '.join([f"{a[0]}" for a in top_adj])
                rules.append(f"Rooms adjacent to {scene} often include: {adj_str}.")
        return ' '.join(rules)

    # ================================================================
    # Phase 2: 感知与记忆生成
    # ================================================================

    def perception_and_memcell(self, obs: dict, detected_objects: list = None):
        """
        Phase 2: 将当前观测封装为 MemCell，进行场景聚类匹配，更新 LTM。
        
        参考 EverMemOS 的 MemCell 提取流程:
        1. 更新 MemCell 的基础字段（observation, detected_objects）
        2. 生成 episode 描述（语义摘要）
        3. 场景聚类匹配（类似 ClusterManager.cluster_memcell）
        4. 触发 Foresight 预测
        5. 更新 LTM（spatial_memory）
        """
        agent_state = obs['agent_state']

        # 2.1 更新工作记忆基础信息
        self.working_memory.update(obs, self.step_ndx)
        if detected_objects:
            self.working_memory.detected_objects = detected_objects
        
        # 更新空间数据
        grid_coords = self._global_to_grid(agent_state.position)
        self.working_memory.spatial_data['grid_coords'] = grid_coords
        
        # 2.2 生成 episode 描述（语义摘要）
        self.working_memory.episode = self._generate_episode_description(
            detected_objects, agent_state
        )

        # 2.3 场景聚类匹配（参考 EverMemOS ClusterManager）
        if detected_objects:
            scene_type, confidence = self.semantic_memory.match_scene_type(detected_objects)
            self.working_memory.local_scene_type = scene_type
            self.working_memory.scene_confidence = confidence
            
            clustering_threshold = self.memory_cfg.get('clustering_threshold', 0.5)
            if confidence > clustering_threshold:
                # 静默激活场景策略（日志在env中统一显示）
                if scene_type in self.semantic_memory.memscenes:
                    memscene = self.semantic_memory.memscenes[scene_type]
                    # 触发 Foresight 预测
                    self._trigger_foresight(memscene, detected_objects)

        # 2.4 更新 LTM
        self.spatial_memory.add_trajectory_point(agent_state.position, agent_state.rotation, self.step_ndx)
        self.spatial_memory.update_semantic_map(
            grid_coords=grid_coords,
            objects=detected_objects or [],
            scene_type=self.working_memory.local_scene_type,
            step=self.step_ndx
        )

        # 2.5 记录发现的物体
        if detected_objects:
            for obj in detected_objects:
                self.spatial_memory.add_discovered_object(obj, agent_state.position, grid_coords, self.step_ndx)

        # 2.6 保存 STM 历史（存储 MemCell 对象的深拷贝，避免引用问题）
        # 创建新的 MemCell 对象作为快照
        memcell_snapshot = MemCell()
        memcell_snapshot.current_observation = self.working_memory.current_observation
        memcell_snapshot.depth_observation = self.working_memory.depth_observation
        memcell_snapshot.agent_pose = self.working_memory.agent_pose
        memcell_snapshot.detected_objects = self.working_memory.detected_objects.copy()
        memcell_snapshot.episode = self.working_memory.episode
        memcell_snapshot.local_scene_type = self.working_memory.local_scene_type
        memcell_snapshot.scene_confidence = self.working_memory.scene_confidence
        memcell_snapshot.spatial_data = copy.deepcopy(self.working_memory.spatial_data)
        memcell_snapshot.foresight = copy.deepcopy(self.working_memory.foresight)  # 深拷贝避免引用问题
        memcell_snapshot.step = self.working_memory.step
        memcell_snapshot.timestamp = self.working_memory.timestamp
        
        self.stm_history.append(memcell_snapshot)
        if len(self.stm_history) > self.stm_history_length:
            self.stm_history.pop(0)

        return self.working_memory
    
    def _generate_episode_description(self, detected_objects: list, agent_state) -> str:
        """
        生成 episode 描述（类似 EverMemOS 的 episode 字段）。
        示例: "我站在走廊尽头，看到了冰箱"
        """
        if not detected_objects:
            return f"Step {self.step_ndx}: No significant objects detected."
        
        obj_list = ', '.join(detected_objects[:3])  # 最多列举3个物体
        scene = self.working_memory.local_scene_type
        if scene != 'unknown':
            return f"Step {self.step_ndx}: In {scene}, detected: {obj_list}"
        else:
            return f"Step {self.step_ndx}: Detected: {obj_list}"
    
    def _trigger_foresight(self, memscene: MemScene, detected_objects: list):
        """
        触发 Foresight 预测（参考 EverMemOS 的 Foresight 记忆）。
        根据检测到的物体和场景规则，生成预测性线索。
        """
        for rule in memscene.foresight_rules:
            trigger = rule['trigger']
            prediction = rule['prediction']
            confidence = rule['confidence']
            
            # 检查触发条件（静默触发，避免日志泛滥）
            if trigger.startswith('scene:'):
                self.working_memory.add_foresight(prediction, confidence, source='memscene')
        
        # 检查物体共现规则（静默）
        for obj in detected_objects:
            co_clues = self.semantic_memory.get_co_occurrence_clues(obj)
            for clue in co_clues:
                prediction = f"{clue['clue_object']} likely within {clue['typical_dist']:.1f}m"
                self.working_memory.add_foresight(
                    prediction, clue['prob'], source='co_occurrence'
                )

    # ================================================================
    # Phase 3: 自省与停滞检测
    # ================================================================

    def review_and_stagnation_check(self, obs: dict, detected_objects: list = None) -> tuple:
        """
        Phase 3: 物理层 + 认知层停滞检测 + 目标检测。
        返回 (planning_mode, physical_stuck, found_goal, goal_close_enough)。
        """
        agent_state = obs['agent_state']
        goal_object = obs.get('goal', '')

        # 3.0 目标检测：如果检测到目标物体，判断距离
        found_goal = False
        goal_close_enough = False
        if goal_object and detected_objects:
            # 检查目标是否在检测列表中
            if goal_object in detected_objects:
                found_goal = True
                # 使用深度图估算目标距离
                estimated_distance = self._estimate_goal_distance_from_depth(obs)
                
                # 距离阈值：小于1.5m认为足够近，可以停止
                if estimated_distance < 1.5:
                    goal_close_enough = True
                    logging.info(f'[Phase3] 🎯 GOAL REACHED: {goal_object} at ~{estimated_distance:.1f}m')
                else:
                    logging.info(f'[Phase3] 🎯 GOAL FOUND: {goal_object} at ~{estimated_distance:.1f}m - approaching...')
                
                # 保存目标信息到working memory
                self.working_memory.goal_detected = True
                self.working_memory.goal_distance = estimated_distance

        # 3.1 物理层自省
        physical_stuck = self.stagnation_detector.check_physical_stuck(agent_state.position)
        if physical_stuck:
            logging.info('[Phase3] Physical stuck detected! Triggering obstacle avoidance.')

        # 3.2 认知层自省
        current_objects = set(detected_objects) if detected_objects else set()
        has_new = self.spatial_memory.check_new_discovery(current_objects)
        grid_coords = self._global_to_grid(agent_state.position)
        is_new = self.spatial_memory.is_new_region(grid_coords)

        planning_mode = self.stagnation_detector.check_cognitive_stagnation(has_new, is_new)
        self.current_planning_mode = planning_mode

        return planning_mode, physical_stuck, found_goal, goal_close_enough

    # ================================================================
    # Phase 4: 价值评估与规划 (增强版)
    # ================================================================

    def enhanced_valuation(self, explorable_value: dict, reason: dict, goal: str) -> tuple:
        """
        Phase 4: 根据 planning_mode 选择局部/全局规划。
        在原有 curiosity_value 基础上融合 MemScenes 先验。
        """
        if explorable_value is None or reason is None:
            return np.random.randint(0, 12), ''

        if self.current_planning_mode == "LOCAL":
            return self._local_valuation(explorable_value, reason, goal)
        else:
            return self._global_valuation(explorable_value, reason, goal)

    def _local_valuation(self, explorable_value: dict, reason: dict, goal: str) -> tuple:
        """
        模式 A: 局部探索。在原有 cvalue 基础上加入场景先验加成。
        """
        scene_prior_scale = self.memory_cfg.get('scene_prior_bonus_scale', 0.3)

        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i - 2) * 30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i + 2) * 30)) if i != 11 else '30'

                if np.all(self.panoramic_mask.get(angle, np.array([False])) == False):
                    continue

                intersection1 = self.effective_mask.get(last_angle, np.zeros_like(self.effective_mask.get(angle, np.array([False])))) & self.effective_mask.get(angle, np.array([False]))
                intersection2 = self.effective_mask.get(angle, np.array([False])) & self.effective_mask.get(next_angle, np.zeros_like(self.effective_mask.get(angle, np.array([False]))))

                mask_minus = self.effective_mask.get(angle, np.array([False])) & ~intersection1 & ~intersection2

                ev = explorable_value.get(angle, 5)
                self.cvalue_map[mask_minus] = self._merge_evalue(self.cvalue_map[mask_minus], ev)
                if not np.all(intersection2 == False):
                    ev_next = explorable_value.get(next_angle, 5)
                    self.cvalue_map[intersection2] = self._merge_evalue(
                        self.cvalue_map[intersection2], (ev + ev_next) / 2
                    )

            if self.goal_mask is not None:
                self.cvalue_map[self.goal_mask] = 10.0

            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                pano_mask = self.panoramic_mask.get(angle, None)
                if pano_mask is None or np.all(pano_mask == False):
                    base_score = explorable_value.get(angle, 5)
                else:
                    base_score = np.mean(self.cvalue_map[pano_mask])

                # === 场景先验加成 ===
                scene_bonus = self._compute_direction_scene_bonus(i, goal)
                final_score[i] = base_score * (1 + scene_prior_scale * scene_bonus)

            idx = max(final_score, key=final_score.get)
            final_reason = reason.get(str(int(idx * 30)), '')
        except Exception as e:
            logging.error(f'[Phase4-LOCAL] Error: {e}')
            idx = np.random.randint(0, 12)
            final_reason = ''

        return idx, final_reason

    def _global_valuation(self, explorable_value: dict, reason: dict, goal: str) -> tuple:
        """
        模式 B: 全局扩张。结合 LTM frontier 和 MemScenes 进行全局规划。
        """
        w1 = self.memory_cfg.get('frontier_exploration_weight', 0.4)
        w2 = self.memory_cfg.get('frontier_semantic_weight', 0.3)
        w3 = self.memory_cfg.get('frontier_distance_weight', 0.2)
        w4 = self.memory_cfg.get('frontier_recency_weight', 0.1)

        try:
            # 首先尝试基于 frontier 的全局评分
            frontier_scores = {}
            if self.spatial_memory.frontier_nodes:
                agent_pos = self.working_memory.agent_pose.position if self.working_memory.agent_pose else None
                for fnode in self.spatial_memory.frontier_nodes:
                    # 计算到 frontier 的方向角度
                    if agent_pos is not None:
                        dx = fnode['position'][0] - agent_pos[0]
                        dz = fnode['position'][2] - agent_pos[2]
                        angle_to_frontier = np.degrees(np.arctan2(dx, -dz)) % 360
                        dist_to_frontier = np.sqrt(dx ** 2 + dz ** 2)

                        # 综合评分
                        exploration_val = fnode.get('exploration_area', 1.0)
                        semantic_val = self.semantic_memory.compute_scene_prior_bonus(
                            goal, fnode.get('adjacent_scene', 'unknown')
                        )
                        distance_penalty = 1.0 / (1.0 + dist_to_frontier)
                        recency = 1.0 / (1.0 + self.step_ndx - fnode.get('step_discovered', 0))

                        score = w1 * exploration_val + w2 * semantic_val + w3 * distance_penalty + w4 * recency
                        # 映射到最近的全景方向
                        dir_idx = int(round(angle_to_frontier / 30)) % 12
                        if dir_idx not in frontier_scores or score > frontier_scores[dir_idx]:
                            frontier_scores[dir_idx] = score

            # 如果有 frontier 信息，结合 explorable_value
            if frontier_scores:
                combined_scores = {}
                for i in range(12):
                    if i % 2 == 0:
                        continue
                    angle = str(int(i * 30))
                    base = explorable_value.get(angle, 5) / 10.0
                    frontier_bonus = frontier_scores.get(i, 0)
                    combined_scores[i] = base + frontier_bonus * 2.0  # 全局模式下 frontier 权重更大
                idx = max(combined_scores, key=combined_scores.get)
                final_reason = reason.get(str(int(idx * 30)), '') + ' [GLOBAL MODE: frontier-guided]'
            else:
                # 回退到局部评估但增加探索偏好
                idx, final_reason = self._local_valuation(explorable_value, reason, goal)
                final_reason += ' [GLOBAL MODE: no frontiers, fallback to local with exploration bias]'

        except Exception as e:
            logging.error(f'[Phase4-GLOBAL] Error: {e}')
            idx = np.random.randint(0, 12)
            final_reason = ''

        return idx, final_reason

    def _compute_direction_scene_bonus(self, direction_idx: int, goal: str) -> float:
        """
        根据方向 idx 和已知的语义信息，计算该方向的场景先验加成。
        """
        # 利用 STM 历史和当前场景类型推断各方向的可能场景
        current_scene = self.working_memory.local_scene_type
        if current_scene == 'unknown':
            return 0.0

        # 查找目标最可能出现的场景
        likely_scenes = self.semantic_memory.get_likely_scenes(goal)
        if not likely_scenes:
            return 0.0

        # 计算从当前场景到目标场景的邻接概率
        bonus = 0.0
        for target_scene, target_prob in likely_scenes[:3]:
            adj_prob = self.semantic_memory.get_adjacent_scene_prob(current_scene, target_scene)
            bonus += target_prob * adj_prob
        return bonus

    # ================================================================
    # Phase 5: 执行与动态更新
    # ================================================================

    def execute_and_update(self, obs: dict, agent_action: PolarAction, detected_objects: list = None):
        """
        Phase 5: 执行动作后的在线记忆更新。
        """
        agent_state = obs['agent_state']
        grid_coords = self._global_to_grid(agent_state.position)

        # 5.1 更新 LTM 语义地图
        self.spatial_memory.update_semantic_map(
            grid_coords=grid_coords,
            objects=detected_objects or [],
            scene_type=self.working_memory.local_scene_type,
            step=self.step_ndx
        )

        # 5.2 从 voxel_map 提取新的 frontier 节点
        new_frontiers = self._extract_frontiers(agent_state)
        self.spatial_memory.update_frontiers(
            new_frontiers,
            max_nodes=self.memory_cfg.get('max_frontier_nodes', 50)
        )

        # 5.3 Learning Phase: 在线学习 —— 如果发现新的场景-物体关联，自动补充到数据库
        # 通用物品黑名单（不参与场景学习）
        GENERIC_OBJECTS = {'door', 'window', 'wall', 'floor', 'ceiling', 'light', 'lamp'}
        # 场景置信度阈值：只有高置信度场景才学习
        SCENE_CONFIDENCE_THRESHOLD = 0.6
        
        if self._auto_update and detected_objects:
            current_scene = self.working_memory.local_scene_type
            scene_confidence = self.working_memory.scene_confidence
            
            # 只在高置信度场景中学习，避免错误关联
            if (current_scene and current_scene != 'unknown' and 
                scene_confidence >= SCENE_CONFIDENCE_THRESHOLD):
                
                for obj in detected_objects:
                    # 跳过通用物品
                    if obj in GENERIC_OBJECTS:
                        continue
                    
                    # 检查该 (scene, object) 是否已在先验库中
                    known = self.semantic_memory.SCENE_OBJECT_PRIORS.get(current_scene, {})
                    if obj not in known:
                        # 全新发现！自动学习（初始概率较低，需要多次观测才能确认）
                        self.semantic_memory.learn_scene_object(
                            scene_type=current_scene,
                            object_name=obj,
                            observed_prob=0.3,  # 降低初始概率，避免误学习
                        )
                        logging.info(f'[Learning] New: {obj} in {current_scene} (conf={scene_confidence:.2f})')
                    else:
                        # 已知关联，用 EMA 更新概率（增强观测到的关联）
                        self.semantic_memory.learn_scene_object(
                            scene_type=current_scene,
                            object_name=obj,
                            observed_prob=1.0,
                            alpha=0.05  # 小步更新
                        )

                # 共现学习：如果同一步检测到多个物体，学习它们的共现关系
                if len(detected_objects) >= 2:
                    for i, obj_a in enumerate(detected_objects):
                        for obj_b in detected_objects[i+1:]:
                            # 估算距离：从深度图或视野范围估算
                            estimated_distance = self._estimate_object_distance(
                                obj_a, obj_b, obs, detected_objects
                            )
                            self.semantic_memory.learn_co_occurrence(
                                obj_a=obj_a, obj_b=obj_b,
                                distance=estimated_distance
                            )

            # 场景邻接学习：如果连续两步的场景类型不同，学习邻接关系
            if (len(self.stm_history) >= 2 and
                    self.stm_history[-1].local_scene_type != 'unknown' and
                    current_scene != 'unknown' and
                    self.stm_history[-1].local_scene_type != current_scene):
                self.semantic_memory.learn_scene_adjacency(
                    scene_a=self.stm_history[-1].local_scene_type,
                    scene_b=current_scene
                )

    def _estimate_goal_distance_from_depth(self, obs: dict) -> float:
        """
        基于深度图估算目标物体的距离。
        
        策略：
        1. 如果有深度图，取中心区域的深度中位数作为目标距离
        2. 深度图通常以米为单位，直接使用
        3. 如果没有深度图，使用默认估算
        
        Returns:
            估算的距离（米）
        """
        depth_sensor = obs.get('depth_sensor')
        
        if depth_sensor is not None:
            # 深度图存在，提取中心区域深度
            h, w = depth_sensor.shape[:2]
            
            # 取中心30%区域（假设目标在视野中心）
            center_h_start = int(h * 0.35)
            center_h_end = int(h * 0.65)
            center_w_start = int(w * 0.35)
            center_w_end = int(w * 0.65)
            
            center_depth = depth_sensor[center_h_start:center_h_end, center_w_start:center_w_end]
            
            # 过滤无效深度值（通常0或极大值表示无效）
            valid_depth = center_depth[(center_depth > 0.1) & (center_depth < 10.0)]
            
            if len(valid_depth) > 0:
                # 使用中位数而非平均值，更鲁棒
                estimated_distance = float(np.median(valid_depth))
                # 限制在合理范围 [0.3m, 5.0m]
                estimated_distance = np.clip(estimated_distance, 0.3, 5.0)
                return round(estimated_distance, 1)
        
        # 深度图不可用，使用保守估算
        return 2.0  # 默认中等距离

    def _estimate_object_distance(self, obj_a: str, obj_b: str, obs: dict, 
                                   detected_objects: list) -> float:
        """
        估算两个检测到的物体之间的距离。
        
        策略：
        1. 如果有深度图，使用深度信息估算
        2. 否则基于FOV和视野范围粗略估算
        3. 默认情况下使用经验值
        
        Returns:
            估算的距离（米）
        """
        # 策略1: 基于FOV的简单估算
        # 假设同一帧中检测到的物体在agent视野范围内
        # 视野宽度约为 2 * distance * tan(FOV/2)
        # 粗略估算物体间距为视野宽度的一半
        
        # 从配置获取视野参数
        max_view_distance = self.cfg.get('max_action_dist', 3.0)  # 最大动作距离
        fov_rad = np.deg2rad(self.cfg.get('fov', 79) / 2)
        
        # 视野宽度（在最大距离处）
        view_width = 2 * max_view_distance * np.tan(fov_rad)
        
        # 物体数量越多，平均间距越小
        num_objects = len(detected_objects)
        if num_objects >= 3:
            # 多个物体，估算间距为视野宽度 / (物体数-1)
            estimated_distance = view_width / (num_objects - 1)
        else:
            # 只有2个物体，估算为视野宽度的一半
            estimated_distance = view_width / 2
        
        # 限制在合理范围内 [0.5m, 5m]
        estimated_distance = np.clip(estimated_distance, 0.5, 5.0)
        
        return round(estimated_distance, 1)

    def _extract_frontiers(self, agent_state) -> list:
        """
        从当前的 voxel_map 中提取 frontier 节点（未探索区域的边界）。
        """
        frontiers = []
        agent_coords = self._global_to_grid(agent_state.position)

        # 在 agent 周围搜索 frontier
        search_radius = int(self.cfg.get('max_action_dist', 3) * self.scale * 2)
        x_center, y_center = agent_coords

        for dx in range(-search_radius, search_radius + 1, 20):
            for dy in range(-search_radius, search_radius + 1, 20):
                gx, gy = x_center + dx, y_center + dy
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    # frontier = 未探索(voxel有颜色) 但 未被 explored_map 标记
                    is_visible = np.any(self.voxel_map[gy, gx] != 0)
                    is_explored = np.all(self.explored_map[gy, gx] == self.explored_color)
                    if is_visible and not is_explored:
                        # 反算回全局坐标
                        global_x = self.init_pos[0] + (gx - self.map_size // 2) / self.scale
                        global_z = self.init_pos[2] + (gy - self.map_size // 2) / self.scale
                        global_pos = np.array([global_x, agent_state.position[1], global_z])
                        frontiers.append({
                            'position': global_pos,
                            'grid_coords': (gx, gy),
                            'adjacent_scene': self.working_memory.local_scene_type,
                            'priority_score': 0.5,
                            'step_discovered': self.step_ndx,
                            'exploration_area': 1.0
                        })

        return frontiers

    # ================================================================
    # 保留原有方法 (navigability, stopping, voxel 等)
    # ================================================================

    def _update_panoramic_voxel(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)

    def _stopping_module(self, obs, threshold_dist=0.8):
        if self.goal_position:
            arr = np.array(self.goal_position)
            avg_goal_position = np.mean(arr, axis=0)
            agent_state = obs['agent_state']
            current_position = np.array([agent_state.position[0], agent_state.position[2]])
            goal_position = np.array([avg_goal_position[0], avg_goal_position[2]])
            dist = np.linalg.norm(current_position - goal_position)
            if dist < threshold_dist:
                return True
        return False

    def _run_threads(self, obs, stopping_images, goal):
        called_stop = False
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

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        arrow_end = np.array([position[0] + direction_vector[0] * arrow_length, position[1],
                              position[2] + direction_vector[2] * arrow_length])
        arrow_end_coords = self._global_to_grid(arrow_end)
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, 1, 2)
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)
        cv2.putText(roomtrack_map, angle_text, text_position, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def generate_voxel(self, agent_state=None, zoom=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)
        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords, angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords, angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords, angle_text=str(angle))
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1.25, 1)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15
        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)
        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, 1.25, RED, 2)
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1, x2 = max(0, x - delta), min(max_x, x + delta)
        y1, y2 = max(0, y - delta), min(max_y, y + delta)
        zoomed_map = topdown_map[y1:y2, x1:x2]
        if self.step_ndx is not None:
            cv2.putText(zoomed_map, f'step {self.step_ndx}', (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
        return zoomed_map

    def update_voxel(self, r, theta, agent_state, temp_map, effective_dist=3):
        agent_coords = self._global_to_grid(agent_state.position)
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)
        cv2.line(temp_map, agent_coords, point, WHITE, 40)
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40)

    def _goal_proposer(self, a_initial, agent_state):
        min_angle = self.fov / self.cfg['spacing_ratio']
        unique = {}
        for mag, theta in a_initial:
            unique.setdefault(theta, []).append(mag)
        arrowData = [[min(mags), theta] for theta, mags in unique.items()]
        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        f = [x for x in arrowData if x[0] > 0]
        f.sort(key=lambda x: x[1])
        if not f:
            return []
        longest = max(f, key=lambda x: x[0])
        longest_theta = longest[1]
        smallest_theta = longest[1]
        longest_ndx = f.index(longest)
        out.append([longest[0], longest[1]])
        thetas.add(longest[1])
        for i in range(longest_ndx + 1, len(f)):
            if f[i][1] - longest_theta > (min_angle * 0.45):
                out.append([f[i][0], f[i][1]])
                thetas.add(f[i][1])
                longest_theta = f[i][1]
        for i in range(longest_ndx - 1, -1, -1):
            if smallest_theta - f[i][1] > (min_angle * 0.45):
                out.append([f[i][0], f[i][1]])
                thetas.add(f[i][1])
                smallest_theta = f[i][1]
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta in out]

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
            a_goal_projected = self._projection(a_goal, candidate_images, agent_state, obs.get('goal', ''), candidate_flag=True)
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
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self.update_voxel(r_i, theta_i, agent_state, temp_map)
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
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
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(r_i, theta_i, agent_state,
                                   clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
                a_initial.append((r_i, theta_i))
        if self.cfg.get('panoramic_padding', False):
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                                         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return self.ActionVLM.get_spend() + self.PlanVLM.get_spend() + self.PredictVLM.get_spend() + self.GoalVLM.get_spend()

    def _prompting(self, goal, a_final, images, step_metadata, subtask):
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))
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
        except:
            number = None
        return number, location_response

    def _get_goal_position(self, action_goal, idx, agent_state):
        for key, value in action_goal.items():
            if value == idx:
                r, theta = key
                break
        local_goal = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        agent_coords = self._global_to_grid(agent_state.position)
        point = self._global_to_grid(global_goal)
        radius = 1
        local_radius = np.array([0, 0, -radius])
        global_radius = local_to_global(agent_state.position, agent_state.rotation, local_radius)
        radius_point = self._global_to_grid(global_radius)
        top_down_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))
        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, top_down_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)
        return global_goal, goal_mask

    def _choose_action(self, obs):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal)

        goal_image = candidate_images['color_sensor'].copy()
        if a_goal is not None:
            goal_number, location_response = self._goal_module(goal_image, a_goal, goal)
            images['goal_image'] = goal_image
            if goal_number is not None and goal_number != 0:
                goal_position, self.goal_mask = self._get_goal_position(a_goal, goal_number, agent_state)
                self.goal_position.append(goal_position)

        step_metadata['object'] = goal

        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if a_goal is not None and goal_number is not None and goal_number != 0:
                logging_data = {}
                logging_data['ACTION_NUMBER'] = int(goal_number)
                step_metadata['action_number'] = goal_number
                a_final = a_goal
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs.get('subtask', '{}'))
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        # Phase 5: 执行后更新记忆
        self.execute_and_update(obs, agent_action, self.working_memory.detected_objects)

        if a_goal is not None:
            logging_data['LOCATOR_RESPONSE'] = location_response

        # 添加记忆系统状态到日志
        stagnation_status = self.stagnation_detector.get_status()
        logging_data['MEMORY_STATUS'] = str({
            'planning_mode': self.current_planning_mode,
            'stagnation': stagnation_status,
            'scene_type': self.working_memory.local_scene_type,
            'frontier_count': len(self.spatial_memory.frontier_nodes)
        })

        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100),
                                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height,
                             10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)
        response = self._predicting_module(inference_image, goal)
        explorable_value = {}
        reason = {}
        try:
            for angle, values in response.items():
                explorable_value[angle] = values['Score']
                reason[angle] = values['Explanation']
        except:
            explorable_value, reason = None, None
        return inference_image, explorable_value, reason

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason, goal=None):
        """增强版: 融合三级记忆进行价值评估。"""
        if goal:
            return self.enhanced_valuation(explorable_value, reason, goal)
        # 回退到原始逻辑
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i - 2) * 30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i + 2) * 30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]
                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2
                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(self.cvalue_map[mask_minus_intersection], explorable_value[angle])
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(self.cvalue_map[intersection2], (explorable_value[angle] + explorable_value[next_angle]) / 2)
            if self.goal_mask is not None:
                self.cvalue_map[self.goal_mask] = 10.0
            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = explorable_value[angle]
                else:
                    final_score[i] = np.mean(self.cvalue_map[self.panoramic_mask[angle]])
            idx = max(final_score, key=final_score.get)
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state=None, zoom=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])
        cvalue_map = (self.cvalue_map / 10 * 255).astype(np.uint8)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        max_x, max_y = cvalue_map.shape[1], cvalue_map.shape[0]
        x1, x2 = max(0, x - delta), min(max_x, x + delta)
        y1, y2 = max(0, y - delta), min(max_y, y + delta)
        zoomed_map = cvalue_map[y1:y2, x1:x2]
        if self.step_ndx is not None:
            cv2.putText(zoomed_map, f'step {self.step_ndx}', (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)
        return zoomed_map

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)
        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'
        return goal_flag, subtask

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
    # 增强后的 Prompt 构建 — 注入记忆上下文
    # ================================================================

    def _construct_prompt(self, goal, prompt_type, subtask='{}', reason='{}', num_actions=0):
        # 构建记忆上下文前缀
        memory_context = self._build_memory_context(goal)

        if prompt_type == 'goal':
            return (
                f"{memory_context}"
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. "
                f"These are labeled with a number in a white circle, which represent the location you can move to. "
                f"First, tell me whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}, return number 0 if there is no {goal}, or if you are not sure. "
                f"Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
                f'Second, if there is {goal} in the image, then determine which circle best represents the location of the {goal}'
                f'(close enough to the target. If a person is standing in that position, they can easily touch the {goal}), and give the number and a reason. '
                f'If none of the circles represent the position of the {goal}, return number 0, and give a reason why you returned 0. '
                "Format your answer in the json {{'Number': <The number you choose>}}"
            )

        if prompt_type == 'predicting':
            return (
                f"{memory_context}"
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, "
                f"each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
                f'Your job is to: (1) List ALL visible objects in your current view, (2) Assign a score to each direction (ranging from 0 to 10). '
                f'Scoring criteria: '
                f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. '
                f"Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
                f'(2) If the {goal} is found, assign a score of 10. '
                f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, '
                f'using your common sense AND the provided spatial knowledge. '
                f'Moving to another area means there is a turn in the corner, an open door, a hallway, etc. '
                f'Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
                "For each direction, provide an explanation for your assigned score. "
                "Format your answer in the json {{'DetectedObjects': [<list of visible objects>], '30': {{'Score': <score>, 'Explanation': <explanation>}}, '90': {{...}}, '150': {{...}}, '210': {{...}}, '270': {{...}}, '330': {{...}}}}."
            )

        if prompt_type == 'planning':
            # 检查是否检测到目标
            goal_guidance = ""
            if hasattr(self, 'working_memory') and self.working_memory.goal_detected:
                goal_distance = self.working_memory.goal_distance
                if goal_distance is not None and goal_distance < 1.5:
                    goal_guidance = (
                        f"🎯 CRITICAL: The {goal} has been detected at approximately {goal_distance:.1f}m away! "
                        f"You are very close to the target. Your HIGHEST priority is to move directly toward the {goal}. "
                        f"Choose actions that bring you closer to the {goal}. "
                    )
                else:
                    goal_guidance = (
                        f"🎯 IMPORTANT: The {goal} has been detected at approximately {goal_distance:.1f}m away! "
                        f"You should prioritize actions that move you closer to the {goal}. "
                    )
            
            stagnation_info = ""
            if self.current_planning_mode == "GLOBAL":
                stagnation_info = (
                    f"IMPORTANT: The agent has been exploring without progress for several steps. "
                    f"Consider searching in a completely different direction or heading to unexplored areas. "
                    f"There are {len(self.spatial_memory.frontier_nodes)} unexplored frontier regions. "
                )

            if reason != '' and subtask != '{}':
                return (
                    f"{memory_context}{goal_guidance}{stagnation_info}"
                    f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                    f"(1)<The observed image>: The image taken from its current location. "
                    f"(2){reason}. This explains why you should go in this direction. "
                    f'Your job is to describe next place to go. '
                    f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. '
                    f"Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
                    f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask}.'
                    f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. '
                    f'Identify a new subtask by describing where you are going next to be more likely to find clues to the {goal}. '
                    f'Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. '
                    f'Note GOING UP OR DOWN STAIRS is an option. '
                    "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}."
                )
            else:
                return (
                    f"{memory_context}{goal_guidance}{stagnation_info}"
                    f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                    f'Your job is to describe next place to go. '
                    f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. '
                    f"Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
                    f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the {goal} '
                    f'and analyze the room type. Note you need to pay special attention to open doors and hallways. '
                    f'Note GOING UP OR DOWN STAIRS is an option. '
                    "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}."
                )

        if prompt_type == 'action':
            if subtask != '{}':
                return (
                    f"{memory_context}"
                    f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                    f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. "
                    f"Explain which action achieves that best. "
                    "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                return (
                    f"{memory_context}"
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                    f"Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                    f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
                    f"Second, tell me which general direction you should go in. "
                    "Lastly, explain which action achieves that best, and return it as {{'action': <action_key>}}. "
                    "Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )

        raise ValueError('Prompt type must be goal, predicting, planning, or action')

    def _build_memory_context(self, goal: str) -> str:
        """
        构建注入到 prompt 中的记忆上下文文本。
        整合 MemScenes 先验 + LTM 已知信息 + STM 当前状态。
        """
        parts = []

        # 1. MemScenes 先验
        if self.enriched_context.get('foresight_rules'):
            parts.append(f"[SPATIAL KNOWLEDGE] {self.enriched_context['foresight_rules']}")

        # 2. LTM 已发现物体
        if self.spatial_memory.discovered_objects:
            discovered_list = list(self.spatial_memory.discovered_objects.keys())[:5]
            parts.append(f"[EXPLORED] Objects found so far: {', '.join(discovered_list)}.")

        # 3. STM 当前场景
        if self.working_memory.local_scene_type != 'unknown':
            parts.append(f"[CURRENT SCENE] You appear to be in a {self.working_memory.local_scene_type}.")

        # 4. 停滞提示
        if self.current_planning_mode == "GLOBAL":
            parts.append("[STRATEGY] No progress detected. Prioritize exploring new, unvisited areas.")

        if parts:
            return ' '.join(parts) + ' '
        return ''
