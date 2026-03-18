"""
ObjectNav Environment — 优化版

优化要点:
1. _step_env 拆分为多个职责单一的方法
2. 消除 isinstance 检查，改用 hasattr/协议 或统一接口
3. 魔数提取为配置常量
4. _post_episode 消除重复，子类仅扩展差异部分
5. 异常处理明确化
6. 感知/规划/执行/可视化 职责分离
7. 全景扫描独立为可复用方法
"""

import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import habitat_sim
import numpy as np
import pandas as pd
from PIL import Image

from simWrapper import PolarAction, SimWrapper
from MerNav_agent import *
from custom_agent import *
from utils import *


# ============================================================
# 常量集中管理（消除魔数）
# ============================================================
@dataclass(frozen=True)
class PanoramicConfig:
    """全景扫描相关配置"""
    num_directions: int = 12                    # 全景扫描方向数
    rotation_fraction: float = 0.167            # 每次旋转占 π 的比例
    angle_step_deg: int = 30                    # 每个方向间隔角度(度)
    nav_sample_interval: int = 2                # 每隔几个方向做一次 navigability

    @property
    def clockwise_action(self) -> PolarAction:
        return PolarAction(0, -self.rotation_fraction * np.pi)

    @property
    def counterclockwise_action(self) -> PolarAction:
        return PolarAction(0, self.rotation_fraction * np.pi)


PANORAMIC_CFG = PanoramicConfig()


# ============================================================
# Base Environment
# ============================================================
class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    """
    task = 'Not defined'

    def __init__(self, cfg: dict):
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'

        self._initialize_logging(cfg)
        self._initialize_agent(cfg)

        self.outer_run_name = f'{self.task}_{self.cfg["name"]}'
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper: Optional[SimWrapper] = None
        self.num_episodes = 0
        self._initialize_experiment()

    # ---- 初始化 ----

    def _initialize_agent(self, cfg: dict):
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg'])
        self.agent_cls = cfg['agent_cls']

    def _initialize_logging(self, cfg: dict):
        self.log_file = os.path.join(
            os.environ.get("LOG_DIR"),
            f'{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt'
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file, level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    def _initialize_experiment(self):
        raise NotImplementedError

    # ---- Episode 生命周期 ----

    def run_experiment(self):
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])
        start_ndx = self.cfg['instance'] * instance_size
        end_ndx = self.num_episodes

        # 支持从指定 episode 开始，并可选过滤 scene_id
        skip_to = self.cfg.get('start_episode', None)       # 如 3
        target_scene = self.cfg.get('target_scene_id', None) # 如 "877"

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], end_ndx)):

            # 跳过指定 episode 之前的任务
            if skip_to is not None and episode_ndx < skip_to:
                continue

            # 跳过不匹配的 scene_id
            if target_scene is not None:
                episode = self.all_episodes[episode_ndx]
                scene_id = self._extract_scene_id(episode)
                if scene_id != str(target_scene):
                    continue

            self.wandb_log_data = {
                'episode_ndx': episode_ndx,
                'instance': self.inner_run_name,
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],
                'task': self.task,
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }
            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                if self.simWrapper:
                    self.simWrapper.reset()

    def _extract_scene_id(self, episode: dict) -> str:
        """从 episode 中提取 scene_id，兼容 hm3d 和 mp3d"""
        dataset = self.cfg['dataset']
        if 'hm3d' in dataset:
            # scene_id 格式如 .../00877-xxx/xxx.basis.glb → 提取 877
            f = episode['scene_id'].split('/')[1:]
            return f[1][2:5]  # 与 _setup_scene 中逻辑一致
        elif 'mp3d' in dataset:
            return episode['scene_id'].split('/')[1]
        return ''

    def _run_episode(self, episode_ndx: int):
        obs = self._initialize_episode(episode_ndx)
        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                agent_action = self._step_env(obs)
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)
            except Exception as e:
                log_exception(e)
            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None

    def _step_env(self, obs: dict):
        logging.info(
            f'\n{"=" * 80}\nSTEP {self.step} | Episode: {self.current_episode["episode_id"]}'
            f' | Goal: {self.current_episode["object"]}\n{"=" * 80}'
        )
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        return None

    def _post_episode(self):
        """Episode 结束后的清理与日志。子类通过 _pre_reset_hook / _extra_gifs 扩展。"""
        # 保存 DataFrame
        self._save_dataframe()

        # 子类扩展点：在 reset 前执行（如保存记忆）
        self._pre_reset_hook()

        # 重置
        self.simWrapper.reset()
        self.agent.reset()

        # 上报 wandb
        self._report_metrics()

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')

        # 生成 GIF
        if self.cfg['log_freq'] == 1:
            self._generate_gifs()

    def _pre_reset_hook(self):
        """子类扩展点：在 reset() 之前执行的操作"""
        pass

    def _save_dataframe(self):
        path = os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_pickle(path)

    def _report_metrics(self):
        if not self.cfg['parallel']:
            return
        try:
            self.wandb_log_data['spend'] = self.agent.get_spend()
            num_failures = len(self.df[self.df['success'] == 0]) if len(self.df) > 0 else 0
            self.wandb_log_data['default_rate'] = num_failures / max(len(self.df), 1)
            response = requests.post(
                f'http://localhost:{self.cfg["port"]}/log',
                json=self.wandb_log_data
            )
            if response.status_code != 200:
                logging.error(f"Failed to send metrics: {response.text}")
        except requests.RequestException as e:
            logging.error(f"Metrics reporting failed: {e}")
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            for frame in tb:
                logging.error(f"Frame {frame.filename} line {frame.lineno}")
            logging.error(e)

    def _generate_gifs(self):
        base_path = os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        )
        create_gif(
            base_path,
            self.agent.cfg['sensor_cfg']['img_height'],
            self.agent.cfg['sensor_cfg']['img_width'],
            agent_cls=self.agent_cls
        )
        create_gif_voxel(base_path, 1800, 1800)

    # ---- 日志与指标 ----

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)
        if self.step % self.cfg['log_freq'] != 0 and step_metadata['success'] != 0:
            return

        path = os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}'
        )
        if not step_metadata['success']:
            path += '_ERROR'
        os.makedirs(path, exist_ok=True)

        for name, im in images.items():
            if im is not None:
                Image.fromarray(im[:, :, :3], mode='RGB').save(f'{path}/{name}.png')

        if step_metadata['success']:
            with open(f'{path}/details.txt', 'w') as file:
                for k, v in logging_data.items():
                    file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state, agent_action, geodesic_path, max_steps):
        self.path_calculator.requested_start = agent_state.position
        distance_to_goal = self.simWrapper.get_path(self.path_calculator)

        metrics = {
            'distance_to_goal': distance_to_goal,
            'spl': 0,
            'goal_reached': False,
            'done': False,
            'finish_status': 'running'
        }

        is_terminal = (agent_action is PolarAction.stop) or (self.step + 1 == max_steps)
        if not is_terminal:
            return metrics

        metrics['done'] = True
        if distance_to_goal < self.cfg['success_threshold']:
            metrics['finish_status'] = 'success'
            metrics['goal_reached'] = True
            metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
        else:
            metrics['finish_status'] = 'fp' if agent_action is PolarAction.stop else 'max_steps'

        self.wandb_log_data.update({
            'spl': metrics['spl'],
            'goal_reached': metrics['goal_reached']
        })
        return metrics


# ============================================================
# MerNav Environment（优化版）
# ============================================================
class MerNavEnv(Env):
    """
    MerNav Environment — 集成三级记忆系统的决策闭环。

    相比原版的改进:
    - _step_env 拆分为独立的 phase 方法，各 phase 职责明确
    - 消除 isinstance 检查，改用 _is_memory_agent 属性
    - 全景扫描抽取为独立方法
    - 可视化与业务逻辑分离
    - 子类通过 hook 扩展父类，而非重写整个方法
    """

    task = 'ObjectNav'

    # ---- 常用物体列表（用于从文本中提取检测到的物体）----
    COMMON_OBJECTS = frozenset([
        'chair', 'table', 'sofa', 'couch', 'bed', 'toilet', 'sink', 'tv', 'tv_monitor',
        'refrigerator', 'microwave', 'oven', 'plant', 'bookshelf', 'dresser', 'desk',
        'door', 'window', 'lamp', 'shelf', 'cabinet', 'counter', 'staircase', 'bathtub',
        'shower', 'mirror', 'rug', 'curtain', 'pillow', 'painting', 'clock'
    ])

    GOAL_HIGH_SCORE_THRESHOLD = 8  # explorable_value >= 此值表示可能看到目标

    def __init__(self, cfg: dict):
        self._is_memory_agent = False  # 在 _initialize_agent 后更新
        super().__init__(cfg)

    def _initialize_agent(self, cfg: dict):
        super()._initialize_agent(cfg)
        # 通过协议检查而非 isinstance，提高解耦性
        self._is_memory_agent = all(
            hasattr(self.agent, attr) for attr in [
                'set_episode_id', 'preload_memory', 'perception_and_memcell',
                'working_memory', 'stagnation_detector', 'spatial_memory',
                'review_and_stagnation_check', 'save_memories_to_disk'
            ]
        )

    # ---- 实验初始化 ----

    def _initialize_experiment(self):
        self.all_episodes = []
        dataset = self.cfg['dataset']

        dataset_configs = {
            'hm3d_v0.1': ('hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json', 'objectnav_hm3d_v1'),
            'hm3d_v0.2': ('hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json', 'objectnav_hm3d_v0.2'),
            'mp3d': ('mp3d/mp3d_annotated_basis.scene_dataset_config.json', 'objectnav_mp3d'),
        }

        if dataset not in dataset_configs:
            raise ValueError(f'Dataset must be one of {list(dataset_configs.keys())}, got: {dataset}')

        scene_config_path, objnav_path = dataset_configs[dataset]
        self.sim_cfg['scene_config'] = os.path.join(os.environ.get("DATASET_ROOT"), scene_config_path)
        self.goals = {}

        content_dir = os.path.join(
            os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content'
        )
        for f in sorted(os.listdir(content_dir)):
            with gzip.open(os.path.join(content_dir, f), 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)

    # ---- Episode 初始化 ----

    def _initialize_episode(self, episode_ndx: int):
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]

        self._setup_scene(episode)
        view_positions = self._setup_goals(episode)

        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)

        object_category = episode['object_category']
        if object_category == 'tv_monitor':
            object_category = 'tv screen'

        self.current_episode = {
            'episode_id': episode_ndx,
            'scene_id': episode.get('scene_id', 'unknown'),
            'object': object_category,
            'shortest_path': episode['info']['geodesic_distance'],
            'view_positions': view_positions
        }

        logging.info(
            f'RUNNING EPISODE {episode_ndx} with {object_category} '
            f'GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]:.2f}'
        )

        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)
        self.previous_subtask = '{}'

        # Phase 1: 记忆预加载
        self._phase1_preload_memory()

        return obs

    def _setup_scene(self, episode: dict):
        """根据数据集类型配置场景路径"""
        dataset = self.cfg['dataset']

        if 'hm3d' in dataset:
            f = episode['scene_id'].split('/')[1:]
            self.sim_cfg['scene_id'] = f[1][2:5]
            hm3d_version = 'hm3d_v0.1' if dataset == 'hm3d_v0.1' else 'hm3d_v0.2'
            self.sim_cfg['scene_path'] = os.path.join(
                os.environ.get("DATASET_ROOT"),
                hm3d_version, f'{self.cfg["split"]}/{f[1]}/{f[2]}'
            )
        elif 'mp3d' in dataset:
            self.sim_cfg['scene_id'] = episode['scene_id'].split('/')[1]
            self.sim_cfg['scene_path'] = os.path.join(
                os.environ.get("DATASET_ROOT"), episode['scene_id']
            )
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        self.simWrapper = SimWrapper(self.sim_cfg)

    def _setup_goals(self, episode: dict) -> list:
        """设置目标物体的 view positions"""
        dataset = self.cfg['dataset']

        if 'hm3d' in dataset:
            f = episode['scene_id'].split('/')[1:]
            goals = self.goals[f[1][6:]]
            all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        elif 'mp3d' in dataset:
            goals = self.goals[self.sim_cfg['scene_id']]
            all_objects = goals[f'{episode["scene_id"].split("/")[2]}_{episode["object_category"]}']
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        return view_positions

    def _phase1_preload_memory(self):
        """Phase 1: 记忆预加载与常识注入"""
        if not self._is_memory_agent:
            return

        goal = self.current_episode['object']
        self.agent.set_episode_id(self.curr_run_name)
        self.agent.preload_memory(goal)

        enriched = self.agent.enriched_context
        likely = [s[0] for s in enriched.get("likely_scenes", [])[:3]]
        foresight = enriched.get("foresight_rules", "")[:200]
        num_memscenes = len(enriched.get("activated_memscenes", []))

        logging.info(
            f'[Phase1] Memory loaded for: {goal} | '
            f'Scenes: {likely} | Foresight: {foresight}... | '
            f'MemScenes: {num_memscenes}'
        )

    # ---- 核心步进逻辑（拆分为多个 Phase） ----

    def _step_env(self, obs: dict):
        """主步进函数，协调各 Phase 的执行"""
        super()._step_env(obs)
        goal = self.current_episode['object']

        # Phase 2: 全景扫描与感知
        pano_result = self._phase2_perception(obs, goal)

        # Phase 3: 自省与停滞检测
        phase3_result = self._phase3_review(obs, pano_result, goal)

        # 如果目标已到达，提前终止
        if phase3_result.found_goal and phase3_result.goal_close_enough:
            return self._handle_goal_reached(obs, goal, pano_result)

        # Phase 4: 价值评估与规划
        obs, phase4_result = self._phase4_planning(
            obs, pano_result, phase3_result, goal
        )

        # Phase 5: 执行与记录
        return self._phase5_execute(
            obs, pano_result, phase3_result, phase4_result, goal
        )

    # ---- Phase 2: 感知 ----

    def _phase2_perception(self, obs: dict, goal: str) -> 'PanoramicResult':
        """全景扫描、navigability 更新、curiosity 评估、MemCell 生成"""
        # 全景扫描
        episode_images, obs_after_scan = self._panoramic_scan(obs)
        nav_map = self.agent.generate_voxel(obs_after_scan['agent_state'])

        # Curiosity value 评估
        panoramic_image, explorable_value, reason = self.agent.make_curiosity_value(
            episode_images[-PANORAMIC_CFG.num_directions:], goal
        )

        # MemCell 生成（仅记忆型 Agent）
        detected_objects = self._extract_detected_objects(explorable_value, reason, goal)
        if self._is_memory_agent:
            self.agent.perception_and_memcell(obs_after_scan, detected_objects)
            self._log_working_memory()

        return PanoramicResult(
            episode_images=episode_images,
            panoramic_image=panoramic_image,
            nav_map=nav_map,
            explorable_value=explorable_value,
            reason=reason,
            detected_objects=detected_objects,
            obs_after_scan=obs_after_scan
        )

    def _panoramic_scan(self, obs: dict) -> tuple:
        """
        执行全景扫描，收集图像并更新 navigability。
        返回 (episode_images, last_obs)
        """
        episode_images = [obs['color_sensor'].copy()[:, :, :3]]
        current_obs = obs

        num_rotations = PANORAMIC_CFG.num_directions - 1  # 已有初始方向

        for i in range(num_rotations):
            current_obs = self.simWrapper.step(PANORAMIC_CFG.clockwise_action)

            # 每隔 nav_sample_interval 个方向做一次 navigability
            if i % PANORAMIC_CFG.nav_sample_interval == 0:
                self.agent.navigability(current_obs, i + 1)

            episode_images.append(current_obs['color_sensor'].copy()[:, :, :3])

        return episode_images, current_obs

    def _log_working_memory(self):
        """记录工作记忆状态（精简输出）"""
        if not self._is_memory_agent:
            return

        wm = self.agent.working_memory
        detected_str = ', '.join(wm.detected_objects[:5]) if wm.detected_objects else 'None'
        if len(wm.detected_objects) > 5:
            detected_str += f' (+{len(wm.detected_objects) - 5} more)'

        logging.info(
            f'[Phase2] Scene: {wm.local_scene_type} (conf={wm.scene_confidence:.2f}) | '
            f'Objects: {detected_str}'
        )

        # 高置信度预测
        if wm.foresight and wm.scene_confidence > 0.5:
            high_conf = [
                f for f in wm.foresight[:5]
                if isinstance(f, dict) and f.get('confidence', 0) >= 0.6
            ]
            for fs in high_conf:
                logging.info(f'  Prediction: {fs["prediction"]} (conf={fs["confidence"]:.2f})')

    # ---- Phase 3: 自省 ----

    def _phase3_review(self, obs: dict, pano: 'PanoramicResult', goal: str) -> 'ReviewResult':
        """自省与停滞检测"""
        if not self._is_memory_agent:
            return ReviewResult()

        obs['goal'] = goal
        planning_mode, physical_stuck, found_goal, goal_close_enough = (
            self.agent.review_and_stagnation_check(obs, pano.detected_objects)
        )

        # 精简日志
        status_parts = []
        if found_goal:
            status_parts.append('🎯 Goal Detected')
            if goal_close_enough:
                status_parts.append('✓ Close Enough')
        if physical_stuck:
            status_parts.append('⚠️ Stuck')
        if planning_mode == "GLOBAL":
            status_parts.append('🌍 Global Search')

        logging.info(f'[Phase3] {" | ".join(status_parts) or "Normal Exploration"}')

        # 物理卡死应急处理
        if physical_stuck and not (found_goal and goal_close_enough):
            logging.info('[Phase3] Physical stuck! Forcing random rotation.')
            random_rotation = PolarAction(0, random.choice([-1, 1]) * np.pi * 0.5)
            self.simWrapper.step(random_rotation)

        return ReviewResult(
            planning_mode=planning_mode,
            physical_stuck=physical_stuck,
            found_goal=found_goal,
            goal_close_enough=goal_close_enough
        )

    # ---- Phase 4: 规划 ----

    def _phase4_planning(self, obs, pano: 'PanoramicResult', review: 'ReviewResult', goal: str):
        """价值评估、方向选择与子任务规划"""
        # 方向评估
        if self._is_memory_agent:
            goal_rotate, goal_reason = self.agent.update_curiosity_value(
                pano.explorable_value, pano.reason, goal=goal
            )
        else:
            goal_rotate, goal_reason = self.agent.update_curiosity_value(
                pano.explorable_value, pano.reason
            )

        # 子任务规划
        direction_image = pano.episode_images[-PANORAMIC_CFG.num_directions:][goal_rotate]
        goal_flag, subtask = self.agent.make_plan(
            direction_image, self.previous_subtask, goal_reason, goal
        )
        self.previous_subtask = subtask

        # 转向目标方向
        obs = self._rotate_to_direction(goal_rotate, obs)

        cvalue_map = self.agent.draw_cvalue_map(obs['agent_state'])

        # 更新 obs
        obs['goal'] = goal
        obs['subtask'] = subtask
        obs['goal_flag'] = goal_flag

        # 更新距离追踪
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(
                agent_state.position - self.prev_agent_position
            )
        self.prev_agent_position = agent_state.position

        return obs, PlanningResult(
            goal_rotate=goal_rotate,
            goal_reason=goal_reason,
            goal_flag=goal_flag,
            subtask=subtask,
            cvalue_map=cvalue_map
        )

    def _rotate_to_direction(self, goal_rotate: int, obs: dict) -> dict:
        """转向目标方向，返回转向后的 obs"""
        num_steps = min(
            PANORAMIC_CFG.num_directions - 1 - goal_rotate,
            1 + goal_rotate
        )
        action = (
            PANORAMIC_CFG.clockwise_action
            if goal_rotate <= PANORAMIC_CFG.num_directions // 2
            else PANORAMIC_CFG.counterclockwise_action
        )
        for _ in range(num_steps):
            obs = self.simWrapper.step(action)
        return obs

    # ---- Phase 5: 执行 ----

    def _phase5_execute(self, obs, pano, review, plan, goal):
        """执行动作、计算指标、记录日志"""
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        # 追加日志数据
        self._append_logging_data(logging_data, pano, review, plan)

        # 可视化
        color_origin = pano.episode_images[0].copy()
        self._annotate_origin_image(color_origin, obs, review)

        planner_images = {
            'panoramic': pano.panoramic_image,
            'color_origin': color_origin,
            'nav_map': pano.nav_map,
            'cvalue_map': plan.cvalue_map
        }
        images.update(planner_images)

        # 指标计算
        metrics = self._calculate_metrics(
            obs['agent_state'], agent_action,
            self.current_episode['shortest_path'],
            self.cfg['max_steps']
        )
        step_metadata.update(metrics)
        self._log(images, step_metadata, logging_data)

        return None if metrics['done'] else agent_action

    def _handle_goal_reached(self, obs, goal, pano):
        """目标已到达时的处理"""
        logging.info(f'🎉 Episode SUCCESS! Goal "{goal}" found and reached!')

        agent_action = PolarAction.stop
        step_metadata = {'action_number': -1, 'success': 1}

        metrics = self._calculate_metrics(
            obs['agent_state'], agent_action,
            self.current_episode['shortest_path'],
            self.cfg['max_steps']
        )
        step_metadata.update(metrics)

        images = {'color_sensor': obs['color_sensor']}
        images['panoramic'] = pano.panoramic_image
        images['nav_map'] = pano.nav_map

        self._log(images, step_metadata, {})
        return None  # 终止 episode

    # ---- 日志增强 ----

    def _append_logging_data(self, logging_data, pano, review, plan):
        """向 logging_data 添加记忆系统相关信息"""
        logging_data['EVALUATOR_RESPONSE'] = str({
            'goal_rotate': plan.goal_rotate * PANORAMIC_CFG.angle_step_deg,
            'explorable_value': pano.explorable_value,
            'reason': pano.reason,
            'planning_mode': review.planning_mode
        })
        logging_data['PLANNING_RESPONSE'] = str({
            'goal_flag': plan.goal_flag,
            'subtask': plan.subtask,
            'physical_stuck': review.physical_stuck
        })

        if self._is_memory_agent:
            logging_data['MEMORY_STATE'] = str({
                'scene_type': self.agent.working_memory.local_scene_type,
                'scene_confidence': self.agent.working_memory.scene_confidence,
                'stagnation_counter': self.agent.stagnation_detector.no_progress_counter,
                'frontier_count': len(self.agent.spatial_memory.frontier_nodes),
                'discovered_objects': list(self.agent.spatial_memory.discovered_objects.keys()),
                'planning_mode': review.planning_mode
            })

    # ---- 可视化（与业务逻辑分离）----

    def _annotate_origin_image(self, image: np.ndarray, obs: dict, review: 'ReviewResult'):
        """在原始图像上添加标注"""
        image = np.ascontiguousarray(image)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Step 计数
        cv2.putText(image, f"step {self.step}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 目标标注
        goal = obs.get('goal')
        if goal:
            scale_factor = image.shape[0] / 1080
            text_size = 2.5 * scale_factor
            text = f"goal:{goal}"
            (tw, th), _ = cv2.getTextSize(text, font, text_size, 2)
            pos = (image.shape[1] - tw - 20, 20 + th)
            cv2.putText(image, text, pos, font, text_size, (255, 0, 0), 2, cv2.LINE_AA)

        # 记忆系统标注
        if self._is_memory_agent:
            mode_color = (0, 0, 255) if review.planning_mode == "GLOBAL" else (0, 255, 0)
            cv2.putText(image, f"MODE: {review.planning_mode}", (10, 70), font, 1, mode_color, 2, cv2.LINE_AA)
            scene = self.agent.working_memory.local_scene_type
            cv2.putText(image, f"SCENE: {scene}", (10, 110), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # ---- 物体检测提取 ----

    def _extract_detected_objects(self, explorable_value: dict, reason: dict, goal: str) -> list:
        """从 VLM 评估响应中提取检测到的物体"""
        # 优先使用 VLM 直接返回的字段
        if explorable_value and isinstance(explorable_value.get('DetectedObjects'), list):
            return explorable_value['DetectedObjects']

        # 回退：从文本中匹配
        detected = set()
        if reason:
            for angle, text in reason.items():
                if not isinstance(text, str):
                    continue
                text_lower = text.lower()
                for obj in self.COMMON_OBJECTS:
                    if obj in text_lower:
                        detected.add(obj)

        # 高分方向表示可能看到了目标
        if explorable_value:
            for angle, score in explorable_value.items():
                if isinstance(score, (int, float)) and score >= self.GOAL_HIGH_SCORE_THRESHOLD:
                    detected.add(goal)

        return list(detected)

    # ---- 子类扩展 Hooks ----

    def _pre_reset_hook(self):
        """Episode 结束前：记录记忆统计并持久化"""
        if not self._is_memory_agent:
            return

        memory_stats = {
            'total_frontier_nodes': len(self.agent.spatial_memory.frontier_nodes),
            'total_discovered_objects': len(self.agent.spatial_memory.discovered_objects),
            'total_trajectory_points': len(self.agent.spatial_memory.trajectory_history),
            'final_planning_mode': self.agent.current_planning_mode,
            'final_stagnation_counter': self.agent.stagnation_detector.no_progress_counter,
            'visited_regions': len(self.agent.spatial_memory.visited_scene_regions)
        }
        logging.info(f'[Memory Stats] {memory_stats}')

        try:
            self.agent.save_memories_to_disk(episode_id=self.curr_run_name)
        except Exception as e:
            logging.error(f'[Persistence] Failed to save memories: {e}')

    def _generate_gifs(self):
        """覆盖父类，生成额外的 nav 和 cvalue GIF"""
        base_path = os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        )
        create_gif(
            base_path,
            self.agent.cfg['sensor_cfg']['img_height'],
            self.agent.cfg['sensor_cfg']['img_width'],
            agent_cls=self.agent_cls
        )
        create_gif_nav(base_path, 1800, 1800)
        create_gif_cvalue(base_path, 1800, 1800)


# ============================================================
# Phase 间传递的数据结构（替代松散的元组/字典）
# ============================================================
@dataclass
class PanoramicResult:
    """Phase 2 全景扫描的输出"""
    episode_images: list
    panoramic_image: np.ndarray
    nav_map: np.ndarray
    explorable_value: Optional[dict]
    reason: Optional[dict]
    detected_objects: list
    obs_after_scan: dict


@dataclass
class ReviewResult:
    """Phase 3 自省检测的输出"""
    planning_mode: str = "LOCAL"
    physical_stuck: bool = False
    found_goal: bool = False
    goal_close_enough: bool = False


@dataclass
class PlanningResult:
    """Phase 4 规划的输出"""
    goal_rotate: int = 0
    goal_reason: str = ''
    goal_flag: bool = False
    subtask: str = '{}'
    cvalue_map: Optional[np.ndarray] = None