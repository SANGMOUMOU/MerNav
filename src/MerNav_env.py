import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim

import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from MerNav_agent import *
from custom_agent import *
from utils import *


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
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

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
            logging.basicConfig(filename=self.log_file, level=logging.INFO,
                                format='%(asctime)s %(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    def _initialize_experiment(self):
        raise NotImplementedError

    def run_experiment(self):
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])
        start_ndx = self.cfg['instance'] * instance_size
        end_ndx = self.num_episodes

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], end_ndx)):
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
                self.simWrapper.reset()

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
        logging.info(f'\n{"="*80}\nSTEP {self.step} | Episode: {self.current_episode["episode_id"]} | Goal: {self.current_episode["object"]}\n{"="*80}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        return None

    def _post_episode(self):
        self.df.to_pickle(os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'
        ))
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)
        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"),
                             f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                self.agent.cfg['sensor_cfg']['img_height'],
                self.agent.cfg['sensor_cfg']['img_width'],
                agent_cls=self.agent_cls
            )
            create_gif_voxel(
                os.path.join(os.environ.get("LOG_DIR"),
                             f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)
        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = os.path.join(
                os.environ.get("LOG_DIR"),
                f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}'
            )
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                if im is not None:
                    im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state, agent_action, geodesic_path, max_steps):
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'
        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True
            if metrics['distance_to_goal'] < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
            else:
                metrics['finish_status'] = 'fp' if agent_action is PolarAction.stop else 'max_steps'
        return metrics


class MerNavEnv(Env):
    """
    MerNav Environment — 集成三级记忆系统的决策闭环。

    决策流程:
      Phase 1: 记忆预加载与常识注入 (episode 初始化时)
      Phase 2: 感知与记忆生成 (全景扫描后)
      Phase 3: 自省与停滞检测 (每步)
      Phase 4: 价值评估与规划 (方向选择)
      Phase 5: 执行与动态更新 (动作执行后)
    """

    task = 'ObjectNav'

    def _initialize_experiment(self):
        self.all_episodes = []
        if self.cfg['dataset'] == 'hm3d_v0.1':
            scene_config_path = 'hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v1'
        elif self.cfg['dataset'] == 'hm3d_v0.2':
            scene_config_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v0.2'
        elif self.cfg['dataset'] == 'mp3d':
            scene_config_path = 'mp3d/mp3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_mp3d'
        else:
            raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')

        self.sim_cfg['scene_config'] = os.path.join(os.environ.get("DATASET_ROOT"), scene_config_path)
        self.goals = {}

        for f in sorted(os.listdir(os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content'))):
            with gzip.open(os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content/{f}'), 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]

        if 'hm3d' in self.cfg['dataset']:
            f = episode['scene_id'].split('/')[1:]
            self.sim_cfg['scene_id'] = f[1][2:5]
            self.sim_cfg['scene_path'] = os.path.join(
                os.environ.get("DATASET_ROOT"),
                'hm3d_v0.1' if self.cfg['dataset'] == 'hm3d_v0.1' else 'hm3d_v0.2',
                f'{self.cfg["split"]}/{f[1]}/{f[2]}'
            )
            self.simWrapper = SimWrapper(self.sim_cfg)
            goals = self.goals[f[1][6:]]
            all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        elif 'mp3d' in self.cfg['dataset']:
            self.sim_cfg['scene_id'] = episode['scene_id'].split('/')[1]
            self.sim_cfg['scene_path'] = os.path.join(os.environ.get("DATASET_ROOT"), f'{episode["scene_id"]}')
            self.simWrapper = SimWrapper(self.sim_cfg)
            goals = self.goals[self.sim_cfg['scene_id']]
            all_objects = goals[f'{episode["scene_id"].split("/")[2]}_{episode["object_category"]}']
        else:
            raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')

        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)

        logging.info(
            f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} '
            f'and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}'
        )

        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'

        self.current_episode = {
            'episode_id': episode_ndx,
            'scene_id': episode.get('scene_id', 'unknown'),
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)

        self.previous_subtask = '{}'

        # ================================================================
        # Phase 1: 记忆预加载与常识注入 (Initialization)
        # ================================================================
        if isinstance(self.agent, MerNavAgent):
            # 设置当前 episode 标识（用于拓扑记忆持久化文件名）
            self.agent.set_episode_id(self.curr_run_name)
            
            # 加载通用语义记忆 (MemScenes)
            # 不仅包含静态概率，还包含动态的预测性知识 (Foresight)
            self.agent.preload_memory(self.current_episode['object'])
            
            # 日志输出预加载的知识
            enriched = self.agent.enriched_context
            logging.info(f'[Phase1-Preload] Memory loaded for goal: {self.current_episode["object"]}')
            logging.info(f'  - Likely scenes: {[s[0] for s in enriched.get("likely_scenes", [])[:3]]}')
            logging.info(f'  - Foresight rules: {enriched.get("foresight_rules", "")[:200]}...')
            logging.info(f'  - Activated MemScenes: {len(enriched.get("activated_memscenes", []))}')

        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment with Tri-Level Memory decision loop.

        Decision Loop:
          Phase 2: 感知与全景扫描 → MemCell 生成
          Phase 3: 自省与停滞检测
          Phase 4: 价值评估与规划 (LOCAL/GLOBAL)
          Phase 5: 执行与动态更新
        """
        episode_images = [(obs['color_sensor'].copy())[:, :, :3]]
        color_origin = episode_images[0]

        loop_action_clockwise = PolarAction(0, -0.167 * np.pi)
        loop_action_counterclock = PolarAction(0, 0.167 * np.pi)

        # ================================================================
        # Phase 2: 感知与记忆生成 (Perception & MemCell Formation)
        # ================================================================
        # 多维观测：获取全景图像与深度信息
        for _ in range(11):
            obs = self.simWrapper.step(loop_action_clockwise)
            if _ % 2 == 0:
                self.agent.navigability(obs, _ + 1)
            episode_images.append((obs['color_sensor'].copy())[:, :, :3])

        nav_map = self.agent.generate_voxel(obs['agent_state'])

        # Phase 2: MemCell 生成与场景匹配
        # 从全景图像中提取检测到的物体（通过 VLM 的 curiosity_value 间接实现）
        panoramic_image, explorable_value, reason = self.agent.make_curiosity_value(
            episode_images[-12:], self.current_episode['object']
        )

        # Phase 2: 更新工作记忆（将当前感知封装为 MemCell）
        if isinstance(self.agent, MerNavAgent):
            # 从 explorable_value 中提取隐含的物体检测信息
            detected_objects = self._extract_detected_objects_from_evaluation(
                explorable_value, reason, self.current_episode['object']
            )
            
            # 核心: MemCell 生成
            # 包含：Episode（语义描述）、Atomic Facts（识别的物体）、Spatial Data（体素更新）
            self.agent.perception_and_memcell(obs, detected_objects)
            
            # 聚类匹配: 判定当前场景类型
            # 如果特征匹配度 > 阈值，激活该场景的搜索策略
            memcell = self.agent.working_memory
            detected_str = ', '.join(memcell.detected_objects[:5]) if memcell.detected_objects else 'None'
            if len(memcell.detected_objects) > 5:
                detected_str += f' (+{len(memcell.detected_objects)-5} more)'
            
            logging.info(
                f'[Phase2 - Working Memory]\n'
                f'  Scene: {memcell.local_scene_type} (conf={memcell.scene_confidence:.2f})\n'
                f'  Objects: {detected_str}'
            )
            
            # 仅显示高置信度的 foresight（精简输出）
            if memcell.foresight and memcell.scene_confidence > 0.5:
                high_conf = [f for f in memcell.foresight[:5] if isinstance(f, dict) and f.get('confidence', 0) >= 0.6]
                if high_conf:
                    logging.info(f'  Key Predictions:')
                    for fs in high_conf:
                        logging.info(f'    • {fs["prediction"]} (conf={fs["confidence"]:.2f})')


        # ================================================================
        # Phase 3: 自省与停滞检测
        # ================================================================
        planning_mode = "LOCAL"
        physical_stuck = False
        found_goal = False
        goal_close_enough = False
        if isinstance(self.agent, MerNavAgent):
            # 将目标信息添加到obs中
            obs['goal'] = self.current_episode['object']
            planning_mode, physical_stuck, found_goal, goal_close_enough = self.agent.review_and_stagnation_check(
                obs, detected_objects
            )
            
            # 精简输出
            status = []
            if found_goal:
                status.append('🎯 Goal Detected')
                if goal_close_enough:
                    status.append('✓ Close Enough')
            if physical_stuck:
                status.append('⚠️ Stuck')
            if planning_mode == "GLOBAL":
                status.append('🌍 Global Search')
            
            status_str = ' | '.join(status) if status else 'Normal Exploration'
            logging.info(f'[Phase3 - Status] {status_str}')

            # 如果找到目标且已经足够近，标记为成功，稍后返回STOP
            if found_goal and goal_close_enough:
                logging.info(f'🎉 Episode SUCCESS! Goal "{self.current_episode["object"]}" found and reached!')
                # 不在这里return，继续执行以完成logging，在后面设置agent_action为stop

            # 物理卡死时的应急处理：强制转向
            if physical_stuck and not (found_goal and goal_close_enough):  # 如果已经到达目标就不处理stuck
                logging.info('[Phase3] Physical stuck! Forcing random rotation.')
                random_rotation = PolarAction(0, random.choice([-1, 1]) * np.pi * 0.5)
                obs = self.simWrapper.step(random_rotation)

        # ================================================================
        # Phase 4: 价值评估与规划
        # ================================================================
        if isinstance(self.agent, MerNavAgent):
            # 增强版：融合三级记忆进行方向评估
            goal_rotate, goal_reason = self.agent.update_curiosity_value(
                explorable_value, reason, goal=self.current_episode['object']
            )
        else:
            goal_rotate, goal_reason = self.agent.update_curiosity_value(explorable_value, reason)

        direction_image = episode_images[-12:][goal_rotate]
        goal_flag, subtask = self.agent.make_plan(
            direction_image, self.previous_subtask, goal_reason, self.current_episode['object']
        )
        self.previous_subtask = subtask

        # 转向目标方向
        for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
            if goal_rotate <= 6:
                obs = self.simWrapper.step(loop_action_clockwise)
            else:
                obs = self.simWrapper.step(loop_action_counterclock)

        cvalue_map = self.agent.draw_cvalue_map(obs['agent_state'])

        # 更新距离追踪
        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']
        obs['subtask'] = subtask
        obs['goal_flag'] = goal_flag
        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position

        # ================================================================
        # Phase 5: 执行与动态更新
        # ================================================================
        # 如果已经到达目标，直接返回STOP，跳过后续执行
        if found_goal and goal_close_enough:
            agent_action = PolarAction.stop
            # 创建简化的metadata用于logging
            metadata = {
                'step_metadata': {'action_number': -1, 'success': 1},
                'logging_data': {},
                'images': {'color_sensor': obs['color_sensor']}
            }
            step_metadata = metadata['step_metadata']
        else:
            agent_action, metadata = self.agent.step(obs)
            step_metadata = metadata['step_metadata']

        # 日志增强：记录记忆系统状态
        metadata['logging_data']['EVALUATOR_RESPONSE'] = str({
            'goal_rotate': goal_rotate * 30,
            'explorable_value': explorable_value,
            'reason': reason,
            'planning_mode': planning_mode  # 新增
        })
        metadata['logging_data']['PLANNING_RESPONSE'] = str({
            'goal_flag': goal_flag,
            'subtask': subtask,
            'physical_stuck': physical_stuck  # 新增
        })

        # 记忆系统状态日志
        if isinstance(self.agent, MerNavAgent):
            metadata['logging_data']['MEMORY_STATE'] = str({
                'scene_type': self.agent.working_memory.local_scene_type,
                'scene_confidence': self.agent.working_memory.scene_confidence,
                'stagnation_counter': self.agent.stagnation_detector.no_progress_counter,
                'frontier_count': len(self.agent.spatial_memory.frontier_nodes),
                'discovered_objects': list(self.agent.spatial_memory.discovered_objects.keys()),
                'planning_mode': planning_mode
            })

        logging_data = metadata['logging_data']
        images = metadata['images']

        # 可视化标注
        if metadata['step'] is not None:
            step_text = f"step {metadata['step']}"
            color_origin = np.ascontiguousarray(color_origin)
            color_origin = cv2.putText(color_origin, step_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if obs['goal'] is not None:
            scale_factor = color_origin.shape[0] / 1080
            padding = 20
            text_size = 2.5 * scale_factor
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                f"goal:{obs['goal']}", cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
            )
            text_position = (color_origin.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(color_origin, f"goal:{obs['goal']}", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), text_thickness, cv2.LINE_AA)

        # 添加规划模式标注到原始图像
        if isinstance(self.agent, MerNavAgent):
            mode_color = (0, 0, 255) if planning_mode == "GLOBAL" else (0, 255, 0)
            mode_text = f"MODE: {planning_mode}"
            cv2.putText(color_origin, mode_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2, cv2.LINE_AA)
            scene_text = f"SCENE: {self.agent.working_memory.local_scene_type}"
            cv2.putText(color_origin, scene_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        planner_images = {
            'panoramic': panoramic_image,
            'color_origin': color_origin,
            'nav_map': nav_map,
            'cvalue_map': cvalue_map
        }
        images.update(planner_images)

        metrics = self._calculate_metrics(
            agent_state, agent_action, self.current_episode['shortest_path'], self.cfg['max_steps']
        )
        step_metadata.update(metrics)

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None

        return agent_action

    def _extract_detected_objects_from_evaluation(self, explorable_value: dict, reason: dict,
                                                   goal_object: str) -> list:
        """
        从 VLM 的 explorable_value 评估响应中提取物体检测信息。
        优先使用VLM返回的DetectedObjects字段，如果没有则回退到文本匹配。
        """
        # 优先使用VLM直接返回的DetectedObjects
        if explorable_value and 'DetectedObjects' in explorable_value:
            detected = explorable_value['DetectedObjects']
            if isinstance(detected, list):
                return detected
        
        # 回退方案：从reason文本中提取
        common_objects = [
            'chair', 'table', 'sofa', 'couch', 'bed', 'toilet', 'sink', 'tv', 'tv_monitor',
            'refrigerator', 'microwave', 'oven', 'plant', 'bookshelf', 'dresser', 'desk',
            'door', 'window', 'lamp', 'shelf', 'cabinet', 'counter', 'staircase', 'bathtub',
            'shower', 'mirror', 'rug', 'curtain', 'pillow', 'painting', 'clock'
        ]

        detected_set = set()
        if reason:
            for angle, text in reason.items():
                if isinstance(text, str):
                    text_lower = text.lower()
                    for obj in common_objects:
                        if obj in text_lower:
                            detected_set.add(obj)

        # 也检查 goal 是否被高分评估（score >= 8 意味着可能看到了）
        if explorable_value:
            for angle, score in explorable_value.items():
                if isinstance(score, (int, float)) and score >= 8:
                    detected_set.add(goal_object)

        return list(detected_set)

    def _post_episode(self):
        """
        Called after the episode is complete.
        Includes memory system statistics in the log.
        """
        self.df.to_pickle(os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'
        ))

        # 记录记忆系统的 episode 级统计
        if isinstance(self.agent, MerNavAgent):
            memory_stats = {
                'total_frontier_nodes': len(self.agent.spatial_memory.frontier_nodes),
                'total_discovered_objects': len(self.agent.spatial_memory.discovered_objects),
                'total_trajectory_points': len(self.agent.spatial_memory.trajectory_history),
                'final_planning_mode': self.agent.current_planning_mode,
                'final_stagnation_counter': self.agent.stagnation_detector.no_progress_counter,
                'visited_regions': len(self.agent.spatial_memory.visited_scene_regions)
            }
            logging.info(f'[Memory Stats] {memory_stats}')

            # ---- 持久化：在 reset 之前保存所有记忆到磁盘 ----
            try:
                self.agent.save_memories_to_disk(episode_id=self.curr_run_name)
            except Exception as e:
                logging.error(f'[Persistence] Failed to save memories: {e}')

        self.simWrapper.reset()
        self.agent.reset()

        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')

        if self.cfg['log_freq'] == 1:
            base_path = os.path.join(
                os.environ.get("LOG_DIR"),
                f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
            )
            create_gif(base_path, self.agent.cfg['sensor_cfg']['img_height'],
                        self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls)
            create_gif_nav(base_path, 1800, 1800)
            create_gif_cvalue(base_path, 1800, 1800)
