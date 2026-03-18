"""
Environment for Embodied ObjectNav — Refactored
=================================================
Key changes vs original:
  - No 12-direction panoramic scan loop
  - No curiosity-value map or VLM-based direction voting
  - Agent.step() does everything: mapping + frontier + planning + action
  - Env just executes the action, computes metrics, logs
"""

import gzip
import json
import logging
import math
import os
import random
import shutil
import traceback
import requests

import cv2
import habitat_sim
import numpy as np
import pandas as pd
from PIL import Image

from simWrapper import PolarAction, SimWrapper
from WMNav_agent import Agent, WMNavAgent
from utils import create_gif, create_gif_voxel, log_exception


# ═════════════════════════════════════════════════════════════════════════
# Base Env (unchanged)
# ═════════════════════════════════════════════════════════════════════════

class Env:
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
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
            )

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
                'goal_reached': False,
            }
            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                if self.simWrapper:
                    self.simWrapper.reset()

    def _run_episode(self, episode_ndx: int):
        obs = self._initialize_episode(episode_ndx)
        logging.info(f'\n=================== STARTING: {self.curr_run_name} ===================\n')
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
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(
                agent_state.position - self.prev_agent_position
            )
        self.prev_agent_position = agent_state.position
        return None

    def _post_episode(self):
        self.df.to_pickle(os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'
        ))
        if self.simWrapper:
            self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = (
                    len(self.df[self.df['success'] == 0]) / max(len(self.df), 1)
                )
                response = requests.post(
                    f'http://localhost:{self.cfg["port"]}/log',
                    json=self.wandb_log_data,
                )
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)
        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n=================== RUN COMPLETE ===================\n')
        if self.cfg['log_freq'] == 1:
            run_dir = os.path.join(
                os.environ.get("LOG_DIR"),
                f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
            )
            h = self.agent.cfg['sensor_cfg']['img_height']
            w = self.agent.cfg['sensor_cfg']['img_width']
            create_gif(run_dir, h, w, agent_cls=self.agent_cls)
            create_gif_voxel(run_dir, 1800, 1800)

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
                if im is not None and im.size > 0:
                    im_rgb = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im_rgb.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state, agent_action, geodesic_path, max_steps):
        self.path_calculator.requested_start = agent_state.position
        metrics = {
            'distance_to_goal': self.simWrapper.get_path(self.path_calculator),
            'spl': 0,
            'goal_reached': False,
            'done': False,
            'finish_status': 'running',
        }
        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True
            if metrics['distance_to_goal'] < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': True,
                })
            else:
                metrics['finish_status'] = (
                    'fp' if agent_action is PolarAction.stop else 'max_steps'
                )
        return metrics


# ═════════════════════════════════════════════════════════════════════════
# WMNavEnv — REFACTORED: no panoramic scan, no curiosity value
# ═════════════════════════════════════════════════════════════════════════

class WMNavEnv(Env):
    """
    Simplified environment loop for the refactored 4-phase agent.

    Each step:
      1. Pass obs (with goal) to agent
      2. Agent internally: mapping → frontier → planning → action
      3. Env executes action, computes metrics, logs
    """

    task = 'ObjectNav'

    def _initialize_experiment(self):
        self.all_episodes = []
        dataset = self.cfg['dataset']
        root = os.environ.get("DATASET_ROOT")

        if dataset == 'hm3d_v0.1':
            scene_cfg_path = 'hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v1'
        elif dataset == 'hm3d_v0.2':
            scene_cfg_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v0.2'
        elif dataset == 'mp3d':
            scene_cfg_path = 'mp3d/mp3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_mp3d'
        else:
            raise ValueError(f'Dataset must be hm3d_v0.1, hm3d_v0.2, or mp3d, got {dataset}')

        self.sim_cfg['scene_config'] = os.path.join(root, scene_cfg_path)
        self.goals = {}

        split = self.cfg['split']
        content_dir = os.path.join(root, objnav_path, f'{split}/content')
        for f in sorted(os.listdir(content_dir)):
            with gzip.open(os.path.join(content_dir, f), 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)
        logging.info(f'Loaded {self.num_episodes} episodes from {dataset}/{split}')

    def _initialize_episode(self, episode_ndx: int):
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        dataset = self.cfg['dataset']
        root = os.environ.get("DATASET_ROOT")

        # ── Scene setup ──────────────────────────────────────────────────
        if 'hm3d' in dataset:
            f = episode['scene_id'].split('/')[1:]
            ver = 'hm3d_v0.1' if dataset == 'hm3d_v0.1' else 'hm3d_v0.2'
            self.sim_cfg['scene_id'] = f[1][2:5]
            self.sim_cfg['scene_path'] = os.path.join(
                root, ver, f'{self.cfg["split"]}/{f[1]}/{f[2]}'
            )
            self.simWrapper = SimWrapper(self.sim_cfg)
            goals = self.goals[f[1][6:]]
            all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        elif 'mp3d' in dataset:
            self.sim_cfg['scene_id'] = episode['scene_id'].split('/')[1]
            self.sim_cfg['scene_path'] = os.path.join(root, episode['scene_id'])
            self.simWrapper = SimWrapper(self.sim_cfg)
            goals = self.goals[self.sim_cfg['scene_id']]
            all_objects = goals[
                f'{episode["scene_id"].split("/")[2]}_{episode["object_category"]}'
            ]
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

        # ── Goal positions ───────────────────────────────────────────────
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)

        obj_cat = episode['object_category']
        if obj_cat == 'tv_monitor':
            obj_cat = 'tv screen'

        self.current_episode = {
            'object': obj_cat,
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions,
        }

        logging.info(
            f'EPISODE {episode_ndx}: {obj_cat}, '
            f'{len(all_objects)} instances, '
            f'geodesic={episode["info"]["geodesic_distance"]:.2f}m'
        )

        # ── Spawn agent ──────────────────────────────────────────────────
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        # Clean stale step dirs
        run_dir = os.path.join(
            os.environ.get("LOG_DIR"),
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        )
        if os.path.exists(run_dir):
            for entry in os.listdir(run_dir):
                if entry.startswith('step'):
                    shutil.rmtree(os.path.join(run_dir, entry), ignore_errors=True)

        self.agent.reset()
        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Simplified step: pass obs to agent → get action → execute → log.
        No panoramic scan, no curiosity map, no VLM direction voting.
        """
        agent_state = obs['agent_state']

        # Track distance
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(
                agent_state.position - self.prev_agent_position
            )
        self.prev_agent_position = agent_state.position

        # Inject goal into observation
        obs['goal'] = self.current_episode['object']

        # ── Agent decides action ─────────────────────────────────────────
        agent_action, metadata = self.agent.step(obs)

        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        # ── Metrics ──────────────────────────────────────────────────────
        metrics = self._calculate_metrics(
            agent_state, agent_action,
            self.current_episode['shortest_path'],
            self.cfg['max_steps'],
        )
        step_metadata.update(metrics)

        # ── Logging ──────────────────────────────────────────────────────
        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            return None
        return agent_action

    def _post_episode(self):
        log_dir = os.environ.get("LOG_DIR")
        run_dir = os.path.join(
            log_dir,
            f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        )
        os.makedirs(run_dir, exist_ok=True)
        self.df.to_pickle(os.path.join(run_dir, 'df_results.pkl'))

        if self.simWrapper:
            self.simWrapper.reset()
        self.agent.reset()

        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = (
                    len(self.df[self.df['success'] == 0]) / max(len(self.df), 1)
                )
                response = requests.post(
                    f'http://localhost:{self.cfg["port"]}/log',
                    json=self.wandb_log_data,
                )
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n=================== RUN COMPLETE ===================\n')
        if self.cfg['log_freq'] == 1:
            h = self.agent.cfg['sensor_cfg']['img_height']
            w = self.agent.cfg['sensor_cfg']['img_width']
            create_gif(run_dir, h, w, agent_cls=self.agent_cls)
