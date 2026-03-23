import logging
import math
import random
import habitat_sim
import numpy as np
import cv2
import ast
import concurrent.futures
from collections import deque
import json as _json

from simWrapper import PolarAction
from utils import *
from api import *
#版本二所用的agent代码，需要使用env_2文件和config中的WMNav_2.yaml配置文件才能运行，api_2文件
class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass

class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        return agent_action, metadata

class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
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

        agent_action, metadata = self._choose_action(obs)  # 做完了所有事情
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number'],
            step=self.step_ndx,
            goal=obs['goal']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        # Visualize the chosen voxel map
        metadata['images']['voxel_map_chosen'] = self._generate_voxel(metadata['a_final'],
                                                                      agent_state=agent_state,
                                                                      chosen_action=metadata['step_metadata']['action_number'],
                                                                      step=self.step_ndx)
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

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()
        
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
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

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        if 'done' in dct and int(dct['done']) == 1:
            return True, stopping_response
        
        return False, stopping_response

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
        min_angle = self.fov/self.cfg['spacing_ratio']
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
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta)

        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]


        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState, goal: str, candidate_flag: bool=False):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor'],
            step=self.step_ndx,
            goal=goal,
            candidate_flag=candidate_flag
        )

        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']) and not candidate_flag:
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor'],
                step=self.step_ndx,
                goal=goal
            )

        return a_final_projected

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
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

    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask

    def _get_default_arrows(self):
        """
        Get the action options for when the agent calls stop the first time, or when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7
        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        
        default_actions.sort(key=lambda x: x[1])
        return default_actions

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
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
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None, step: int=None, goal: str='', candidate_flag: bool=False):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        # if candidate_flag:
        #     scale_factor /= 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
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
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        # Add step counter in the top-left corner
        if step is not None:
            step_text = f'step {step}'
            cv2.putText(rgb_image, step_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Add the target string in the top-right corner
        if goal is not None:
            padding = 20
            text_size = 2.5 * scale_factor
            if candidate_flag:
                text_thickness = 2
            else:
                text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(f"goal:{goal}", font, text_size, text_thickness)
            text_position = (rgb_image.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(rgb_image, f"goal:{goal}", text_position, font, text_size, (255, 0, 0), text_thickness,
                        cv2.LINE_AA)

        return projected


    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None, step: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
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

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if step is not None:
            step_text = f'step {step}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    # def _eval_response(self, response: str):
    #     """Converts the VLM response string into a dictionary, if possible"""
    #     try:
    #         eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
    #         if isinstance(eval_resp, dict):
    #             return eval_resp
    #         else:
    #             raise ValueError
    #     except (ValueError, SyntaxError):
    #         logging.error(f'Error parsing response {response}')
    #         return {}
    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

class WMNavAgent(VLMNavAgent):
    def reset(self):
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

        # ── 模块一: 语义先验 ──
        self.target_scene_prior = ""
        self.prior_likely_rooms = []
        self.prior_unlikely_rooms = []
        self._prior_injected = False

        # ── 模块二: 工作记忆 & 局部拓扑 ──
        self.door_memory = []
        self.current_scene_type = "unknown"
        self.steps_since_door_entry = 0
        self._room_exhausted = False
        self._room_mismatch_blocked = False

        # ── 模块三: 空间记忆 & 鲁棒兜底 ──
        self.known_target_coords = None
        self.position_history = deque(maxlen=8)
        self.stuck_counter = 0
        self.recovery_8_dirs = []
        self._recovery_active = False
        self._consecutive_stop_calls = 0

        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.PredictVLM.reset()
        self.GoalVLM.reset()

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

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _stopping_module(self, obs, threshold_dist=1.5):
        if self.goal_position:
            arr = np.array(self.goal_position)
            # 按列计算平均值
            avg_goal_position = np.mean(arr, axis=0)
            agent_state = obs['agent_state']
            current_position = np.array([agent_state.position[0], agent_state.position[2]])
            goal_position = np.array([avg_goal_position[0], avg_goal_position[2]])
            dist = np.linalg.norm(current_position - goal_position)

            if dist < threshold_dist:
                return True
        return False


    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        called_stop = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, obs)

            a_final, images, a_goal, candidate_images = preprocessing_thread.result()
            called_stop = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, a_goal, candidate_images

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray, effective_dist: float=3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40) # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40) # effective area

    def _goal_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        min_angle = self.fov / self.cfg['spacing_ratio']

        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]

        arrowData = []

        for theta, mags in unique.items():
            mag = min(mags)
            arrowData.append([mag, theta])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0
        f = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        f.sort(key=lambda x: x[1])
        if f == []:
            return []
        # Add unexplored actions with spacing, starting with the longest one
        if len(f) > 0:
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

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        
        # GoalVLM 现在只需要干净的无箭头原图
        candidate_images = {'color_sensor': obs['color_sensor'].copy()}

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

        # 仅将方向箭头绘制在给 ActionVLM 的图上
        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        
        # 注意：这里我们移除了 a_goal 的生成和投射
        return a_final_projected, images, None, candidate_images
    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return self.ActionVLM.get_spend()  + self.PlanVLM.get_spend() + self.PredictVLM.get_spend() + self.GoalVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
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

    def _goal_module(self, goal_image: np.array, goal):
        """Determines if the target is visible and returns its pixel coordinates."""
        location_prompt = self._construct_prompt(goal, 'goal')
        location_response = self.GoalVLM.call([goal_image], location_prompt)
        dct = self._eval_response(location_response)

        visible = dct.get('Visible', False)
        coords = dct.get('Coordinates', None)
        reason = dct.get('Reason', '') # 解析新增的 Reason 字段

        if visible and isinstance(coords, list) and len(coords) == 2:
            try:
                coords = [int(coords[0]), int(coords[1])]
            except ValueError:
                coords = None
        else:
            coords = None
            # 只有在没找到目标时才打印原因
            if reason:
                logging.info(f"[GoalVLM] {goal} not visible. Reason: {reason}")

        return visible, coords, location_response

    def _get_goal_position_from_pixels(self, px: int, py: int, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, depth_image: np.ndarray):
        """Convert 2D pixel coordinates to 3D global position using depth."""
        # 限制坐标在图像分辨率范围内
        px = int(np.clip(px, 0, self.resolution[1] - 1))
        py = int(np.clip(py, 0, self.resolution[0] - 1))

        # 获取该像素点的深度值
        depth_val = depth_image[py, px]

        # 如果深度无效或太远（比如天空盒），可以做个安全检查
        if depth_val <= 0 or depth_val > 50.0:
            logging.warning(f"[_get_goal_position] Invalid depth {depth_val} at ({px}, {py})")
            return None, None

        # 反投影到相机坐标系
        camera_coords = unproject_2d(px, py, depth_val, resolution=self.resolution, focal_length=self.focal_length)
        
        # 转换为Agent局部坐标系，再转为全局坐标系
        local_coords = global_to_local(
            agent_state.position, agent_state.rotation,
            local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
        )
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_coords)

        # 生成目标掩码（用于更新 Curiosity Value Map，和原来逻辑保持一致）
        agent_coords = self._global_to_grid(agent_state.position)
        point = self._global_to_grid(global_goal)
        radius = 1  # 目标半径影响范围 (m)
        local_radius = np.array([0, 0, -radius])
        global_radius = local_to_global(agent_state.position, agent_state.rotation, local_radius)
        radius_point = self._global_to_grid(global_radius)
        top_down_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, top_down_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)

        return global_goal, goal_mask

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']
        plan_visible = obs['goal_flag']

        a_final, images, step_metadata, _, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal)

        goal_image = candidate_images['color_sensor'].copy()
        images['goal_image'] = goal_image
        
        distance_to_goal = float('inf')
        logging_data = {}
        step_metadata['object'] = goal

        # 1. 每次 PlanVLM 检测到目标都调用 GoalVLM（去掉 known_target_coords is None 拦截）
        if plan_visible:
            goal_visible, target_pixels, location_response = self._goal_module(goal_image, goal)
            logging_data['LOCATOR_RESPONSE'] = location_response
            
            if goal_visible and target_pixels is not None:
                px, py = target_pixels[0], target_pixels[1]
                
                cv2.circle(images['goal_image'], (px, py), radius=8, color=(0, 255, 0), thickness=-1)
                cv2.circle(images['goal_image'], (px, py), radius=10, color=(255, 0, 0), thickness=2)
                text_str = f"Target: {goal}"
                cv2.putText(images['goal_image'], text_str, (px + 15, py - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                depth_image = obs['depth_sensor']
                if self.cfg['navigability_mode'] == 'depth_estimate':
                    depth_image = self.depth_estimator.call(obs['color_sensor'])
                    
                sensor_state = agent_state.sensor_states['color_sensor']
                goal_position, goal_mask = self._get_goal_position_from_pixels(
                    px, py, agent_state, sensor_state, depth_image
                )

                if goal_position is not None:
                    self.goal_mask = goal_mask
                    self.goal_position.append(goal_position)
                    self.anchor_target_position(goal_position)
                    logging.info(f"[Goal Updated] Target position updated. Total observations: {len(self.goal_position)}")

        # 2. 距离计算模块
        if self.known_target_coords is not None:
            current_pos_2d = np.array([agent_state.position[0], agent_state.position[2]])
            goal_pos_2d = np.array([self.known_target_coords[0], self.known_target_coords[2]])
            
            distance_to_goal = np.linalg.norm(current_pos_2d - goal_pos_2d)
            logging.info(f"[Termination Check] Target Locked. Dist: {distance_to_goal:.2f}m")
            
            dist_str = f"Dist to Target: {distance_to_goal:.2f}m"
            cv2.putText(images['goal_image'], "GOAL LOCKED", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(images['goal_image'], dist_str, (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # 3. 停止判定
        threshold_dist = self.cfg.get('success_threshold', 1.0)
        
        # 连续粗判计数
        if step_metadata.get('called_stopping', False):
            self._consecutive_stop_calls += 1
            logging.info(f"[StopModule] Consecutive stop calls: {self._consecutive_stop_calls}")
        else:
            self._consecutive_stop_calls = 0

        if self._consecutive_stop_calls >= 2:
            # 连续2次粗判（距离<1.5m）都认为到了，强制停止
            logging.info(
                f"SUCCESS! Consecutive stop calls reached {self._consecutive_stop_calls}. "
                f"Dist: {distance_to_goal:.2f}m. Force STOPPING."
            )
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            
        elif self.known_target_coords is not None and distance_to_goal < threshold_dist:
            # 精确距离达标，立刻停止
            logging.info(
                f"SUCCESS! Reached target coordinates. "
                f"Dist: {distance_to_goal:.2f}m < {threshold_dist}m. STOPPING."
            )
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            
        else:
            # 继续选路
            step_metadata, prompt_logging, action_response = self._prompting(
                goal, a_final, images, step_metadata, obs['subtask']
            )
            logging_data.update(prompt_logging)
            agent_action = self._action_number_to_polar(
                step_metadata['action_number'], list(a_final)
            )

        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _predicting_module(self, evaluator_image, goal):
        logging.info(f"[PredictVLM] Goal received: '{goal}'")
        """Determines if the agent should stop."""
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)

        return dct

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
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        # ── 模块二/三 prompt override 注入 ──
        _extra_predict_prompt = ""
        if hasattr(self, '_room_exhausted') and self._room_exhausted:
            _extra_predict_prompt += self.get_exhaustion_prompt_override()
        if hasattr(self, 'stuck_counter'):
            _stuck_level = self.detect_stuck()
            if _stuck_level == 1:
                _extra_predict_prompt += self.get_stuck_prompt_override(_stuck_level)
        _orig_sys = getattr(self.PredictVLM, 'system_instruction', '') or ''
        if _extra_predict_prompt:
            self.PredictVLM.system_instruction = _orig_sys + _extra_predict_prompt
        response = self._predicting_module(inference_image, goal)
        if _extra_predict_prompt:
            self.PredictVLM.system_instruction = _orig_sys

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
    def _merge_evalue(arr, num, alpha=0.4):
        """加权平均，允许历史低分被新高分拉回"""
        # 原来：return np.minimum(arr, num)  # 只降不升
        return arr * (1 - alpha) + num * alpha

    def update_curiosity_value(self, explorable_value, reason):
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i-2)*30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i+2)*30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]

                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2

                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(
                    self.cvalue_map[mask_minus_intersection], explorable_value[angle]
                )
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(
                    self.cvalue_map[intersection2],
                    (explorable_value[angle] + explorable_value[next_angle]) / 2
                )

            if self.goal_mask is not None:
                self.cvalue_map[self.goal_mask] = 10.0

            # ===== 改进：融合当前VLM分数和历史地图分数 =====
            current_weight = 0.4  # 当前VLM分数权重
            history_weight = 0.6  # 历史地图分数权重

            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                current_score = explorable_value[angle]

                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = current_score
                else:
                    history_score = np.mean(self.cvalue_map[self.panoramic_mask[angle]])
                    final_score[i] = current_weight * current_score + history_weight * history_score

            # ===== 新增调试日志 =====
            logging.info(f"[CuriosityValue] Raw VLM scores: {explorable_value}")
            logging.info(f"[CuriosityValue] Final scores: { {k: round(v, 2) for k, v in final_score.items()} }")

            # ── 模块三: 全局 recovery 方向强制 ──
            if hasattr(self, '_recovery_active') and self._recovery_active and self.recovery_8_dirs:
                _rec_dir = self.get_recovery_direction()
                if _rec_dir is not None:
                    _rec_idx = int(round(_rec_dir / 30)) % 12
                    if _rec_idx in final_score:
                        final_score[_rec_idx] = 100.0

            idx = max(final_score, key=final_score.get)
            logging.info(f"[CuriosityValue] Chosen: {idx} ({idx*30}deg), score={final_score[idx]:.2f}")
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        cvalue_map = (self.cvalue_map / 10 * 255).astype(np.uint8)


        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = cvalue_map.shape[1], cvalue_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = cvalue_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

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


    # ═══════════════════════════════════════════════════════════════════
    # 模块一: 语义记忆与先验引导
    # ═══════════════════════════════════════════════════════════════════

    def _vlm_text_only(self, vlm, prompt: str) -> str:
        """
        统一的纯文本 VLM 调用入口。
        优先使用 call_text_only()，若不存在则用占位图调用 call()。
        """
        if hasattr(vlm, 'call_text_only'):
            return vlm.call_text_only(prompt)
        else:
            _placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
            return vlm.call([_placeholder], prompt)

    def acquire_semantic_prior(self, goal: str):
        """
        任务初始化时调用 LLM 获取目标的预期场景列表。
        仅用于模块二的场景匹配判断，不注入 PredictVLM（VLM 看图时已有常识）。
        prompt 要求简短回复（每列表最多5项），避免被 max_tokens 截断。
        """
        prior_prompt = (
            f"A robot must find a \"{goal}\" in a home. "
            f"Return JSON: "
            f'{{\"likely_rooms\": [up to 5 room types where {goal} is usually found], '
            f'\"unlikely_rooms\": [up to 5 room types where {goal} is rarely found]}}. '
            f"Keep it SHORT. No markdown, no explanation, ONLY the JSON object."
        )
        try:
            raw_response = self._vlm_text_only(self.PlanVLM, prior_prompt)
            self.target_scene_prior = raw_response.strip()
            self._parse_prior_json(raw_response)
        except Exception as e:
            logging.warning(f"[SemanticPrior] Failed to acquire prior: {e}")
            self.target_scene_prior = ""
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []

        if self.prior_likely_rooms or self.prior_unlikely_rooms:
            logging.info(f"[SemanticPrior] Likely: {self.prior_likely_rooms}")
            logging.info(f"[SemanticPrior] Unlikely: {self.prior_unlikely_rooms}")
        else:
            logging.warning(f"[SemanticPrior] No prior acquired for \'{goal}\'")

    def _parse_prior_json(self, raw_response: str):
        """
        从 LLM 响应中提取 likely_rooms / unlikely_rooms。
        鲁棒处理：markdown fence、被截断的 JSON、单引号、多余文本。
        """
        import re as _re
        text = raw_response.strip()

        # 1) 去除 markdown code fence
        text = _re.sub(r'^```(?:json)?\s*', '', text)
        text = _re.sub(r'\s*```$', '', text)
        text = text.strip()

        # 2) 提取 JSON 块
        brace_start = text.find('{')
        if brace_start == -1:
            logging.warning(f"[SemanticPrior] No JSON found in: {raw_response[:150]}")
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []
            return

        # 寻找匹配的闭合大括号
        depth = 0
        brace_end = -1
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break

        if brace_end != -1:
            # 正常闭合
            text = text[brace_start:brace_end + 1]
        else:
            # JSON 被截断（没有闭合括号）→ 尝试修复
            text = text[brace_start:]
            logging.warning(f"[SemanticPrior] JSON truncated, attempting repair...")
            # 去掉最后一个不完整的元素（截断处通常在引号中间）
            # 策略：找最后一个完整的引号字符串后截断
            last_good = max(text.rfind('",'), text.rfind("',"))
            if last_good > 0:
                text = text[:last_good + 1]
            # 补齐未闭合的括号
            open_sq = text.count('[') - text.count(']')
            open_br = text.count('{') - text.count('}')
            text += ']' * max(open_sq, 0) + '}' * max(open_br, 0)

        # 3) 多轮尝试解析
        data = None
        # 尝试1: 直接 json.loads
        try:
            data = _json.loads(text)
        except:
            pass
        # 尝试2: 单引号转双引号
        if data is None:
            try:
                data = _json.loads(text.replace("'", '"'))
            except:
                pass
        # 尝试3: ast.literal_eval
        if data is None:
            try:
                data = ast.literal_eval(text)
            except:
                pass

        if data is None:
            logging.warning(f"[SemanticPrior] Cannot parse prior JSON: {raw_response[:200]}")
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []
            return

        if isinstance(data, dict):
            self.prior_likely_rooms = list(set(
                str(r).strip().lower() for r in data.get('likely_rooms', []) if str(r).strip()
            ))
            self.prior_unlikely_rooms = list(set(
                str(r).strip().lower() for r in data.get('unlikely_rooms', []) if str(r).strip()
            ))
        else:
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []

    # ═══════════════════════════════════════════════════════════════════
    # 模块二: 工作记忆与局部拓扑
    # ═══════════════════════════════════════════════════════════════════

    def _compute_local_cvalue_stats(self, agent_state):
        agent_coords = self._global_to_grid(agent_state.position)
        radius_px = int(2.5 * self.scale)
        x, y = agent_coords
        h, w = self.cvalue_map.shape[:2]
        y1, y2 = max(0, y - radius_px), min(h, y + radius_px)
        x1, x2 = max(0, x - radius_px), min(w, x + radius_px)

        region_cvalue = self.cvalue_map[y1:y2, x1:x2]
        vals = region_cvalue[:, :, 0].astype(np.float32)
        explored_mask = vals < 9.9
        if explored_mask.sum() == 0:
            return 10.0, 0.0
        mean_cvalue = float(np.mean(vals[explored_mask]))

        region_explored = self.explored_map[y1:y2, x1:x2]
        explored_pixels = np.all(region_explored == self.explored_color, axis=-1).sum()
        total_pixels = max((y2 - y1) * (x2 - x1), 1)
        coverage = explored_pixels / total_pixels
        return mean_cvalue, coverage

    def check_room_exhaustion(self, agent_state):
        mean_cv, coverage = self._compute_local_cvalue_stats(agent_state)
        self._room_exhausted = (mean_cv < 3.0 and coverage > 0.6)
        if self._room_exhausted:
            logging.info(f"[WorkingMemory] Room exhausted: mean_cvalue={mean_cv:.2f}, coverage={coverage:.2f}")
        return self._room_exhausted

    def get_exhaustion_prompt_override(self):
        if not self._room_exhausted:
            return ""
        return (
            "\n[OVERRIDE] The current room/area has been thoroughly searched and "
            "the target was NOT found here. You MUST prioritize directions that lead "
            "OUT of this room — look for doors, hallways, corridors, or openings to "
            "other areas. Assign score >= 8 to any exit direction and score 0 to "
            "directions deeper into this already-explored room."
        )

    def detect_scene_type_from_vlm(self, image, goal):
        scene_prompt = (
            f"Look at this image of an indoor environment. "
            f"What type of room or area is this? "
            f"Reply with ONLY a short room type label (e.g., bedroom, kitchen, bathroom, "
            f"living room, hallway, closet, garage, office, laundry room, pantry, mudroom, "
            f"sunroom, guest room, workshop, utility room, dining room, etc.). "
            f"If uncertain, reply with your best guess. ONE label only, no extra text."
        )
        try:
            resp = self.PlanVLM.call([image], scene_prompt)
            self.current_scene_type = resp.strip().lower()
            logging.info(f"[WorkingMemory] Detected scene type: {self.current_scene_type}")
        except Exception as e:
            logging.warning(f"[WorkingMemory] Scene detection failed: {e}")
            self.current_scene_type = "unknown"

    def _is_scene_mismatch(self):
        if self.current_scene_type == "unknown":
            return False
        scene = self.current_scene_type

        if self.prior_likely_rooms or self.prior_unlikely_rooms:
            for likely in self.prior_likely_rooms:
                if self._semantic_room_match(scene, likely):
                    logging.info(f"[Mismatch L1] '{scene}' matches likely '{likely}' -> OK")
                    return False
            for unlikely in self.prior_unlikely_rooms:
                if self._semantic_room_match(scene, unlikely):
                    logging.info(f"[Mismatch L1] '{scene}' matches unlikely '{unlikely}' -> MISMATCH")
                    return True

        return self._llm_judge_scene_match(scene)

    @staticmethod
    def _semantic_room_match(scene_type: str, room_label: str) -> bool:
        s = scene_type.lower().strip()
        r = room_label.lower().strip()
        if s in r or r in s:
            return True

        _synonym_groups = [
            {"living room", "lounge", "family room", "sitting room", "front room", "den"},
            {"bathroom", "washroom", "restroom", "toilet", "powder room", "lavatory", "bath"},
            {"bedroom", "master bedroom", "guest room", "guest bedroom", "sleeping room"},
            {"kitchen", "kitchenette"},
            {"dining room", "dining area", "breakfast room", "breakfast nook", "dinette"},
            {"hallway", "corridor", "hall", "passage", "passageway", "entryway", "foyer", "vestibule", "lobby"},
            {"closet", "wardrobe", "storage room", "walk-in closet", "linen closet", "storage closet"},
            {"garage", "carport", "car garage"},
            {"office", "study", "home office", "workroom", "workspace"},
            {"laundry", "laundry room", "utility room", "mudroom", "mud room"},
            {"basement", "cellar"},
            {"attic", "loft", "attic space"},
            {"nursery", "baby room", "children's room", "kids room", "playroom", "kid's room"},
            {"patio", "terrace", "deck", "veranda", "porch", "lanai"},
            {"balcony", "terrace", "juliet balcony"},
            {"pantry", "larder", "food storage", "butler's pantry"},
            {"sunroom", "conservatory", "solarium", "sun porch", "sun room"},
            {"workshop", "workbench area", "tool room", "tool shed"},
            {"rec room", "recreation room", "game room", "bonus room", "man cave", "media room"},
        ]
        for group in _synonym_groups:
            s_in = any(s in name or name in s for name in group)
            r_in = any(r in name or name in r for name in group)
            if s_in and r_in:
                return True
        return False

    def _llm_judge_scene_match(self, scene_type: str) -> bool:
        context_parts = []
        if self.target_scene_prior:
            context_parts.append(f"Background knowledge: {self.target_scene_prior}")
        if self.prior_likely_rooms:
            context_parts.append(f"Likely rooms: {', '.join(self.prior_likely_rooms)}")
        if self.prior_unlikely_rooms:
            context_parts.append(f"Unlikely rooms: {', '.join(self.prior_unlikely_rooms)}")
        if not context_parts:
            return False

        context = "\n".join(context_parts)
        judge_prompt = (
            f"A robot is searching for a specific target object inside a home.\n"
            f"{context}\n"
            f"The robot just entered a room identified as: \"{scene_type}\".\n"
            f"Is it VERY UNLIKELY that the target object would be found in a \"{scene_type}\"?\n"
            f"Answer ONLY \"yes\" or \"no\":\n"
            f"  yes = very unlikely here, robot should leave\n"
            f"  no  = might be here, or not sure"
        )
        try:
            resp = self._vlm_text_only(self.PlanVLM, judge_prompt).strip().lower()
            is_mismatch = resp.startswith("yes") or ("yes" in resp and "no" not in resp)
            logging.info(f"[Mismatch L2] LLM judge: scene='{scene_type}', mismatch={is_mismatch}, raw='{resp[:80]}'")
            return is_mismatch
        except Exception as e:
            logging.warning(f"[Mismatch L2] LLM judge failed: {e}")
            return False

    def check_scene_mismatch_and_block(self, agent_state):
        """
        场景不匹配处理（重构版）：
        不直接封杀当前区域，而是：
        1. 降低当前已探索区域的 cvalue（降权但不清零，仍允许路过）
        2. 提高所有未探索方向的 cvalue（引导去找预期场景）
        3. 如果有门记忆，优先引导回到门的方向（去其他房间找预期场景）
        """
        if self.steps_since_door_entry < 3:
            return False
        if not self._is_scene_mismatch():
            return False

        logging.info(
            f"[WorkingMemory] Scene MISMATCH: current='{self.current_scene_type}', "
            f"expected={self.prior_likely_rooms}. Prioritizing unexplored areas."
        )
        self._room_mismatch_blocked = True

        # 1) 降低当前区域 cvalue（降权，不是清零）
        agent_coords = self._global_to_grid(agent_state.position)
        radius_px = int(2.0 * self.scale)
        x, y = agent_coords
        h, w = self.cvalue_map.shape[:2]
        y1, y2 = max(0, y - radius_px), min(h, y + radius_px)
        x1, x2 = max(0, x - radius_px), min(w, x + radius_px)
        # 将已探索的当前区域 cvalue 减半（最低到 1.0），而非直接置 0.1
        region = self.cvalue_map[y1:y2, x1:x2].astype(np.float32)
        region = np.maximum(region * 0.5, 1.0)
        self.cvalue_map[y1:y2, x1:x2] = region.astype(np.float16)

        # 2) 提高未探索方向的 cvalue（引导 agent 离开去寻找预期场景）
        for angle_key, mask in self.panoramic_mask.items():
            if not np.any(mask):
                continue
            # 检查该方向的 explored_map 覆盖率
            direction_explored = self.explored_map[mask]
            explored_count = np.all(direction_explored == self.explored_color, axis=-1).sum()
            total_count = max(mask.sum(), 1)
            explore_ratio = explored_count / total_count
            # 未充分探索的方向 → 提高 cvalue
            if explore_ratio < 0.4:
                current_vals = self.cvalue_map[mask].astype(np.float32)
                boosted = np.minimum(current_vals + 3.0, 10.0)
                self.cvalue_map[mask] = boosted.astype(np.float16)
                logging.info(
                    f"[WorkingMemory] Boosted unexplored direction {angle_key}deg "
                    f"(explore_ratio={explore_ratio:.2f})"
                )

        # 3) 如果有门记忆，强力引导回到最近的门方向（去其他房间）
        if self.door_memory:
            last_door = self.door_memory[-1]
            door_grid = self._global_to_grid(
                np.array([last_door[0], agent_state.position[1], last_door[1]])
            )
            dr = int(1.5 * self.scale)
            dy1, dy2 = max(0, door_grid[1] - dr), min(h, door_grid[1] + dr)
            dx1, dx2 = max(0, door_grid[0] - dr), min(w, door_grid[0] + dr)
            self.cvalue_map[dy1:dy2, dx1:dx2] = 10.0
            logging.info("[WorkingMemory] Door direction boosted to 10.0")

        return True

    def record_door_position(self, agent_state):
        pos = agent_state.position
        self.door_memory.append((float(pos[0]), float(pos[2])))
        self.steps_since_door_entry = 0
        self._room_mismatch_blocked = False
        logging.info(f"[WorkingMemory] Door recorded at ({pos[0]:.2f}, {pos[2]:.2f})")

    # ═══════════════════════════════════════════════════════════════════
    # 模块三: 空间记忆与鲁棒兜底
    # ═══════════════════════════════════════════════════════════════════

    def anchor_target_position(self, global_goal_position):
        self.known_target_coords = np.array(global_goal_position, dtype=np.float64)
        logging.info(f"[SpatialMemory] Target anchored at {self.known_target_coords}")

    def apply_target_anchor_bias(self, agent_state):
        """
        障碍物感知的目标锚定偏置：
        
        原方法：直接给目标正方向 cvalue=10，不管该方向是否有墙
        新方法：从目标方向开始，向两侧扫描可通行方向，给最优方向加分
        
        搜索逻辑：
        目标正方向 → 左偏30° → 右偏30° → 左偏60° → 右偏60° → ...
        评分逻辑：
        偏移0° → 10分, 偏移30° → 8分, 偏移60° → 6分, 偏移90° → 4分
        """
        if self.known_target_coords is None:
            return
        
        dx = self.known_target_coords[0] - agent_state.position[0]
        dz = self.known_target_coords[2] - agent_state.position[2]
        dist = np.sqrt(dx * dx + dz * dz)
        if dist < 0.3:
            return

        # 计算目标相对于 agent 前方的角度
        target_angle_world = np.arctan2(dx, -dz)
        forward = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)
        agent_angle_world = np.arctan2(forward[0], -forward[2])
        relative_yaw = target_angle_world - agent_angle_world
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi

        # 映射到最近的30°方向索引 (0~11)
        target_dir_idx = int(round(np.degrees(relative_yaw) / 30)) % 12
        target_angle_key = str(target_dir_idx * 30)

        # 从目标方向开始，向两侧搜索可通行方向
        # 最大搜索范围 ±90°（3格）
        search_offsets = [0, 1, -1, 2, -2, 3, -3]
        
        chosen_key = None
        chosen_score = 0.0

        for offset in search_offsets:
            check_idx = (target_dir_idx + offset) % 12
            angle_key = str(check_idx * 30)

            # 检查该方向是否存在可通行区域
            if angle_key not in self.panoramic_mask:
                continue
            if not np.any(self.panoramic_mask[angle_key]):
                continue
            
            # 进一步检查：该方向是否有有效通行距离（effective_mask）
            has_effective_path = (
                angle_key in self.effective_mask 
                and np.any(self.effective_mask[angle_key])
            )
            
            if has_effective_path:
                # 偏离目标方向越少，分数越高
                chosen_score = max(10.0 - abs(offset) * 2.0, 4.0)
                chosen_key = angle_key
                break
        
        # 如果 effective_mask 都不通，退而求其次用 panoramic_mask
        if chosen_key is None:
            for offset in search_offsets:
                check_idx = (target_dir_idx + offset) % 12
                angle_key = str(check_idx * 30)
                if angle_key in self.panoramic_mask and np.any(self.panoramic_mask[angle_key]):
                    chosen_score = max(10.0 - abs(offset) * 2.0, 4.0)
                    chosen_key = angle_key
                    break

        # 应用偏置
        if chosen_key is not None:
            mask = self.panoramic_mask[chosen_key]
            self.cvalue_map[mask] = np.maximum(
                self.cvalue_map[mask], chosen_score
            )
            
            if chosen_key != target_angle_key:
                logging.info(
                    f"[SpatialMemory] Target dir {target_angle_key}° BLOCKED, "
                    f"rerouted to {chosen_key}° (score={chosen_score:.1f}, dist={dist:.2f}m)"
                )
            else:
                logging.info(
                    f"[SpatialMemory] Target anchor bias -> {chosen_key}° "
                    f"(score={chosen_score:.1f}, dist={dist:.2f}m)"
                )
        else:
            logging.warning(
                f"[SpatialMemory] No navigable direction found near target "
                f"(target_dir={target_angle_key}°, dist={dist:.2f}m)"
            )

    def update_position_history(self, agent_state):
        pos = np.array([agent_state.position[0], agent_state.position[2]])
        self.position_history.append(pos)

    def detect_stuck(self, threshold=0.15, window=3):
        if len(self.position_history) < window + 1:
            return 0
        positions = list(self.position_history)
        displacements = [np.linalg.norm(positions[i] - positions[i - 1]) for i in range(-window, 0)]

        if all(d < threshold for d in displacements):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self._recovery_active = False
            self.recovery_8_dirs = []
            return 0

        if self.stuck_counter >= 6:
            return 2
        elif self.stuck_counter >= 1:
            return 1
        return 0

    def get_stuck_prompt_override(self, stuck_level):
        if stuck_level == 1:
            return (
                "\n[STUCK OVERRIDE] The robot appears stuck. It has barely moved for several steps. "
                "Prioritize the direction with the LARGEST open/connected area. "
                "Assign score >= 9 to the most open direction and 0 to blocked directions."
            )
        return ""

    def init_global_recovery(self):
        if not self._recovery_active:
            self._recovery_active = True
            self.recovery_8_dirs = list(range(0, 360, 45))
            logging.info("[Recovery] Global recovery initiated: 8-direction sweep")

    def get_recovery_direction(self):
        if self.recovery_8_dirs:
            return self.recovery_8_dirs.pop(0)
        return None

    def apply_recovery_direction_to_cvalue(self, agent_state, direction_deg):
        dir_idx = int(round(direction_deg / 30)) % 12
        angle_key = str(dir_idx * 30)
        if angle_key in self.panoramic_mask:
            mask = self.panoramic_mask[angle_key]
            if np.any(mask):
                self.cvalue_map[mask] = 10.0
                logging.info(f"[Recovery] Forced {direction_deg}deg -> {angle_key}deg cvalue=10")
                return True
        return False

    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', reason: str='{}', num_actions: int=0):
        if prompt_type == 'goal':
            location_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                f"First, tell me whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}. "
                f"Second, if there is a {goal} in the image, provide the 2D pixel coordinates [x, y] of the center of the {goal} in the image. "
                f"Assume the image width is {self.resolution[1]} and height is {self.resolution[0]}. (x is the horizontal axis from 0 to {self.resolution[1]}, y is the vertical axis from 0 to {self.resolution[0]}). "
                f"If you are certain the {goal} is NOT visible in the image at all, set 'Visible' to False and provide a short, clear reason why (e.g., 'The image only shows a blank wall', 'This is a kitchen with no bed'). "
                "Format your answer as a JSON dict strictly like this: {'Visible': True, 'Coordinates': [x, y], 'Reason': ''} or {'Visible': False, 'Coordinates': [], 'Reason': '<short explanation>'}."
            )
            return location_prompt
        if prompt_type == 'predicting':
            evaluator_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
            f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
            f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
            f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f'(2) If you can ACTUALLY SEE a {goal} in the image for that direction (not just guessing it might exist nearby), assign a score of 10. '
            f'CRITICAL: If your explanation describes a {goal} as physically visible in that direction (e.g., "there is a chair", "a chair is visible", "with a chair"), you MUST give a score of 10. '
            f'It is WRONG to describe seeing a {goal} but give a score less than 10. '
            f'However, do NOT give 10 for mere speculation like "a chair might be in the adjacent room" — that is just a guess, not actual detection. '
            f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
            "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
            "Answer Example: {'30': {'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Explanation': 'Living room area with a recliner.  Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                f"(1)<The observed image>: The image taken from its current location. "
                f"(2){reason}. This explains why you should go in this direction. "
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed.'
                f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            else:
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be goal, predicting, planning, or action')