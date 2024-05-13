import numpy as np
import torch

from envs import make_vec_envs
from utils.mapping.l3mvn_mapping import L3MVNMapManager
from utils.mapping.llm_mapping import LLMAgentMapManager
from utils.util_helper import get_sensor_data

from utils.llm.l3mvn_zeroshot_llm_service import (
    L3MVN_LanguageModelService_ZeroShot,
)
from constants import category_to_id

from utils.llm.l3mvn_ffn_llm_service import setup_language_model_and_ff_net
from utils.llm.llm_zeroshot_service import LanguageModelService_ZeroShot
from utils.map_helper import (
    setup_semantic_mapping_module,
    get_local_map_boundaries,
)
from utils.navigation.l3mvn_zeroshot_nav_manager import (
    L3MVN_Zeroshot_NavigationManager,
)
from utils.navigation.l3mvn_ffn_nav_manager import L3MVN_FFN_NavigationManager
from utils.navigation.llm_zeroshot_nav_manager import (
    LLM_Zeroshot_NavigationManager,
)
from utils.navigation_helper import NavigationManager


class SimulationContext:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_scenes = args.num_processes
        self.num_episodes = int(args.num_eval_episodes)
        self.finished = np.zeros((args.num_processes))
        self.wait_env = np.zeros((args.num_processes))
        self.g_process_rewards = 0
        self.g_total_rewards = np.ones((self.num_scenes))
        self.g_sum_rewards = 1
        self.g_sum_global = 1
        self.stair_flag = np.zeros((self.num_scenes))
        self.clear_flag = np.zeros((self.num_scenes))
        self.replan_flag = np.zeros((self.num_scenes))
        self.sem_map_module = setup_semantic_mapping_module(args, device)
        if args.agent == "l3mvn_zeroshot":
            lm_service = L3MVN_LanguageModelService_ZeroShot(args.llm_name, device)
            self.map_manager = L3MVNMapManager(args, device, self.num_scenes)
            self.navigation_manager = L3MVN_Zeroshot_NavigationManager(
                self.map_manager, lm_service, args, device
            )
        elif args.agent == "l3mvn_ffn":
            lm_service, ff_net = setup_language_model_and_ff_net(
                args, device, category_to_id
            )
            self.map_manager = L3MVNMapManager(args, device, self.num_scenes)
            self.navigation_manager = L3MVN_FFN_NavigationManager(
                self.map_manager, lm_service, ff_net, args, device
            )
        elif args.agent == "llm_sem_agent":
            lm_service = LanguageModelService_ZeroShot(args.llm_name, device)
            self.map_manager = LLMAgentMapManager(args, device, self.num_scenes)
            self.navigation_manager = LLM_Zeroshot_NavigationManager(
                self.map_manager, lm_service, args, device
            )

        else:
            self.navigation_manager = NavigationManager(self.map_manager, args, device)
            self.map_manager = L3MVNMapManager(args, device, self.num_scenes)
        # Additional initializations as needed
        torch.set_grad_enabled(False)
        self.envs, self.infos, self.obs = self._setup_environment()
        self.full_pose, self.lmb, self.local_pose, self.origins = (
            self._initialize_poses_and_boundaries()
        )

    def _setup_environment(self):
        # Starting environments
        torch.set_num_threads(1)
        envs = make_vec_envs(self.args)
        obs, infos = envs.reset()
        return envs, infos, obs

    def _initialize_poses_and_boundaries(self):
        # Initial full and local pose
        full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        # Origin of local map
        origins = np.zeros((self.num_scenes, 3))
        # Local Map Boundaries
        lmb = np.zeros((self.num_scenes, 4)).astype(int)

        self.map_manager.local_pose = local_pose

        return full_pose, lmb, local_pose, origins

    def init_map_and_pose(
        self,
        planner_pose_inputs,
    ):
        self.map_manager.full_map.fill_(0.0)
        self.full_pose.fill_(0.0)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.args.map_resolution),
                int(c * 100.0 / self.args.map_resolution),
            ]

            self.map_manager.full_map[
                e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2
            ] = 1.0

            self.lmb[e] = get_local_map_boundaries(
                self.args,
                (loc_r, loc_c),
                (self.map_manager.local_w, self.map_manager.local_h),
                (self.map_manager.full_w, self.map_manager.full_h),
            )

            planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [
                self.lmb[e][2] * self.args.map_resolution / 100.0,
                self.lmb[e][0] * self.args.map_resolution / 100.0,
                0.0,
            ]

        for e in range(self.num_scenes):
            self.map_manager.local_map[e] = self.map_manager.full_map[
                e,
                :,
                self.lmb[e, 0] : self.lmb[e, 1],
                self.lmb[e, 2] : self.lmb[e, 3],
            ]
            self.map_manager.local_pose[e] = (
                self.full_pose[e]
                - torch.from_numpy(self.origins[e]).to(self.device).float()
            )

    def init_map_and_pose_for_env(
        self,
        step_masks,
        planner_pose_inputs,
        e,
    ):
        local_w, local_h = self.map_manager.local_w, self.map_manager.local_h
        full_w, full_h = self.map_manager.full_w, self.map_manager.full_h

        self.map_manager.full_map[e].fill_(0.0)
        self.full_pose[e].fill_(0.0)
        self.map_manager.local_ob_map[e] = np.zeros((local_w, local_h))
        self.map_manager.local_ex_map[e] = np.zeros((local_w, local_h))
        self.map_manager.target_edge_map[e] = np.zeros((local_w, local_h))
        self.map_manager.target_point_map[e] = np.zeros((local_w, local_h))

        step_masks[e] = 0  # this is maybe not needed....
        self.stair_flag[e] = 0
        self.clear_flag[e] = 0
        self.replan_flag[e] = False  # NOTE: replace all these to boolean

        self.full_pose[e, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / self.args.map_resolution),
            int(c * 100.0 / self.args.map_resolution),
        ]

        self.map_manager.full_map[
            e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2
        ] = 1.0

        self.lmb[e] = get_local_map_boundaries(
            self.args, (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
        )

        planner_pose_inputs[e, 3:] = self.lmb[e]
        self.origins[e] = [
            self.lmb[e][2] * self.args.map_resolution / 100.0,
            self.lmb[e][0] * self.args.map_resolution / 100.0,
            0.0,
        ]

        self.map_manager.local_map[e] = self.map_manager.full_map[
            e,
            :,
            self.lmb[e, 0] : self.lmb[e, 1],
            self.lmb[e, 2] : self.lmb[e, 3],
        ]
        self.map_manager.local_pose[e] = (
            self.full_pose[e]
            - torch.from_numpy(self.origins[e]).to(self.device).float()
        )

    def pred_update_sem_map(self):
        eve_angle, poses = get_sensor_data(self.device, self.infos, self.num_scenes)

        self.map_manager.update_local_map(
            eve_angle, self.obs, poses, self.sem_map_module
        )

    def pred_update_sem_map_no_clean(self):
        eve_angle, poses = get_sensor_data(self.device, self.infos, self.num_scenes)

        self.map_manager.update_local_map_no_clean(
            eve_angle, self.obs, poses, self.sem_map_module
        )

    def plan_act_and_preprocess(self, planner_inputs):
        self.obs, fail_case, done, self.infos = self.envs.plan_act_and_preprocess(
            planner_inputs
        )
        return fail_case, done
