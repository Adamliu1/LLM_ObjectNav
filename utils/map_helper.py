import torch
import numpy as np
import cv2
from model import Semantic_Mapping


def get_local_map_boundaries(args, agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def get_frontier_boundaries(frontier_loc, frontier_sizes, map_sizes):
    loc_r, loc_c = frontier_loc
    local_w, local_h = frontier_sizes
    full_w, full_h = map_sizes

    gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
    gx2, gy2 = gx1 + local_w, gy1 + local_h
    if gx1 < 0:
        gx1, gx2 = 0, local_w
    if gx2 > full_w:
        gx1, gx2 = full_w - local_w, full_w

    if gy1 < 0:
        gy1, gy2 = 0, local_h
    if gy2 > full_h:
        gy1, gy2 = full_h - local_h, full_h

    return [int(gx1), int(gx2), int(gy1), int(gy2)]


class MapManager:
    def __init__(self, args, device, num_scenes):
        self.args = args
        self.device = device
        self.num_scenes = num_scenes
        self.initialize_maps()

    def initialize_maps(self):
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        # Number of channels: obstacle map, explored area, current agent location, past agent locations, semantic categories
        nc = self.args.num_sem_categories + 4
        # Calculating full and local map sizes
        map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.args.global_downscaling)
        self.local_h = int(self.full_h / self.args.global_downscaling)
        # Initializing full and local maps
        self.full_map = (
            torch.zeros(self.num_scenes, nc, self.full_w, self.full_h)
            .float()
            .to(self.device)
        )
        self.local_map = (
            torch.zeros(self.num_scenes, nc, self.local_w, self.local_h)
            .float()
            .to(self.device)
        )
        self.local_ob_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.local_ex_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.target_edge_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.target_point_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        # Kernels for map dilation
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def update_local_map(self, eve_angle, obs, poses, sem_map_module):
        """
        Update the local map based on observations using the semantic mapping module.
        """
        (
            self.increase_local_map,
            self.local_map,
            self.local_map_stair,
            self.local_pose,
        ) = sem_map_module(obs, poses, self.local_map, self.local_pose, eve_angle)
        self.local_map[:, 0, :, :][self.local_map[:, 13, :, :] > 0] = 0

    # TODO: need to rethink this function
    def update_local_map_no_clean(self, eve_angle, obs, poses, sem_map_module):
        """
        Update the local map based on observations using the semantic mapping module.
        """
        (
            self.increase_local_map,
            self.local_map,
            self.local_map_stair,
            self.local_pose,
        ) = sem_map_module(obs, poses, self.local_map, self.local_pose, eve_angle)

    def create_global_goal_maps(self, global_goals):
        """
        Create goal maps for navigation tasks.
        """
        goal_maps = [
            np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)
        ]
        for e in range(self.num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
        return goal_maps

    # OTHER FUNCTIONS
    def get_agent_location_in_map(self, pose):
        r, c = pose[1], pose[0]
        loc_r = int(r * 100.0 / self.args.map_resolution)
        loc_c = int(c * 100.0 / self.args.map_resolution)
        return loc_r, loc_c

    def reset_current_location_and_mark_agent(self):
        self.local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(self.num_scenes):
            # NOTE: careful here maybe need to detach to cpu
            loc_r, loc_c = self.get_agent_location_in_map(self.local_pose[e])

            self.local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

    # GLOBAL POLICY
    # def calculate_map_boundaries(self, loc_r, loc_c):
    #     # Calculate local map boundaries based on agent's position This
    #     # method should return the boundaries (lmb) for the local map Adjust
    #     # the logic to calculate lmb based on your simulation requirements
    #     return [
    #         loc_r,
    #         loc_r + self.local_h,
    #         loc_c,
    #         loc_c + self.local_w,
    #     ]  # Simplified example

    def update_maps_and_poses_for_global_step(
        self, e, lmb, origins, device, full_pose, local_pose, clear_flag
    ):
        # Synchronize full and local maps at global steps
        self.full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = (
            self.local_map[e]
        )
        full_pose[e] = local_pose[e] + torch.from_numpy(origins[e]).to(device).float()

        # NOTE: here maybe need to detach to cpu
        loc_r, loc_c = self.get_agent_location_in_map(full_pose[e])
        # lmb[e] = self.calculate_map_boundaries(loc_r, loc_c)
        lmb[e] = get_local_map_boundaries(
            self.args,
            (loc_r, loc_c),
            (self.local_w, self.local_h),
            (self.full_w, self.full_h),
        )

    # Frontier stuff

    def clear_boundary_areas(self, map_array, e):
        # Clear the boundary areas of the map to avoid edge goals
        map_array[e, 0:2, 0 : self.local_w] = 0.0
        map_array[e, self.local_w - 2 : self.local_w, 0 : self.local_w - 1] = 0.0
        map_array[e, 0 : self.local_w, 0:2] = 0.0
        map_array[e, 0 : self.local_w, self.local_w - 2 : self.local_w] = 0.0

    def calculate_target_edges(self, e):
        # Calculate target edges for goal selection
        # target_edge = np.zeros((self.local_w, self.local_h))
        target_edge = self.local_ex_map[e] - self.local_ob_map[e]

        target_edge[target_edge > 0.8] = 1.0  # Considered as potential goal
        target_edge[target_edge != 1.0] = 0.0  # Non-goal areas

        return target_edge


def initialize_maps(args, device, num_scenes):
    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels
    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size  # 2400/5=480
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)
    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)
    local_ob_map = np.zeros((num_scenes, local_w, local_h))
    local_ex_map = np.zeros((num_scenes, local_w, local_h))
    target_edge_map = np.zeros((num_scenes, local_w, local_h))
    target_point_map = np.zeros((num_scenes, local_w, local_h))
    # dialate for target map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return (
        full_h,
        full_map,
        full_w,
        kernel,
        local_ex_map,
        local_h,
        local_map,
        local_ob_map,
        local_w,
        target_edge_map,
        target_point_map,
        tv_kernel,
    )


def create_global_goal_maps(global_goals, local_h, local_w, num_scenes):
    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
    return goal_maps


def setup_semantic_mapping_module(args, device):
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()
    return sem_map_module
