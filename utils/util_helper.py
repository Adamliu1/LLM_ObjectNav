from collections import deque
import numpy as np
import torch
from skimage import measure
from torch import nn as nn


def find_big_connect(image):
    img_label, num = measure.label(
        image, connectivity=2, return_num=True
    )  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
    # print("img_label.shape: ", img_label.shape) # 480*480
    resMatrix = np.zeros(img_label.shape)
    tmp_area = 0
    for i in range(0, len(props)):
        if props[i].area > tmp_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix = tmp
            tmp_area = props[i].area

    return resMatrix


def prepare_planner_inputs(
    args,
    finished,
    goal_maps,
    local_map,
    num_scenes,
    planner_pose_inputs,
    target_edge_map,
    target_point_map,
    wait_env,
):
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
        p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
        p_input["pose_pred"] = planner_pose_inputs[e]
        p_input["goal"] = goal_maps[e]  # global_goals[e]
        p_input["map_target"] = target_point_map[e]  # global_goals[e]
        p_input["new_goal"] = 1
        p_input["found_goal"] = 0
        p_input["wait"] = wait_env[e] or finished[e]
        if args.visualize or args.print_images:
            p_input["map_edge"] = target_edge_map[e]
            local_map[e, -1, :, :] = 1e-5
            p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
    return planner_inputs


def random_action_goals(local_h, local_w, num_scenes):
    actions = torch.randn(num_scenes, 2) * 6
    # print("actions: ", actions.shape)
    cpu_actions = nn.Sigmoid()(actions).cpu().numpy()
    global_goals = [
        [int(action[0] * local_w), int(action[1] * local_h)] for action in cpu_actions
    ]
    global_goals = [
        [min(x, int(local_w - 1)), min(y, int(local_h - 1))] for x, y in global_goals
    ]
    return global_goals


def get_sensor_data(device, infos, num_scenes):
    poses = (
        torch.from_numpy(
            np.asarray([infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)])
        )
        .float()
        .to(device)
    )
    eve_angle = np.asarray(
        [infos[env_idx]["eve_angle"] for env_idx in range(num_scenes)]
    )
    return eve_angle, poses


def load_object_norm_inv_perplexity(device):
    object_norm_inv_perplexity = torch.tensor(
        np.load("data/object_norm_inv_perplexity.npy")
    ).to(device)
    return object_norm_inv_perplexity


def initialize_planner_inputs_and_scores(args, num_scenes):
    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))
    frontier_score_list = []
    for _ in range(args.num_processes):
        frontier_score_list.append(deque(maxlen=10))
    return frontier_score_list, planner_pose_inputs
