from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np

from sem_exp_baseline.model import RL_Policy, Semantic_Mapping
from sem_exp_baseline.utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import sem_exp_baseline.algo as algo
from utils.DataLogger import DataLogger


os.environ["OMP_NUM_THREADS"] = "1"


def initialize_seed_and_device(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    return device


def main():
    args = get_args()

    device = initialize_seed_and_device(args)

    wandb_logger = None
    if not args.wandb_not_log:
        from utils.wandb_logger import WandbLogger

        wandb_logger = WandbLogger(args)

    logger = DataLogger(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    # device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        pass
    else:
        logger.episode_success = deque(maxlen=1000)
        logger.episode_spl = deque(maxlen=1000)
        logger.episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

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
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
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

    def init_map_and_pose():
        full_map.fill_(0.0)
        full_pose.fill_(0.0)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution),
            ]

            full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries(
                (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
            )

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [
                lmb[e][2] * args.map_resolution / 100.0,
                lmb[e][0] * args.map_resolution / 100.0,
                0.0,
            ]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
            local_pose[e] = (
                full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
            )

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.0)
        full_pose[e].fill_(0.0)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
        )

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [
            lmb[e][2] * args.map_resolution / 100.0,
            lmb[e][0] * args.map_resolution / 100.0,
            0.0,
        ]

        local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
        local_pose[e] = full_pose[e] - torch.from_numpy(origins[e]).to(device).float()

    def update_intrinsic_rew(e):
        prev_explored_area = full_map[e, 1].sum(1).sum(0)
        full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.0) ** 2  # to m^2

    init_map_and_pose()

    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    es = 2
    g_observation_space = gym.spaces.Box(0, 1, (ngc, local_w, local_h), dtype="uint8")

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy
    g_policy = RL_Policy(
        g_observation_space.shape,
        g_action_space,
        model_type=1,
        base_kwargs={
            "recurrent": args.use_recurrent_global,
            "hidden_size": g_hidden_size,
            "num_sem_categories": ngc - 8,
        },
    ).to(device)
    # print(g_policy)

    g_agent = algo.PPO(
        g_policy,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, 2)

    # Storage
    g_rollouts = GlobalRolloutStorage(
        args.num_global_steps,
        num_scenes,
        g_observation_space.shape,
        g_action_space,
        g_policy.rec_state_size,
        es,
    ).to(device)

    if args.load != "0":
        print("Loading model {}".format(args.load))
        state_dict = torch.load(args.load, map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = (
        torch.from_numpy(
            np.asarray([infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)])
        )
        .float()
        .to(device)
    )

    _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        local_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :]
    )
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
    goal_cat_id = torch.from_numpy(
        np.asarray([infos[env_idx]["goal_cat_id"] for env_idx in range(num_scenes)])
    )

    extras = torch.zeros(num_scenes, 2)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = g_policy.act(
        g_rollouts.obs[0],
        g_rollouts.rec_states[0],
        g_rollouts.masks[0],
        extras=g_rollouts.extras[0],
        deterministic=False,
    )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [
        [int(action[0] * local_w), int(action[1] * local_h)] for action in cpu_actions
    ]
    global_goals = [
        [min(x, int(local_w - 1)), min(y, int(local_h - 1))] for x, y in global_goals
    ]

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
        p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
        p_input["pose_pred"] = planner_pose_inputs[e]
        p_input["goal"] = goal_maps[e]  # global_goals[e]
        p_input["new_goal"] = 1
        p_input["found_goal"] = 0
        p_input["wait"] = wait_env[e] or finished[e]
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = infos[e]["spl"]
                success = infos[e]["success"]
                dist = infos[e]["distance_to_goal"]
                spl_per_category[infos[e]["goal_name"]].append(spl)
                success_per_category[infos[e]["goal_name"]].append(success)
                if args.eval:
                    logger.episode_success[e].append(success)
                    logger.episode_spl[e].append(spl)
                    logger.episode_dist[e].append(dist)
                    if len(logger.episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    logger.episode_success.append(success)
                    logger.episode_spl.append(spl)
                    logger.episode_dist.append(dist)
                wait_env[e] = 1.0
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = (
            torch.from_numpy(
                np.asarray(
                    [infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)]
                )
            )
            .float()
            .to(device)
        )

        _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution),
            ]
            local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.0
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = (
                    local_map[e]
                )
                full_pose[e] = (
                    local_pose[e] + torch.from_numpy(origins[e]).to(device).float()
                )

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / args.map_resolution),
                    int(c * 100.0 / args.map_resolution),
                ]

                lmb[e] = get_local_map_boundaries(
                    (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
                )

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [
                    lmb[e][2] * args.map_resolution / 100.0,
                    lmb[e][0] * args.map_resolution / 100.0,
                    0.0,
                ]

                local_map[e] = full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]
                local_pose[e] = (
                    full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
                )

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
                full_map[:, 0:4, :, :]
            )
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(
                np.asarray(
                    [infos[env_idx]["goal_cat_id"] for env_idx in range(num_scenes)]
                )
            )
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            # Get exploration reward and metrics
            g_reward = (
                torch.from_numpy(
                    np.asarray(
                        [infos[env_idx]["g_reward"] for env_idx in range(num_scenes)]
                    )
                )
                .float()
                .to(device)
            )
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input,
                    g_rec_states,
                    g_action,
                    g_action_log_prob,
                    g_value,
                    g_reward,
                    g_masks,
                    extras,
                )

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states = g_policy.act(
                g_rollouts.obs[g_step + 1],
                g_rollouts.rec_states[g_step + 1],
                g_rollouts.masks[g_step + 1],
                extras=g_rollouts.extras[g_step + 1],
                deterministic=False,
            )
            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [
                [int(action[0] * local_w), int(action[1] * local_h)]
                for action in cpu_actions
            ]
            global_goals = [
                [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                for x, y in global_goals
            ]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

        for e in range(num_scenes):
            cn = infos[e]["goal_cat_id"] + 4
            if local_map[e, cn, :, :].sum() != 0.0:
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = planner_pose_inputs[e]
            p_input["goal"] = goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = l_step == args.num_local_steps - 1
            p_input["found_goal"] = found_goal[e]
            p_input["wait"] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()

        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if (
            g_step % args.num_global_steps == args.num_global_steps - 1
            and l_step == args.num_local_steps - 1
        ):
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1],
                ).detach()

                g_rollouts.compute_returns(
                    g_next_value, args.use_gae, args.gamma, args.tau
                )
                g_value_loss, g_action_loss, g_dist_entropy = g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end_time = time.time()
            time_elapsed = time.gmtime(end_time - start)
            log = " ".join(
                [
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(step * num_scenes),
                    "FPS {},".format(int(step * num_scenes / (end_time - start))),
                ]
            )

            log += "\n\tRewards:"
            if not args.wandb_not_log:
                wandb_logger.log_evaluation(
                    data={
                        "num_timesteps": step * num_scenes,
                        "FPS": int(step * num_scenes / (end_time - start)),
                    },
                    step=step,
                    commit=False,
                )

            if len(g_episode_rewards) > 0:
                log += " ".join(
                    [
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards),
                        ),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards),
                        ),
                    ]
                )

            # total_collision = []
            # total_exploration = []
            # total_detection = []
            # total_fail_case_success = []

            if args.eval:
                (total_success, total_spl, total_dist) = logger.update_eval_metrics()

                # for e in range(args.num_processes):
                #     total_collision.append(fail_case[e]["collision"])
                #     total_exploration.append(fail_case[e]["exploration"])
                #     total_detection.append(fail_case[e]["detection"])
                #     total_fail_case_success.append(fail_case[e]["success"])
                episode_metrics = {
                    "total_success": total_success,
                    "total_spl": total_spl,
                    "total_dist": total_dist,
                    "total_collision": [],
                    "total_exploration": [],
                    "total_detection": [],
                    "total_fail_case_success": [],
                }

                logger.log_evaluation(
                    step,
                    num_scenes,
                    end_time,
                    g_process_rewards,
                    1,  # N/A for semexp
                    1,  # N/A for semexp
                    episode_metrics,
                )
                if not args.wandb_not_log:
                    # wandb_logger.log_metrics(step, episode_metrics)

                    log_data = {
                        "step": step,
                        "num_scenes": num_scenes,
                        "end_time": end_time,
                        "process_rewards": g_process_rewards,  # Assuming this is a list or a single value
                        # "sum_rewards": g_sum_rewards,
                        # "sum_global": g_sum_global,
                    }
                    wandb_logger.log_evaluation(data=log_data, step=step, commit=False)

                    # Log metrics for each process
                    metric_data = {
                        "total_success": (np.mean(episode_metrics["total_success"])),
                        "total_spl": (np.mean(episode_metrics["total_spl"])),
                        "total_dist": (np.mean(episode_metrics["total_dist"])),
                        "total_collision": (np.sum(episode_metrics["total_collision"])),
                        "total_exploration": (
                            np.sum(episode_metrics["total_exploration"])
                        ),
                        "total_detection": (np.sum(episode_metrics["total_detection"])),
                        "total_fail_case_success": (
                            np.sum(episode_metrics["total_fail_case_success"])
                        ),
                    }
                    # print(metric_data)
                    wandb_logger.log_evaluation(
                        data=metric_data, step=step, commit=True
                    )
            else:
                if len(logger.episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(logger.episode_success),
                        np.mean(logger.episode_spl),
                        np.mean(logger.episode_dist),
                        len(logger.episode_spl),
                    )

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join(
                    [
                        " Policy Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies),
                        ),
                    ]
                )

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < num_scenes:
            if (
                len(g_episode_rewards) >= 1000
                and (np.mean(g_episode_rewards) >= best_g_reward)
                and not args.eval
            ):
                torch.save(
                    g_policy.state_dict(),
                    os.path.join(logger.log_dir, "model_best.pth"),
                )
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(
                    g_policy.state_dict(),
                    os.path.join(
                        logger.dump_dir, "periodic_{}.pth".format(total_steps)
                    ),
                )
        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        (total_success, total_spl, total_dist) = logger.update_eval_metrics()

        logger.log_final_evaluation(
            g_process_rewards,
            1,  # N/A for semexp
            1,  # N/A for semexp
            total_success,
            total_spl,
            total_dist,
            success_per_category,
            spl_per_category,
        )


if __name__ == "__main__":
    main()
