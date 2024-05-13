from collections import defaultdict
import os
import time
import torch
import numpy as np
import cv2

from arguments import get_args
from constants import hm3d_category

from utils.DataLogger import DataLogger

os.environ["OMP_NUM_THREADS"] = "1"

from utils.SimulationContext import SimulationContext
from utils.util_helper import (
    find_big_connect,
    get_sensor_data,
    initialize_planner_inputs_and_scores,
    prepare_planner_inputs,
    random_action_goals,
)

import openai
import dotenv

dotenv.load_dotenv(".env", override=True)


def set_openai_key(key, org):
    """Sets OpenAI key."""
    openai.api_key = key
    openai.organization = org
    print("Openai credentials set.")


set_openai_key(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG"))


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
    # init wandb
    wandb_logger = None
    if not args.wandb_not_log:
        from utils.wandb_logger import WandbLogger

        wandb_logger = WandbLogger(args)

    logger = DataLogger(args)

    num_scenes = args.num_processes

    ############ init tracking vars
    g_masks = torch.ones(num_scenes).float().to(device)
    step_masks = torch.zeros(num_scenes).float().to(device)

    episode_sem_frontier = []
    episode_sem_goal = []
    episode_loc_frontier = []
    for _ in range(args.num_processes):
        episode_sem_frontier.append([])
        episode_sem_goal.append([])
        episode_loc_frontier.append([])
    ############
    sim_context = SimulationContext(args, device)
    frontier_score_list, planner_pose_inputs = initialize_planner_inputs_and_scores(
        args, num_scenes
    )
    sim_context.init_map_and_pose(planner_pose_inputs)
    # I assume first process
    sim_context.pred_update_sem_map()  # predict semantic map frame 1

    # create a random goal map
    global_goals = random_action_goals(
        sim_context.map_manager.local_h,
        sim_context.map_manager.local_w,
        num_scenes,
    )

    goal_maps = sim_context.map_manager.create_global_goal_maps(global_goals)
    planner_inputs = prepare_planner_inputs(
        args,
        sim_context.finished,
        goal_maps,
        sim_context.map_manager.local_map,
        num_scenes,
        planner_pose_inputs,
        sim_context.map_manager.target_edge_map,
        sim_context.map_manager.target_point_map,
        sim_context.wait_env,
    )

    _, done = sim_context.plan_act_and_preprocess(planner_inputs)

    main_simulation_loop(
        done,
        frontier_score_list,
        g_masks,
        global_goals,
        planner_pose_inputs,
        step_masks,
        logger,
        wandb_logger,
        sim_context,
    )


def main_simulation_loop(
    done,
    frontier_score_list,
    g_masks,
    global_goals,
    planner_pose_inputs,
    step_masks,
    logger,
    wandb_logger,
    sim_context,
):
    args, device = sim_context.args, sim_context.device
    num_scenes, num_episodes = sim_context.num_scenes, sim_context.num_episodes

    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)
    # main training loop
    for step in range(args.num_training_frames // args.num_processes + 1):
        if sim_context.finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps  # global step
        l_step = step % args.num_local_steps  # local step

        l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = sim_context.infos[e]["spl"]
                success = sim_context.infos[e]["success"]
                dist = sim_context.infos[e]["distance_to_goal"]
                spl_per_category[sim_context.infos[e]["goal_name"]].append(spl)
                success_per_category[sim_context.infos[e]["goal_name"]].append(success)
                if args.eval:
                    logger.episode_success[e].append(success)
                    logger.episode_spl[e].append(spl)
                    logger.episode_dist[e].append(dist)
                    if len(logger.episode_success[e]) == num_episodes:
                        sim_context.finished[e] = 1

                # this is when epoisode is done
                sim_context.wait_env[e] = 1.0
                sim_context.init_map_and_pose_for_env(
                    step_masks, planner_pose_inputs, e
                )

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        sim_context.pred_update_sem_map_no_clean()

        locs = sim_context.map_manager.local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + sim_context.origins

        sim_context.map_manager.reset_current_location_and_mark_agent()
        # Check if local step
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                step_masks[e] += 1  # NOTE: I think this is not used...

                if sim_context.wait_env[e] == 1:  # New episode
                    sim_context.wait_env[e] = 0.0

                sim_context.map_manager.update_maps_and_poses_for_global_step(
                    e,
                    sim_context.lmb,
                    sim_context.origins,
                    device,
                    sim_context.full_pose,
                    sim_context.map_manager.local_pose,
                    sim_context.clear_flag,
                )

                planner_pose_inputs[e, 3:] = sim_context.lmb[e]
                sim_context.origins[e] = [
                    sim_context.lmb[e][2] * args.map_resolution / 100.0,
                    sim_context.lmb[e][0] * args.map_resolution / 100.0,
                    0.0,
                ]

                sim_context.map_manager.local_map[e] = sim_context.map_manager.full_map[
                    e,
                    :,
                    sim_context.lmb[e, 0] : sim_context.lmb[e, 1],
                    sim_context.lmb[e, 2] : sim_context.lmb[e, 3],
                ]
                sim_context.map_manager.local_pose[e] = (
                    sim_context.full_pose[e]
                    - torch.from_numpy(sim_context.origins[e]).to(device).float()
                )
                if sim_context.infos[e]["replan_flag"]:
                    sim_context.clear_flag[e] = 1

                if sim_context.clear_flag[e]:
                    sim_context.map_manager.local_map[e].fill_(0.0)
                    sim_context.clear_flag[e] = 0
            # ------------------------------------------------------------------
            ### select the frontier edge
            # ------------------------------------------------------------------

            # Edge Update
            for e in range(num_scenes):
                ############################ choose global goal map #############################
                (
                    Wall_lines,
                    Goal_area_list,
                    Goal_edge,
                    Goal_point,
                    Goal_score,
                ) = sim_context.map_manager.get_frontiers(e, sim_context.local_pose)

                # ------------------------------------------------------------------
                # #### frontier score
                # ------------------------------------------------------------------
                frontier_score_list = (
                    sim_context.navigation_manager.calculate_frontier_scores_fbe(
                        e,
                        frontier_score_list,
                        sim_context.infos,
                        sim_context.local_pose,
                        hm3d_category,
                        found_goal,
                    )
                )

            # ------------------------------------------------------------------
            ##### select randomly point
            # ------------------------------------------------------------------
            global_goals = random_action_goals(
                sim_context.map_manager.local_h,
                sim_context.map_manager.local_w,
                sim_context.num_scenes,
            )
            # g_masks = torch.ones(num_scenes).float().to(device)
        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        # NOW It's Local Step
        local_goal_maps = [
            np.zeros(
                (
                    sim_context.map_manager.local_w,
                    sim_context.map_manager.local_h,
                )
            )
            for _ in range(num_scenes)
        ]

        for e in range(num_scenes):
            # ------------------------------------------------------------------
            ##### select frontier point
            # ------------------------------------------------------------------
            global_item = 0
            if len(frontier_score_list[e]) > 0:
                global_item = frontier_score_list[e].index(max(frontier_score_list[e]))

            if np.any(sim_context.map_manager.target_point_map[e] == global_item + 1):
                local_goal_maps[e][
                    sim_context.map_manager.target_point_map[e] == global_item + 1
                ] = 1
                # print("Find the edge")
                sim_context.g_sum_global += 1
            else:
                local_goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                # print("Don't Find the edge")

            # ------------------------------------------------------------------
            cn = sim_context.infos[e]["goal_cat_id"] + 4
            if sim_context.map_manager.local_map[e, cn, :, :].sum() != 0.0:
                # print("Find the target")
                cat_semantic_map = (
                    sim_context.map_manager.local_map[e, cn, :, :].cpu().numpy()
                )
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                if cn == 9:
                    cat_semantic_scores = cv2.dilate(
                        cat_semantic_scores, sim_context.map_manager.tv_kernel
                    )
                local_goal_maps[e] = find_big_connect(cat_semantic_scores)
                found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = (
                sim_context.map_manager.local_map[e, 0, :, :].cpu().numpy()
            )
            p_input["exp_pred"] = (
                sim_context.map_manager.local_map[e, 1, :, :].cpu().numpy()
            )
            p_input["pose_pred"] = planner_pose_inputs[e]
            p_input["goal"] = local_goal_maps[e]  # global_goals[e]
            p_input["map_target"] = sim_context.map_manager.target_point_map[
                e
            ]  # global_goals[e]
            p_input["new_goal"] = l_step == args.num_local_steps - 1
            p_input["found_goal"] = found_goal[e]
            p_input["wait"] = sim_context.wait_env[e] or sim_context.finished[e]
            if args.visualize or args.print_images:
                p_input["map_edge"] = sim_context.map_manager.target_edge_map[e]
                sim_context.map_manager.local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = (
                    sim_context.map_manager.local_map[e, 4:, :, :]
                    .argmax(0)
                    .cpu()
                    .numpy()
                )

        fail_case, done = sim_context.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        if step % args.log_interval == 0:
            total_collision = []
            total_exploration = []
            total_detection = []
            total_fail_case_success = []

            end_time = time.time()
            if args.eval:

                (total_success, total_spl, total_dist) = logger.update_eval_metrics()

                for e in range(args.num_processes):
                    total_collision.append(fail_case[e]["collision"])
                    total_exploration.append(fail_case[e]["exploration"])
                    total_detection.append(fail_case[e]["detection"])
                    total_fail_case_success.append(fail_case[e]["success"])

            episode_metrics = {
                "total_success": total_success,
                "total_spl": total_spl,
                "total_dist": total_dist,
                "total_collision": total_collision,
                "total_exploration": total_exploration,
                "total_detection": total_detection,
                "total_fail_case_success": total_fail_case_success,
            }

            logger.log_evaluation(
                step,
                num_scenes,
                end_time,
                sim_context.g_process_rewards,
                sim_context.g_sum_rewards,
                sim_context.g_sum_global,
                episode_metrics,
            )
            if not args.wandb_not_log:
                log_data = {
                    "step": step,
                    "num_scenes": num_scenes,
                    "end_time": end_time,
                    "process_rewards": sim_context.g_process_rewards,
                    "sum_rewards": sim_context.g_sum_rewards,
                    "sum_global": sim_context.g_sum_global,
                }
                wandb_logger.log_evaluation(data=log_data, step=step, commit=False)

                # Log metrics for each process
                metric_data = {
                    "total_success": (np.mean(episode_metrics["total_success"])),
                    "total_spl": (np.mean(episode_metrics["total_spl"])),
                    "total_dist": (np.mean(episode_metrics["total_dist"])),
                    "total_collision": (np.sum(episode_metrics["total_collision"])),
                    "total_exploration": (np.sum(episode_metrics["total_exploration"])),
                    "total_detection": (np.sum(episode_metrics["total_detection"])),
                    "total_fail_case_success": (
                        np.sum(episode_metrics["total_fail_case_success"])
                    ),
                }
                # print(metric_data)
                wandb_logger.log_evaluation(data=metric_data, step=step, commit=True)

        # ------------------------------------------------------------------
    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        (total_success, total_spl, total_dist) = logger.update_eval_metrics()

        logger.log_final_evaluation(
            sim_context.g_process_rewards,
            sim_context.g_sum_rewards,
            sim_context.g_sum_global,
            total_success,
            total_spl,
            total_dist,
            success_per_category,
            spl_per_category,
        )
        # if not args.wandb_not_log:
        #     wandb_logger.log_metrics(step, episode_metrics)


if __name__ == "__main__":
    main()
