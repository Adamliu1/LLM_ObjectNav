# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
from habitat.config.default_structured_configs import (
    HabitatSimDepthSensorConfig,
    HabitatSimRGBSensorConfig,
    HabitatSimSemanticSensorConfig,
)
import numpy as np
import torch

from habitat.datasets import make_dataset

from agents.llm_agent import LLM_Agent_Env
from .utils.vector_env import VectorEnv

from agents.l3mvn_agent import L3MVNAgent
from agents.sem_exp_baseline import SemExpBaseLineAgent

from habitat.config import get_config as cfg_env
from omegaconf import OmegaConf

from habitat.version import VERSION as __version__  # noqa: F401


def make_env_fn(args, config_env, rank):
    dataset = make_dataset(
        config_env["habitat"]["dataset"]["type"],
        config=config_env["habitat"]["dataset"],
    )
    OmegaConf.set_readonly(config_env, False)
    config_env["habitat"]["simulator"]["scene"] = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)

    if args.agent == "sem_exp":
        env = SemExpBaseLineAgent(
            args=args, rank=rank, config_env=config_env, dataset=dataset
        )
    elif args.agent == "llm_sem_agent":
        env = LLM_Agent_Env(
            args=args, rank=rank, config_env=config_env, dataset=dataset
        )
    elif args.agent == "l3mvn":
        env = L3MVNAgent(args=args, rank=rank, config_env=config_env, dataset=dataset)
    else:
        env = L3MVNAgent(args=args, rank=rank, config_env=config_env, dataset=dataset)

    env.seed(rank)
    return env


def construct_envs21(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path=args.task_config)
    OmegaConf.set_readonly(basic_config, False)  # equivalent to defrost
    basic_config["habitat"]["dataset"]["split"] = args.split
    basic_config["habitat"]["dataset"]["data_path"] = basic_config["habitat"][
        "dataset"
    ]["data_path"].replace("v1", args.version)

    scenes = basic_config["habitat"]["dataset"]["content_scenes"]
    dataset = make_dataset(
        basic_config["habitat"]["dataset"]["type"],
        config=basic_config["habitat"]["dataset"],
    )
    OmegaConf.set_readonly(basic_config, True)
    if "*" in scenes:
        scenes = dataset.get_scenes_to_load(basic_config["habitat"]["dataset"])

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there " "aren't enough number of scenes"
        )

        scene_split_sizes = [
            int(np.floor(len(scenes) / args.num_processes))
            for _ in range(args.num_processes)
        ]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")

    for i in range(args.num_processes):
        config_env = cfg_env(config_path=args.task_config)

        OmegaConf.set_readonly(config_env, False)
        config_env["habitat"]["dataset"]["split"] = args.split
        config_env["habitat"]["dataset"]["data_path"] = config_env["habitat"][
            "dataset"
        ]["data_path"].replace("v1", args.version)

        if len(scenes) > 0:
            config_env["habitat"]["dataset"]["content_scenes"] = scenes[
                sum(scene_split_sizes[:i]) : sum(scene_split_sizes[: i + 1])
            ]
            print(
                "Thread {}: {}".format(
                    i, config_env["habitat"]["dataset"]["content_scenes"]
                )
            )

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = (
                int((i - args.num_processes_on_first_gpu) // args.num_processes_per_gpu)
                + args.sim_gpu_id
            )
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env["habitat"]["simulator"]["habitat_sim_v0"]["gpu_device_id"] = gpu_id

        rgb_sensor = HabitatSimRGBSensorConfig(
            height=args.env_frame_height,
            width=args.env_frame_width,
            position=[0.0, args.camera_height, 0.0],
            hfov=args.hfov,
        )
        depth_sensor = HabitatSimDepthSensorConfig(
            height=args.env_frame_height,
            width=args.env_frame_width,
            position=[0.0, args.camera_height, 0.0],
            hfov=args.hfov,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

        semantic_sensor = HabitatSimSemanticSensorConfig(
            height=args.env_frame_height,
            width=args.env_frame_width,
            position=[0.0, args.camera_height, 0.0],
            hfov=args.hfov,
        )

        # Configuring sensors with updated parameters from args
        config_env["habitat"]["simulator"]["agents"]["main_agent"]["sim_sensors"] = {
            "rgb_sensor": rgb_sensor,
            "depth_sensor": depth_sensor,
            "semantic_sensor": semantic_sensor,
        }

        config_env["habitat"]["environment"]["max_episode_steps"] = 10000000
        config_env["habitat"]["environment"]["iterator_options"]["shuffle"] = False
        config_env["habitat"]["simulator"]["turn_angle"] = args.turn_angle
        config_env["habitat"]["dataset"]["split"] = args.split

        OmegaConf.set_readonly(config_env, True)
        env_configs.append(config_env)

        args_list.append(args)

    # NOTE: VectorEnv takes more resources than ThreadedVectorEnv, but faster
    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(args_list, env_configs, range(args.num_processes)))
        ),
    )

    return envs
