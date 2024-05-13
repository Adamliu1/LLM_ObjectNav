# GPT-3 HLP generator

# import pandas as pd
import openai
import dotenv
import os

dotenv.load_dotenv(".env", override=True)


def set_openai_key(key, org):
    """Sets OpenAI key."""
    openai.api_key = key
    openai.organization = org
    print("Openai credentials set.")


set_openai_key(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG"))

import random
from ast import literal_eval

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions


# ADAM HERE:
# import alfworld.agents.environment
import os
import base64
import sys
import json
import yaml
import cv2
import argparse

from habitat.tasks.nav.object_nav_task import (
    ObjectGoalNavEpisode,
    ObjectViewLocation,
    ObjectGoal,
    ObjectNavigationTask,
)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def encode_image(numpy_img):
    _, JPEG = cv2.imencode(".jpeg", numpy_img)
    return base64.b64encode(JPEG).decode("utf-8")


class LLM_HLP_Generator:
    def __init__(self):
        pass

    def generate_prompt(self, curr_task):

        # completed_plans = curr_task["completed_plans"]
        allowed_cmd = curr_task["allowed_cmd"]

        # header
        prompt = (
            "Create a high-level plan for completing a Object finding navigation "
            + "task using the ALLOWED_ACTION. You will be given some information about the "
            + "environment.\n\n"
        )
        prompt += f"TASK: Find a {curr_task['obj_goal']}, Generate Next Action. When agent is close enough to the object (<0.1), you should use STOP action\n"

        # add a trial number
        prompt += f"TRIAL: {cur_task['trial_num']}\n"
        prompt += f"PREVIOUS_ACTION: {cur_task['prev_action']}'\n"
        prompt += f"PREVIOUS_METRIC: {cur_task['prev_metric']}\n"

        # ENV info
        prompt += "ENVIRONMENT INFO:\n"
        prompt += "''\n"
        prompt += "Compass: " + str(curr_task["compass"]) + "\n"
        prompt += "GPS: " + str(curr_task["gps"]) + "\n"
        prompt += "CURRENT_METRIC: " + str(curr_task["current_metric"]) + "\n"
        prompt += "''\n"
        # prompt += "\nCOMPLETED_PLANS: " + task_past_plan_str \
        prompt += "\nALLOWED_ACTION:" + str(allowed_cmd)
        prompt += "\n\n"
        # MAYBE HERE I CAN HARD-CODED
        prompt += "NOTE: If PREVIOUS_METRIC and CURRENT_METRIC are same,you should try different ACTION as PREVIOUS_ACTION\n"
        prompt += "\n\n"
        prompt += "[YOUR RESPONSE SHOULD BE JUST THIS LINE]"

        return prompt

    # Main point of entry for LLM HLP prompt generator, use in run_eval
    def generate_gpt_prompt(self, curr_task):
        prompt = self.generate_prompt(curr_task)
        return prompt


def llm(prompt, engine, images=None, stop=["\n"]):
    if engine == "gpt-4":
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can plan Object Navigation tasks.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return response.choices[0].message.content

    elif engine == "gpt-4v":
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{images[0]}"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return response.choices[0].text


def parse_llm_output(llm_out: str, curr_task):
    llm_out = llm_out.lower()
    for cmd in curr_task["allowed_cmd"]:
        if cmd in llm_out:
            return cmd


def eval_single_task(cur_task, env, hlp_generator):
    prompt = hlp_generator.generate_gpt_prompt(cur_task)
    # print("prompt: " + prompt)
    rbg_img = cur_task["rgb"]
    llm_out = llm(prompt, engine="gpt-4v", images=[encode_image(rbg_img)], stop=None)
    # print("llm out: " + llm_out.lower())
    next_action = parse_llm_output(llm_out, cur_task)
    print("next action: " + str(next_action))
    if next_action == "move_forward":
        action = HabitatSimActions.move_forward
        # print("action: FORWARD")
    elif next_action == "turn_left":
        action = HabitatSimActions.turn_left
        # print("action: LEFT")
    elif next_action == "turn_right":
        action = HabitatSimActions.turn_right
        # print("action: RIGHT")
    elif next_action == "look_up":
        action = HabitatSimActions.look_up
        # print("action: UP")
    elif next_action == "look_down":
        action = HabitatSimActions.look_down
        # print("action: DOWN")
    elif next_action == "stop":
        action = HabitatSimActions.stop
        # print("action: FINISH")
    else:
        print("INVALID KEY")
        cur_task["trial_num"] += 1
        return observations

    cur_task["trial_num"] = 1
    cur_task["prev_action"] = action
    cur_task["prev_metric"] = env.get_metrics()
    observations = env.step(action)
    print(env.get_metrics())
    return observations


if __name__ == "__main__":
    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    )

    # print(f"goal obj: {env.current_episode.object_category}")
    # print(f"action space: {list(env.action_space.keys())}")
    # print(f"observation space: {env.observation_space}")
    # print("Environment creation successful")
    observations = env.reset()
    # print(transform_rgb_bgr(observations["rgb"]).shape)
    # print(observations["depth"].shape)
    # print("ob", observations["depth"])
    # print("Object Goal: ", observations["objectgoal"])
    # print("Compass: ", observations["compass"])
    hlp_generator = LLM_HLP_Generator()

    count_steps = 0

    cur_task = {
        "obj_goal": env.current_episode.object_category,
        "compass": observations["compass"],
        "gps": observations["gps"],
        "allowed_cmd": list(env.action_space.keys()),
        "rgb": observations["rgb"],
        "depth": observations["depth"],
        "current_metric": env.get_metrics(),
        "prev_metric": None,
        "trial_num": 1,
        "prev_action": None,
    }

    while not env.episode_over:

        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("Depth", observations["depth"])
        cv2.imshow("Objectgoal", observations["objectgoal"])
        cv2.imshow("Compass", transform_rgb_bgr(observations["compass"]))
        keystroke = cv2.waitKey(1)
        observations = eval_single_task(cur_task, env, hlp_generator)

        cur_task["rgb"] = observations["rgb"]
        cur_task["depth"] = observations["depth"]
        cur_task["compass"] = observations["compass"]
        cur_task["gps"] = observations["gps"]
        cur_task["current_metric"] = env.get_metrics()
        count_steps += 1
