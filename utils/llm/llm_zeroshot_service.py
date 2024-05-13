from utils.llm_helper import LanguageModelService
import retry
import openai
from enum import Enum


POSITIVE_PROMPT_NO_COT = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Answer: 3


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
"""

POS_COT = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. Because we are looking for a knife which is typically located in a ktichen we should check cluster 3.
Answer: 3


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer.
3. Feel free to think multiple steps in advance; for example if one room is typically located near another then it is ok to use that information to provide higher scores in that direction.
"""

NEG_COT = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide reasoning along with a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space after Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. A knife is not likely to be in an entertainment room or an office room so we should avoid searching those spaces.
Answer: 1,2


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer
"""


DIRECT_SELECT_POSITION = """You are a robot exploring an environment for the first time.

Objective: Control robot equipped with cameras and 5-meter range depth sensors. The task is to guide the robot on where to explore next in an indoor home scene on a 2D map to locate a specific target object. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore.

Map Information:
Origin: Top-left corner.
Coordinates: Represented in pixels.

Representation Details:
Robot Position: Format - (x, y).
Scene Objects: Defined by vertices in clockwise order. Format - <(x1, y1), (x2, y2)...>.
Walls: Lines with start and end points. Format - <(x1_start, y1_start, x1_end, y1_end)>.
Frontiers: Defined by centroid and pixel count. Format - <centroid:(x, y), number: n>.

Strategy Considerations:
1. Robot need to choose a frontier to explore.
2. Assign robot based on the relationships between different objects, the structure of the explored areas robot's position, frontier proximity, and their previous movement directions.
3. Minimize frequent switches between frontiers. Use centroid for frontier selection.
4. A robot should maintain its exploration direction unless an efficient switch is evident.
5. You will not be given room labels. Use your best judgment when determining what room the fronter is likely to be in using the Scene Objects' locations.
6. Provide reasoning before giving the final answer.
7. Feel free to think multiple steps in advance; for example if one room is typically located near another then it is ok to use that information to provide higher scores in that direction.

Instructions: Given the scene details, decide the best frontier each robot should explore next. Provide ONLY the reasoning in the [reasoning:] format. Provide ONLY the decision in the [output:] format without additional explanations or additional text.

Example: 

[input:]
Task: Locate the chairs

Position: 
robot: (240, 240)

Scene Objects:
sofa: <(280, 200), (280, 150), (330, 150), (330, 180), (300, 180), (300, 200)> 
bed: <(220, 250), (240, 250), (250, 250), (250, 220)> 
chest_of_drawers: <(200, 240), (210, 240), (210, 250), (200, 250)> 
tv_monitor: <(220, 200), (240, 200), (240, 210), (220, 210)> 
table: <(200, 150), (230, 150), (230, 170), (200, 170)> 

Walls:
wall_0: <(190, 180, 300, 180)> 
wall_1: <(160, 180, 160, 250)> 
wall_2: <(160, 250, 30, 250)> 

Previous Movements:
robot: <centroid:(195, 320), number: 60>

Unexplored Frontier: 
frontier_0: <centroid:(180, 175), number: 40> 
frontier_1: <centroid:(195, 280), number: 80>

[reasoning:]
The robot is currently at position (240, 240), and the unexplored frontier_0 is located at (180, 175). frontier_0 has a cluster pixel size of 40, frontier_1 has a cluster pixel size of 80. frontier_0 is more likey than frontier_1 to contain chairs as the sofa is located near the frontier. The sofa is a common object found in living rooms and the robot is likely to find chairs in the same area.

[output:]
robot: frontier_1

Please give the output based on the following input:\n"""


class LLMPromptMethod(Enum):
    POS_NO_COT = 1
    POS_COT = 2
    NEG_COT = 3
    DIRECT_SELECT_POSITION = 4


class LanguageModelService_ZeroShot(LanguageModelService):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)

    @retry.retry(tries=5)
    def _ask_gpts_no_cot(
        self,
        goal,
        object_clusters,
        env="an indoor environment",
        num_samples=5,
        model="gpt-4",
    ):
        prompt_msg = POSITIVE_PROMPT_NO_COT

        messages = [
            {"role": "system", "content": prompt_msg},
        ]
        if len(object_clusters) > 0:
            options = ""
            for i, cluster in enumerate(object_clusters):
                cluser_string = ""
                for ob in cluster:
                    cluser_string += ob + ", "
                options += f"{i+1}. {cluser_string[:-2]}\n"
            messages.append(
                {
                    "role": "user",
                    "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?",
                }
            )
            completion = openai.ChatCompletion.create(
                model=model, temperature=1, n=num_samples, messages=messages
            )

            answers = []
            for choice in completion.choices:
                try:
                    complete_response = choice.message["content"]
                    # Make the response all lowercase
                    complete_response = complete_response.lower()
                    # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                    if len(complete_response.split("answer:")) > 1:
                        answer = complete_response.split("answer:")[1].split("\n")[0]
                        # Separate the answers by commas
                        answers.append([int(x) for x in answer.split(",")])
                    else:
                        answers.append([])
                except:
                    answers.append([])

            # Flatten answers
            flattened_answers = [item for sublist in answers for item in sublist]
            # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
            filtered_flattened_answers = [
                x for x in flattened_answers if x >= 1 and x <= len(object_clusters)
            ]
            # Aggregate into counts and normalize to probabilities
            answer_counts = {
                x: filtered_flattened_answers.count(x) / len(answers)
                for x in set(filtered_flattened_answers)
            }

            return answer_counts
        raise Exception("Error: Empty object categories")

    @retry.retry(tries=5)
    def _ask_gpts_cot(
        self,
        goal,
        object_clusters,
        env="an indoor environment",
        positives=True,
        num_samples=5,
        model="gpt-4",
        chain_of_thought=True,
    ):
        if positives:
            system_message = POS_COT
        else:
            system_message = NEG_COT

        messages = [
            {"role": "system", "content": system_message},
        ]
        if len(object_clusters) > 0:
            options = ""
            for i, cluster in enumerate(object_clusters):
                cluser_string = ""
                for ob in cluster:
                    cluser_string += ob + ", "
                options += f"{i+1}. {cluser_string[:-2]}\n"
            if positives:
                messages.append(
                    {
                        "role": "user",
                        "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I avoid spending time searching if I am looking for {goal}?",
                    }
                )

            completion = openai.ChatCompletion.create(
                model=model, temperature=1, n=num_samples, messages=messages
            )

            answers = []
            reasonings = []
            for choice in completion.choices:
                try:
                    complete_response = choice.message["content"]
                    # Make the response all lowercase
                    complete_response = complete_response.lower()
                    if chain_of_thought:
                        reasoning = complete_response.split("reasoning: ")[1].split(
                            "\n"
                        )[0]
                    else:
                        reasoning = "disabled"
                    # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                    if len(complete_response.split("answer:")) > 1:
                        answer = complete_response.split("answer:")[1].split("\n")[0]
                        # Separate the answers by commas
                        answers.append([int(x) for x in answer.split(",")])
                    else:
                        answers.append([])
                    reasonings.append(reasoning)
                except:
                    answers.append([])

            # Flatten answers
            flattened_answers = [item for sublist in answers for item in sublist]
            # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
            filtered_flattened_answers = [
                x for x in flattened_answers if x >= 1 and x <= len(object_clusters)
            ]
            # Aggregate into counts and normalize to probabilities
            answer_counts = {
                x: filtered_flattened_answers.count(x) / len(answers)
                for x in set(filtered_flattened_answers)
            }
            return answer_counts, reasonings
        raise Exception("Error: Empty object categories")

    def form_prompt_direct_goal_pos(
        self,
        goal_name,
        pose_pred,
        object_list,
        Wall_list,
        Frontier_list,
        last_decision,
        Frontier_points,
    ):

        Robot_Position = "\n ".join([f"robot: {pose_pred[0], pose_pred[1]}"])

        for _, value in object_list.items():
            value = ", ".join([f"{value[i]}" for i in range(len(value))])
        Objects_Position = (
            "\n ".join(
                [
                    f"{key}: "
                    + ", ".join(
                        [
                            f"<"
                            + ", ".join(
                                [
                                    f"{value[i][j][0][0], value[i][j][0][1]}"
                                    for j in range(len(value[i]))
                                ]
                            )
                            + f">"
                            for i in range(len(value))
                        ]
                    )
                    for key, value in object_list.items()
                ]
            )
            + "\n"
        )

        if Wall_list is not None:
            Walls_Position = (
                "\n ".join(
                    [
                        f"wall_{i}: <"
                        + ", ".join(
                            [
                                f"{Wall_list[i][0][j]}"
                                for j in range(len(Wall_list[i][0]))
                            ]
                        )
                        + f">"
                        for i in range(len(Wall_list))
                    ]
                )
                + "\n"
            )
        else:
            Walls_Position = None

        Frontiers = "\n ".join(
            [
                f"frontier_{i}: <centroid: {Frontier_points[i][0], Frontier_points[i][1]}, number: {Frontier_list[i]}>"
                for i in range(len(Frontier_points))
            ]
        )

        if last_decision is not None:
            Last_Decision = "\n ".join([f"robot: {last_decision}"])
        else:
            Last_Decision = "No frontiers"

        prompt_template = """
        [input:]
        Task: Locate the {GOAL_NAME}

        Position: 
        {ROBOT_POSITION}

        Scene Objects:
        {OBJECTS_POSITION}

        Walls:
        {WALLS_POSITION}

        Previous Movements:
        {LAST_DECISION}

        Unexplored Frontier: 
        {FRONTIERS}
        
        [reasoning:] 
        ''reasoning here''

        [output:]
        ''output here''

        """

        User_prompt = prompt_template.format(
            GOAL_NAME=str(goal_name),
            ROBOT_POSITION=Robot_Position,
            OBJECTS_POSITION=Objects_Position,
            WALLS_POSITION=Walls_Position,
            FRONTIERS=Frontiers,
            LAST_DECISION=Last_Decision,
        )

        Frontiers_dict = {}
        for i in range(len(Frontier_points)):
            Frontiers_dict["frontier_" + str(i)] = (
                f"<centroid: {Frontier_points[i][0], Frontier_points[i][1]}, number: {Frontier_list[i]}>"
            )

        return User_prompt, Frontiers_dict

    def query_llm(
        self,
        method: LLMPromptMethod,
        object_clusters: list,
        goal: str,
    ) -> list:
        # Convert object clusters to a tuple of tuples so we can hash it and get unique elements
        object_clusters_tuple = [tuple(x) for x in object_clusters]
        # Remove empty clusters and duplicate clusters
        query = list(set(tuple(object_clusters_tuple)) - set({tuple([])}))
        reasoning = []
        print(f"method: {method}")
        if method == LLMPromptMethod.POS_NO_COT:
            try:
                answer_counts = self._ask_gpts_no_cot(
                    goal,
                    query,
                    model="gpt-3.5-turbo",
                )  # NOTE: too fucking expensive, change to gpt3
            except Exception as excptn:
                answer_counts = {}, "GPTs failed"
                print("GPTs failed:", excptn)
            language_scores = [0] * len(object_clusters_tuple)
            for key, value in answer_counts.items():
                for i, x in enumerate(object_clusters_tuple):
                    if x == query[key - 1]:
                        language_scores[i] = value
        elif method == LLMPromptMethod.POS_COT:
            try:
                answer_counts, reasoning = self._ask_gpts_cot(
                    goal,
                    query,
                    positives=True,
                    model="gpt-3.5-turbo",
                    chain_of_thought=True,
                )  # NOTE: too fucking expensive, change to gpt3
            except Exception as excptn:
                answer_counts, reasoning = {}, "GPTs failed"
                print("GPTs failed:", excptn)
            language_scores = [0] * len(object_clusters_tuple)
            for key, value in answer_counts.items():
                for i, x in enumerate(object_clusters_tuple):
                    if x == query[key - 1]:
                        language_scores[i] = value
        elif method == LLMPromptMethod.NEG_COT:
            try:
                answer_counts, reasoning = self._ask_gpts_cot(
                    goal,
                    query,
                    positives=False,
                    model="gpt-3.5-turbo",
                    chain_of_thought=True,
                )  # NOTE: too fucking expensive, change to gpt3
            except Exception as excptn:
                answer_counts, reasoning = {}, "GPTs failed"
                print("GPTs failed:", excptn)
            language_scores = [0] * len(object_clusters_tuple)
            for key, value in answer_counts.items():
                for i, x in enumerate(object_clusters_tuple):
                    if x == query[key - 1]:
                        language_scores[i] = value

        else:
            raise Exception("Invalid method")

        # The first element of language scores is the scores for uncertain, the last n-1 correspond to the semantic scores for each point
        return language_scores, reasoning

    @retry.retry(tries=5)
    def query_llm_direct_prompt(
        self,
        prompt: str,
        model="gpt-3.5-turbo",
        # model="gpt-4-turbo",
        env="an indoor environment",
        num_samples=5,
    ):
        system_message = DIRECT_SELECT_POSITION
        messages = [
            {"role": "system", "content": system_message},
        ]
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        completion = openai.ChatCompletion.create(
            model=model, temperature=1, n=num_samples, messages=messages
        )
        answers = []
        reasonings = []
        for choice in completion.choices:
            try:
                complete_response = choice.message["content"]
                # Make the response all lowercase
                complete_response = complete_response.lower()
                # print(f"#############\n{complete_response}\n############")
                reasoning = complete_response.split("[reasoning:]\n")[1].split(
                    "[output:]"
                )[0]
                # print(f"########\nreasoning: {reasoning}\n########")
                answer = complete_response.split("[output:]\n")[1].split("robot: ")[1]
                answers.append(answer)
                # print(f"########\nanswer: {answer}\n########")
                reasonings.append(reasoning)
            except:
                answers.append([])
        return answers, reasonings
