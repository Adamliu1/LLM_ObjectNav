from collections import Counter
import re
import skimage
from envs.utils.fmm_planner import FMMPlanner
from utils.util_helper import get_sensor_data
from utils.map_helper import get_frontier_boundaries
from utils.navigation_helper import NavigationManager
from constants import category_to_id
import torch.nn.functional as F
import numpy as np

from utils.llm.llm_zeroshot_service import LLMPromptMethod


class LLM_Zeroshot_NavigationManager(NavigationManager):
    def __init__(self, map_manager, lm_service, args, device):
        super().__init__(map_manager, args, device)
        self.lm_service = lm_service

    def calculate_frontier_scores_fbe(
        self,
        e,
        frontier_score_list,
        infos,
        local_pose,
        hm3d_category,
        found_goal,
    ):
        cname = infos[e]["goal_name"]
        frontier_score_list[e] = []
        tpm = len(set(self.map_manager.target_point_map[e].ravel())) - 1

        cur_loc_x, cur_loc_y = self.map_manager.get_agent_location_in_map(local_pose[e])
        print("current location: ", cur_loc_x, cur_loc_y)

        # generate obj clusters
        frontier_dists = []
        distance_scores = [0] * tpm
        for lay in range(tpm):
            f_pos = np.argwhere(self.map_manager.target_point_map[e] == lay + 1)
            print("frontier pos: ", f_pos)
            print("f_pos_shape: ", f_pos.shape)

            distances = np.linalg.norm(f_pos - np.array([cur_loc_x, cur_loc_y]), axis=1)
            frontier_dists.extend(distances)
            print("Current Frontier Distances:", distances)

        # Normalize the distances if any distances were recorded
        if frontier_dists:
            max_distance = max(frontier_dists)
            min_distance = min(frontier_dists)
            distance_range = max_distance - min_distance
            distance_scores = [
                ((1 - ((d - min_distance) / distance_range)) if distance_range else 1.0)
                for d in frontier_dists
            ]
        else:
            distance_scores = [0] * tpm

        print(f"debug: distance score: {distance_scores}")
        frontier_score_list[e] = distance_scores
        return frontier_score_list

    # FROM HERE, IS THE FINAL REDO
    def calculate_frontier_scores_no_cot(
        self,
        e,
        frontier_score_list,
        infos,
        local_pose,
        hm3d_category,
        found_goal,
    ):
        cname = infos[e]["goal_name"]
        frontier_score_list[e] = []
        tpm = len(set(self.map_manager.target_point_map[e].ravel())) - 1

        # NOTE: num of scene, bad coding example here
        cur_loc_x, cur_loc_y = self.map_manager.get_agent_location_in_map(local_pose[e])
        print("current location: ", cur_loc_x, cur_loc_y)

        # generate obj clusters
        clusters = []
        frontier_dists = []
        distance_scores = [0] * tpm
        for lay in range(tpm):
            f_pos = np.argwhere(self.map_manager.target_point_map[e] == lay + 1)
            print("frontier pos: ", f_pos)
            print("f_pos_shape: ", f_pos.shape)
            objs = self._get_objs(
                e,
                lay,
                hm3d_category,
            )
            clusters.append(objs)
            # Calculate Euclidean distances
            distances = np.linalg.norm(f_pos - np.array([cur_loc_x, cur_loc_y]), axis=1)
            frontier_dists.extend(distances)
            print("Current Frontier Distances:", distances)

        # Normalize the distances if any distances were recorded
        if frontier_dists:
            max_distance = max(frontier_dists)
            min_distance = min(frontier_dists)
            distance_range = max_distance - min_distance
            distance_scores = [
                ((1 - ((d - min_distance) / distance_range)) if distance_range else 1.0)
                for d in frontier_dists
            ]
        else:
            distance_scores = [0] * len(
                clusters
            )  # Set all scores to zero if no distances

        # Mask and filtered clusters handling
        filtered_clusters = [
            cluster for cluster in clusters if cluster and found_goal[e] == 0
        ]
        filtered_indexes = [
            i for i, cluster in enumerate(clusters) if cluster and found_goal[e] == 0
        ]

        dis_weight = 0.1  # Define distance weight
        # Language model queries for valid clusters
        if filtered_clusters:
            pos_scores, _ = self.lm_service.query_llm(
                LLMPromptMethod.POS_NO_COT,
                object_clusters=filtered_clusters,
                goal=cname,
            )
            print("pos_scores: ", pos_scores)
            # Update scores with LLM results and distance weight where applicable
            for idx, pos_score in enumerate(pos_scores):
                final_idx = filtered_indexes[idx]
                final_score = pos_score
                distance_scores[final_idx] = (
                    final_score + dis_weight * distance_scores[final_idx]
                )

        # Apply the distance weight for all distances regardless of LLM query
        for i in range(len(distance_scores)):
            if i not in filtered_indexes:
                distance_scores[i] *= dis_weight

        # Update the scores for each layer in the frontier score list
        frontier_score_list[e] = distance_scores
        print("Frontier scores updated:", frontier_score_list[e])

        return frontier_score_list

    def calculate_frontier_scores_v2(
        self,
        e,
        frontier_score_list,
        infos,
        local_pose,
        hm3d_category,
        found_goal,
    ):
        # NOTE: ONLY POSITIVE SAMPLING BUT WITH CHAIN OF THOUGHT
        cname = infos[e]["goal_name"]
        frontier_score_list[e] = []
        tpm = len(set(self.map_manager.target_point_map[e].ravel())) - 1

        # NOTE: num of scene, bad coding example here
        cur_loc_x, cur_loc_y = self.map_manager.get_agent_location_in_map(local_pose[e])
        print("current location: ", cur_loc_x, cur_loc_y)

        # generate obj clusters
        clusters = []
        frontier_dists = []
        distance_scores = [0] * tpm
        for lay in range(tpm):
            f_pos = np.argwhere(self.map_manager.target_point_map[e] == lay + 1)
            print("frontier pos: ", f_pos)
            print("f_pos_shape: ", f_pos.shape)
            objs = self._get_objs(
                e,
                lay,
                hm3d_category,
            )
            clusters.append(objs)
            # Calculate Euclidean distances
            distances = np.linalg.norm(f_pos - np.array([cur_loc_x, cur_loc_y]), axis=1)
            frontier_dists.extend(distances)
            print("Current Frontier Distances:", distances)

        # Normalize the distances if any distances were recorded
        if frontier_dists:
            max_distance = max(frontier_dists)
            min_distance = min(frontier_dists)
            distance_range = max_distance - min_distance
            distance_scores = [
                ((1 - ((d - min_distance) / distance_range)) if distance_range else 1.0)
                for d in frontier_dists
            ]
        else:
            distance_scores = [0] * len(
                clusters
            )  # Set all scores to zero if no distances

        # print(f"debug: distance score: {distance_scores}")
        # Mask and filtered clusters handling
        filtered_clusters = [
            cluster for cluster in clusters if cluster and found_goal[e] == 0
        ]
        filtered_indexes = [
            i for i, cluster in enumerate(clusters) if cluster and found_goal[e] == 0
        ]

        dis_weight = 0.1  # Define distance weight
        # Language model queries for valid clusters
        if filtered_clusters:
            pos_scores, _ = self.lm_service.query_llm(
                LLMPromptMethod.POS_COT,
                object_clusters=filtered_clusters,
                goal=cname,
            )

            # Update scores with LLM results and distance weight where applicable
            for idx, pos_score in enumerate(pos_scores):
                final_idx = filtered_indexes[idx]
                final_score = pos_score
                distance_scores[final_idx] = (
                    final_score + dis_weight * distance_scores[final_idx]
                )

        # Apply the distance weight for all distances regardless of LLM query
        for i in range(len(distance_scores)):
            if i not in filtered_indexes:
                distance_scores[i] *= dis_weight

        # Update the scores for each layer in the frontier score list
        frontier_score_list[e] = distance_scores
        print("Frontier scores updated:", frontier_score_list[e])

        return frontier_score_list

    def calculate_frontier_scores_v3(
        self,
        e,
        frontier_score_list,
        infos,
        local_pose,
        hm3d_category,
        found_goal,
    ):
        # NOTE: POSITIVE SAMPLING PLUS NEGATIVE SAMPLING ALL WITH CHAIN OF THOUGHT
        cname = infos[e]["goal_name"]
        frontier_score_list[e] = []
        tpm = len(set(self.map_manager.target_point_map[e].ravel())) - 1

        # NOTE: num of scene, bad coding example here
        cur_loc_x, cur_loc_y = self.map_manager.get_agent_location_in_map(local_pose[e])
        print("current location: ", cur_loc_x, cur_loc_y)

        # generate obj clusters
        clusters = []
        frontier_dists = []
        distance_scores = [0] * tpm
        for lay in range(tpm):
            f_pos = np.argwhere(self.map_manager.target_point_map[e] == lay + 1)
            print("frontier pos: ", f_pos)
            print("f_pos_shape: ", f_pos.shape)
            objs = self._get_objs(
                e,
                lay,
                hm3d_category,
            )
            clusters.append(objs)
            # Calculate Euclidean distances
            distances = np.linalg.norm(f_pos - np.array([cur_loc_x, cur_loc_y]), axis=1)
            frontier_dists.extend(distances)
            print("Current Frontier Distances:", distances)

        # Normalize the distances if any distances were recorded
        if frontier_dists:
            max_distance = max(frontier_dists)
            min_distance = min(frontier_dists)
            distance_range = max_distance - min_distance
            distance_scores = [
                ((1 - ((d - min_distance) / distance_range)) if distance_range else 1.0)
                for d in frontier_dists
            ]
        else:
            distance_scores = [0] * len(
                clusters
            )  # Set all scores to zero if no distances

        print(f"debug: distance score: {distance_scores}")
        # Mask and filtered clusters handling
        filtered_clusters = [
            cluster for cluster in clusters if cluster and found_goal[e] == 0
        ]
        filtered_indexes = [
            i for i, cluster in enumerate(clusters) if cluster and found_goal[e] == 0
        ]

        dis_weight = 0.1  # Define distance weight
        w_pos = 1
        w_neg = 0.5
        # Language model queries for valid clusters
        if filtered_clusters:
            pos_scores, _ = self.lm_service.query_llm(
                LLMPromptMethod.POS_COT,
                object_clusters=filtered_clusters,
                goal=cname,
            )
            neg_scores, _ = self.lm_service.query_llm(
                LLMPromptMethod.NEG_COT,
                object_clusters=filtered_clusters,
                goal=cname,
            )

            # Update scores with LLM results and distance weight where applicable
            for idx, (pos_score, neg_score) in enumerate(zip(pos_scores, neg_scores)):
                final_idx = filtered_indexes[idx]
                final_score = w_pos * pos_score - w_neg * neg_score
                distance_scores[final_idx] = (
                    final_score + dis_weight * distance_scores[final_idx]
                )

        # Apply the distance weight for all distances regardless of LLM query
        for i in range(len(distance_scores)):
            if i not in filtered_indexes:
                distance_scores[i] *= dis_weight

        # Update the scores for each layer in the frontier score list
        frontier_score_list[e] = distance_scores
        print("Frontier scores updated:", frontier_score_list[e])

        return frontier_score_list

    def _get_objs(
        self,
        e,
        lay,
        hm3d_category,
    ):
        # Find positions within the target layer
        f_pos = np.argwhere(self.map_manager.target_point_map[e] == lay + 1)
        if not len(f_pos):
            return 0.1  # Default score if no positions found

        # Determine the boundary for objects in the current layer
        fmb = get_frontier_boundaries(
            (f_pos[0][0], f_pos[0][1]),
            (self.map_manager.local_w / 4, self.map_manager.local_h / 4),
            (self.map_manager.local_w, self.map_manager.local_h),
        )
        objs_list = []

        # Check for semantic objects within the boundary
        for se_cn in range(self.args.num_sem_categories - 1):
            semantic_layer = self.map_manager.local_map[e][
                se_cn + 4, fmb[0] : fmb[1], fmb[2] : fmb[3]
            ]
            if semantic_layer.sum() != 0.0:
                objs_list.append(hm3d_category[se_cn])

        return objs_list

    def _choose_nearlest_frontier(
        self,
        Frontier_list,
        target_point_map,
    ):

        smallest_index = Frontier_list.index(min(Frontier_list))
        Frontiers = "\n ".join(
            [
                f"frontier_{i}: <centroid: {target_point_map[i][0], target_point_map[i][1]}, number: {Frontier_list[i]}>"
                for i in range(len(target_point_map))
            ]
        )
        # select nearest frontier
        print("Selecting nearest frontier")
        selected_frontier_str = Frontiers[smallest_index]
        x, y = target_point_map[smallest_index]
        return x, y, selected_frontier_str

    def calculate_frontier_scores_v4(
        self,
        e,
        local_pose,
        Wall_list,
        Frontier_list,
        target_edge_map,
        target_point_map,
        object_clusters: dict,
        previous_frontier_movement,
        infos,
    ):
        # cn = infos[e]["goal_cat_id"] + 4
        cname = infos[e]["goal_name"]
        cur_loc_x, cur_loc_y = self.map_manager.get_agent_location_in_map(local_pose[e])
        print("current location: ", cur_loc_x, cur_loc_y)
        prompt, frontiers_dict = self.lm_service.form_prompt_direct_goal_pos(
            goal_name=cname,
            pose_pred=(cur_loc_x, cur_loc_y),
            object_list=object_clusters,
            Wall_list=Wall_list,
            Frontier_list=Frontier_list,
            last_decision=previous_frontier_movement[e],
            Frontier_points=target_point_map,
        )
        # print("prompt: ", prompt)
        print("frontiers_dict: ", frontiers_dict)
        answers, reasonings = self.lm_service.query_llm_direct_prompt(prompt=prompt)

        # Filter out invalid answers (only keep strings)
        valid_answers = [ans for ans in answers if isinstance(ans, str)]

        # Proceed only if there are valid answers
        if valid_answers:
            # Use Counter to determine the most common answer
            selected_frontier_str, num_of_count = Counter(valid_answers).most_common(1)[
                0
            ]
            print("Selected frontier string:", selected_frontier_str)

            # Retrieve the selected frontier's information
            selected_frontier = frontiers_dict.get(selected_frontier_str)
            print("Selected frontier:", selected_frontier)
            if not selected_frontier:
                x, y, selected_frontier_str = self._choose_nearlest_frontier(
                    Frontier_list, target_point_map
                )
            else:
                # Extract the coordinates from the selected frontier string
                centroid_part = selected_frontier.split("centroid: (")[1].split("),")[0]
                x, y = map(int, centroid_part.split(","))
                print(f"Coordinates extracted: ({x}, {y})")
        else:
            x, y, selected_frontier_str = self._choose_nearlest_frontier(
                Frontier_list, target_point_map
            )

        previous_frontier_movement[e] = selected_frontier_str

        return x, y
