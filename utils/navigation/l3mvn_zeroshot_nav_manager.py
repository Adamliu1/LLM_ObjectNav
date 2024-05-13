from utils.map_helper import get_frontier_boundaries
from utils.navigation_helper import NavigationManager
from constants import category_to_id
import torch.nn.functional as F
import numpy as np


class L3MVN_Zeroshot_NavigationManager(NavigationManager):
    def __init__(self, map_manager, lm_service, args, device):
        super().__init__(map_manager, args, device)
        self.lm_service = lm_service

    def calculate_frontier_scores(
        self,
        e,
        frontier_score_list,
        infos,
        hm3d_category,
        Goal_score,
        found_goal,
    ):
        cn = infos[e]["goal_cat_id"] + 4
        cname = infos[e]["goal_name"]
        frontier_score_list[e] = []
        tpm = len(set(self.map_manager.target_point_map[e].ravel())) - 1

        for lay in range(tpm):
            score = self._evaluate_layer_zeroshot(
                e,
                lay,
                cname,
                hm3d_category,
                Goal_score,
                found_goal,
            )
            frontier_score_list[e].append(score)

        return frontier_score_list

    def _evaluate_layer_zeroshot(
        self,
        e,
        lay,
        cname,
        hm3d_category,
        Goal_score,
        found_goal,
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

        # Use language model to evaluate the relevance of found objects to the goal
        if len(objs_list) > 0 and found_goal[e] == 0:
            ref_dist = F.softmax(self.lm_service.construct_dist(objs_list), dim=0).to(
                self.device
            )
            new_dist = ref_dist
            # print("cname", cname)
            # print(new_dist)
            scores = new_dist[category_to_id.index(cname)]
            # frontier_score_list[e].append(scores)
            return scores
        else:
            # Fallback score calculation
            scores = Goal_score[lay] / max(Goal_score) * 0.1 + 0.1
        return scores
