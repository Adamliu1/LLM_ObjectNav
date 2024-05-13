from utils.navigation_helper import NavigationManager
from utils.map_helper import get_frontier_boundaries
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from constants import category_to_id


class L3MVN_FFN_NavigationManager(NavigationManager):
    def __init__(self, map_manager, lm_service, ff_net, args, device):
        super().__init__(map_manager, args, device)
        self.lm_service = lm_service
        self.ff_net = ff_net

    def calculate_frontier_scores(
        self,
        e,
        frontier_score_list,
        infos,
        object_norm_inv_perplexity,
        hm3d_category,
        hm3d_semantic_index,
        hm3d_semantic_index_inv,
        Goal_score,
    ):

        frontier_score_list[e] = []
        cn = infos[e]["goal_cat_id"] + 4
        cname = infos[e]["goal_name"]
        tpm = len(set(self.map_manager.target_point_map[e].ravel())) - 1

        for lay in range(tpm):
            score = self._evaluate_layer(
                e,
                lay,
                cn,
                cname,
                infos,
                object_norm_inv_perplexity,
                hm3d_category,
                hm3d_semantic_index,
                hm3d_semantic_index_inv,
                Goal_score,
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

    def _evaluate_layer(
        self,
        e,
        lay,
        cn,
        cname,
        infos,
        object_norm_inv_perplexity,
        hm3d_category,
        hm3d_semantic_index,
        hm3d_semantic_index_inv,
        Goal_score,
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

        if len(objs_list) > 0:
            # Convert object categories into tensor indices
            objs_p = torch.tensor(
                [hm3d_semantic_index[obj] for obj in objs_list],
                device=self.device,
            )
            # NOTE: if this doesn't work, try 42, it was using this magic number
            # y_object = (
            #     F.one_hot(objs_p, num_classes=len(hm3d_semantic_index))
            #     .sum(dim=0)
            #     .float()
            # )
            y_object = F.one_hot(objs_p, 42).type(torch.LongTensor)
            y_object = y_object.to(self.device)

            # Calculate scores based on object norm and perplexity
            scores = y_object * object_norm_inv_perplexity.reshape([1, -1])
            # score = scores.sum()

            maxes = torch.max(scores, dim=1).values
            top_max_inds = torch.topk(maxes, max(min((maxes > 0).sum(), 3), 1)).indices
            objs = torch.argmax(scores[top_max_inds], dim=1)
            objs = torch.where(torch.bincount(objs, minlength=len(objs)) > 0)[0]
            # for objs_p in multiset_permutations(np_objs, k_room):
            objs = objs.cpu().numpy()
            objs_n = [hm3d_semantic_index_inv[obj] for obj in objs]

            # Prepare query for language model and get prediction
            query_str = self.lm_service.object_query_constructor(objs_n)
            query_embedding = self.lm_service.embed_sentence(query_str)
            pred = self.ff_net(query_embedding)
            pred = nn.Softmax(dim=1)(pred)

            # Match prediction with the category index
            # category_score = pred[0][hm3d_category.index(cname)].item()
            # NOTE: careful
            # frontier_score_list[e].append(
            #     pred[0][hm3d_category.index(cname)].cpu().numpy()
            # )

            # final_score = score * category_score
            return pred[0][hm3d_category.index(cname)].cpu().numpy()
        else:
            # Default score for areas with no detected semantic objects
            final_score = Goal_score[lay] / max(Goal_score) * 0.1 + 0.1

        return final_score
