import skimage
from skimage import measure

from constants import hm3d_category
from utils.map_helper import MapManager
import cv2
import numpy as np
from envs.utils.fmm_planner import FMMPlanner


def remove_small_points(local_w, local_ob_map, image, threshold_point, pose):
    selem = skimage.morphology.disk(1)
    traversible = skimage.morphology.binary_dilation(local_ob_map, selem) != True
    planner = FMMPlanner(traversible)
    goal_pose_map = np.zeros((local_ob_map.shape))
    pose_x = int(pose[0].cpu()) if int(pose[0].cpu()) < local_w - 1 else local_w - 1
    pose_y = int(pose[1].cpu()) if int(pose[1].cpu()) < local_w - 1 else local_w - 1
    goal_pose_map[pose_x, pose_y] = 1
    planner.set_multi_goal(goal_pose_map)

    img_label, num = measure.label(image, connectivity=2, return_num=True)
    props = measure.regionprops(img_label)

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = np.zeros(img_label.shape)
    Goal_score = []

    dict_cost = {}
    for i in range(1, len(props)):
        # print("area: ", props[i].area)
        # dist = pu.get_l2_distance(props[i].centroid[0], pose[0], props[i].centroid[1], pose[1])
        dist = (
            planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
        )
        # dist_s = 8 if dist < 300 else 0
        # cost = props[i].area + dist_s
        # NOTE: modified here
        cost = dist

        if props[i].area > threshold_point and dist > 50 and dist < 500:
            dict_cost[i] = cost

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        # print(dict_cost)
        for i, (key, value) in enumerate(dict_cost):
            # print(i, key)
            Goal_edge[img_label == key + 1] = 1
            Goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = (
                i + 1
            )  #
            Goal_score.append(value)
            if i == 3:
                break

    return Goal_edge, Goal_point, Goal_score


class LLMAgentMapManager(MapManager):
    def __init__(self, args, device, num_scenes):
        super().__init__(args, device, num_scenes)

    def handle_stairs_in_validation(self, infos, stair_flag):
        if self.args.eval:
            for e in range(self.num_scenes):
                loc_r, loc_c = self.get_agent_location_in_map(self.local_pose[e])
                if loc_r > self.local_w - 1:
                    loc_r = self.local_w - 1
                if loc_c > self.local_h - 1:
                    loc_c = self.local_h - 1
                if infos[e]["clear_flag"] or self.local_map[e, 18, loc_r, loc_c] > 0.5:
                    stair_flag[e] = 1

    def get_frontiers(self, e, local_pose, kernel_size=(5, 5)):
        # ------------------------------------------------------------------
        ##### Get the frontier map and filter
        # ------------------------------------------------------------------
        _local_ob_map = self.local_map[e][0].cpu().numpy()
        self.local_ob_map[e] = cv2.dilate(_local_ob_map, self.kernel)

        # Extract explored area and apply morphological operations
        show_ex = cv2.inRange(self.local_map[e][1].cpu().numpy(), 0.1, 1)
        self.kernel = np.ones(kernel_size, dtype=np.uint8)
        free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, self.kernel)
        # Detect contours in the explored area
        contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(self.local_ex_map[e], [contour], -1, 1, 1)

        # Clear boundary areas
        self.clear_boundary_areas(self.local_ex_map, e)

        # Calculate target edge map based on obstacle and explored areas
        target_edge = self.calculate_target_edges(e)

        local_pose_map = [
            local_pose[e][1] * 100 / self.args.map_resolution,
            local_pose[e][0] * 100 / self.args.map_resolution,
        ]

        self.target_edge_map[e], self.target_point_map[e], Goal_score = (
            remove_small_points(
                self.local_w,
                _local_ob_map,
                target_edge,
                4,
                local_pose_map,
            )
        )

        # ADDDED OTHER INFO
        wall_edge = self.local_ex_map[e] - target_edge
        contours, hierarchy = cv2.findContours(
            cv2.inRange(wall_edge, 0.1, 1),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 0:
            dst = np.zeros(wall_edge.shape)
            cv2.drawContours(dst, contours, -1, 1, 1)
        edges = cv2.Canny(cv2.inRange(wall_edge, 0.1, 1), 30, 90)
        Wall_lines = cv2.HoughLinesP(
            cv2.inRange(wall_edge, 0.1, 1),
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=10,
            maxLineGap=10,
        )
        ####################################################################
        img_label, num = measure.label(target_edge, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)

        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = []
        Goal_area_list = []
        dict_cost = {}
        for i in range(1, len(props)):
            if props[i].area > 4:
                dict_cost[i] = props[i].area

        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)
            for i, (key, value) in enumerate(dict_cost):
                Goal_edge[img_label == key + 1] = 1
                Goal_point.append(
                    [int(props[key].centroid[0]), int(props[key].centroid[1])]
                )
                Goal_area_list.append(value)
                if i == 3:
                    break

        # clear state
        self.local_ob_map[e] = np.zeros((self.local_w, self.local_h))
        self.local_ex_map[e] = np.zeros((self.local_w, self.local_h))

        return Wall_lines, Goal_area_list, Goal_edge, Goal_point

    def extract_object_infos(self, e):
        map_pred = self.local_map[e]
        semantic_map = map_pred[4:]

        dst = np.zeros(semantic_map[0, :, :].shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        object_list = {}
        for i in range(len(semantic_map)):
            if semantic_map[i, :, :].sum() != 0:
                single_object_list = []
                se_object_map = semantic_map[i, :, :].cpu().numpy()
                se_object_map[se_object_map > 0.1] = 1
                se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
                contours, hierarchy = cv2.findContours(
                    cv2.inRange(se_object_map, 0.1, 1),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_NONE,
                )
                for cnt in contours:
                    if len(cnt) > 30:
                        epsilon = 0.05 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        single_object_list.append(approx)
                        cv2.polylines(dst, [approx], True, 1)
                if len(single_object_list) > 0:
                    object_list[hm3d_category[i]] = single_object_list

        return object_list
