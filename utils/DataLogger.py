import json
import logging
import os
import time
from collections import deque
import numpy as np

LOG_DIR_TEMPLATE = "{}/models/{}/"
DUMP_DIR_TEMPLATE = "{}/dump/{}/"
TRAIN_LOG_FILENAME = "train.log"


class DataLogger:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.log_dir, self.dump_dir = self.setup_logging_directory()
        self.configure_logging()

        self.num_scenes = args.num_processes
        self.num_episodes = int(args.num_eval_episodes)

        # For tracking evals
        self.episode_success = []
        self.episode_success = []
        self.episode_spl = []
        self.episode_dist = []
        if args.eval:
            self.__init_eval_lists()

    def setup_logging_directory(self):
        log_dir = LOG_DIR_TEMPLATE.format(self.args.dump_location, self.args.exp_name)
        dump_dir = DUMP_DIR_TEMPLATE.format(self.args.dump_location, self.args.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(dump_dir, exist_ok=True)
        return log_dir, dump_dir

    def configure_logging(self):
        log_file = os.path.join(self.log_dir, TRAIN_LOG_FILENAME)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        print("Dumping at {}".format(self.log_dir))
        print(self.args)
        logging.info(self.args)

    def __init_eval_lists(self):
        for _ in range(self.args.num_processes):
            self.episode_success.append(deque(maxlen=self.num_episodes))
            self.episode_spl.append(deque(maxlen=self.num_episodes))
            self.episode_dist.append(deque(maxlen=self.num_episodes))

    def update_eval_metrics(self):
        total_success = []
        total_spl = []
        total_dist = []
        if self.args.eval:
            for e in range(self.args.num_processes):
                for acc in self.episode_success[e]:
                    total_success.append(acc)
                for dist in self.episode_dist[e]:
                    total_dist.append(dist)
                for spl in self.episode_spl[e]:
                    total_spl.append(spl)
        return total_success, total_spl, total_dist

    def log_evaluation(
        self,
        step,
        num_scenes,
        end_time,
        g_process_rewards,
        g_sum_rewards,
        g_sum_global,
        episode_metrics,
    ):
        time_elapsed = time.gmtime(end_time - self.start_time)
        log_message = "Time: {0:0=2d}d {1}, num timesteps {2}, FPS {3}, LLM Rewards: {4}, LLM use rate: {5}".format(
            time_elapsed.tm_mday - 1,
            time.strftime("%Hh %Mm %Ss", time_elapsed),
            step * num_scenes,
            int(step * num_scenes / (end_time - self.start_time)),
            g_process_rewards / g_sum_rewards,
            g_sum_rewards / g_sum_global,
        )

        if self.args.eval:
            eval_metrics = self._compute_eval_metrics(episode_metrics)
            log_message += eval_metrics

        logging.info(log_message)
        print(log_message)

    @staticmethod
    def _compute_eval_metrics(episode_metrics):
        total_spl = episode_metrics["total_spl"]
        if total_spl:
            eval_message = " ObjectNav succ/spl/dtg: {:.3f}/{:.3f}/{:.3f}({}), Fail Case: collision/exploration/detection/success: {:.0f}/{:.0f}/{:.0f}/{:.0f}({})".format(
                np.mean(episode_metrics["total_success"]),
                np.mean(episode_metrics["total_spl"]),
                np.mean(episode_metrics["total_dist"]),
                len(episode_metrics["total_spl"]),
                np.sum(episode_metrics["total_collision"]),
                np.sum(episode_metrics["total_exploration"]),
                np.sum(episode_metrics["total_detection"]),
                np.sum(episode_metrics["total_fail_case_success"]),
                len(episode_metrics["total_spl"]),
            )
            return eval_message
        return ""

    def save_to_file(self, success_per_category, spl_per_category):
        # Save the spl per category
        spl_file_path = os.path.join(
            self.log_dir,
            "{}_spl_per_cat_pred_thr.json".format(self.args.split),
        )
        with open(spl_file_path, "w") as f:
            json.dump(spl_per_category, f)

        # Save the success per category
        success_file_path = os.path.join(
            self.log_dir,
            "{}_success_per_cat_pred_thr.json".format(self.args.split),
        )
        with open(success_file_path, "w") as f:
            json.dump(success_per_category, f)

    def log_final_evaluation(
        self,
        g_process_rewards,
        g_sum_rewards,
        g_sum_global,
        total_success,
        total_spl,
        total_dist,
        success_per_category,
        spl_per_category,
    ):
        eval_log = "\n\tLLM Rewards: " + str(g_process_rewards / g_sum_rewards)
        eval_log += "\n\tLLM use rate: " + str(g_sum_rewards / g_sum_global)

        if len(total_spl) > 0:
            eval_log += (
                "Final ObjectNav succ/spl/dtg: {:.3f}/{:.3f}/{:.3f}({}),".format(
                    np.mean(total_success),
                    np.mean(total_spl),
                    np.mean(total_dist),
                    len(total_spl),
                )
            )

        category_log = "Success | SPL per category\n"
        for key in success_per_category:
            category_log += "{}: {} | {}\n".format(
                key,
                sum(success_per_category[key]) / len(success_per_category[key]),
                sum(spl_per_category[key]) / len(spl_per_category[key]),
            )

        full_log = eval_log + "\n" + category_log
        print(full_log)
        logging.info(full_log)

        self.save_to_file(success_per_category, spl_per_category)
