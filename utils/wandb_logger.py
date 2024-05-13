import numpy as np
import torch
import wandb


def get_dict_from_args_str(args_str: str) -> dict[str, str]:
    """Converts a comma-separated string of key-value pairs into a dictionary.

    Args:
        args_str (str): Comma-separated string of key-value pairs.

    Returns:
        dict[str, str]: Converted dictionary with corresponding key-value pairs.
    """
    args_dict = {}
    for key_val in args_str.split(","):
        assert (
            "=" in key_val
        ), "Wrong format of wandb_args. Make sure it has the following format: key1=value1,key2=value2"
        k, v = key_val.split("=")
        args_dict[k] = v

    assert (
        "project" in args_dict
    ), "Wrong format of wandb_args. Missing value for `project`."

    return args_dict


class WandbLogger:
    def __init__(self, args):
        r"""A Wrapper for wandb. It initializes a wandb run and logs videos from numpy images using a configuration string.

        Args:
            args_str.
        """
        # Start a new wandb run
        self.run = wandb.init(
            **get_dict_from_args_str(args.wandb_args), config=vars(args)
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.run.finish()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        r"""Write video into wandb from image frames.

        Args:
            video_name: Name of the video.
            step_idx: Checkpoint index to be displayed.
            images: List of n frames. Each frame is a np.ndarray of shape (H, W, 3).
            fps: Frames per second for the output video.

        Returns:
            None.
        """
        # Convert numpy arrays to tensor and add batch and channel dimensions
        frame_tensors = [
            torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors)).unsqueeze(
            0
        )  # Shape: (1, n, 3, H, W)

        # Log video to wandb
        wandb.log(
            {video_name: wandb.Video(video_tensor.numpy(), fps=fps, format="mp4")},
            step=step_idx,
        )

    def log_evaluation(self, step, data, commit=True):
        wandb.log(data, step=step, commit=commit)
