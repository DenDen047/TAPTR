import argparse
import gradio as gr
import cv2
import os
import numpy as np
import pickle
from pprint import pprint
import torch
from gradio_image_prompter import ImagePrompter
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.cuda.amp import autocast

from models.dino.taptr import build_taptr
from util.slconfig import DictAction, SLConfig
from moviepy.editor import ImageSequenceClip
from datasets.kubric import make_temporal_transforms, get_aux_target_hacks_list
from datasets.tapvid import resize_video


def get_args():
    from main import get_args_parser
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--path_ckpt", type=str,
                        default="./checkpoints/taptr.pth")
    parser.add_argument("--video_path", type=str,
                        required=True, help="Path to the input video")
    parser.add_argument("--points", type=str, required=True,
                        help="Tracking points in format 'x1,y1;x2,y2;...'")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Frame to start tracking from")
    args = parser.parse_args()

    # Load config
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError(f"Key {k} can be used by args only")

    return args


class Demo():
    def __init__(self, args):
        self.args = args
        self.model, _, _ = build_taptr(self.args)
        self.load_model(self.args.path_ckpt)
        self.model.eval().cuda()
        self.aux_target_hacks = get_aux_target_hacks_list("val", args)
        self.transforms = make_temporal_transforms("val", fix_size=args.fix_size, strong_aug=False, args=args)
        self.num_query_per_video = args.num_queries_per_video_eval

    def load_model(self, path_ckpt):
        checkpoint = torch.load(path_ckpt, map_location="cpu")
        model_state_dict = {}
        for name, value in checkpoint["ema_model"].items():
            model_state_dict[name.replace("module.", "")] = value
        self.model.load_state_dict(model_state_dict)

    def process_one_video(self, video, fps, points, start_tracking_frames):
        """
        1. padding input points.
        2. sent to model to get results.
        3. plot the results in the video.
        return: the ploted video.
        Args:
            video (_type_): _description_
        """
        # TODO: process video using model.
        n_frames, height, width = video.shape[:3]  # video.shape[2], video.shape[1]
        if n_frames > 700:
            print("\nWarning, Too long video, clipping to 700 frames!"*10)
            video = video[:700]
            # return None
        video = torch.FloatTensor(video)
        input_video = torch.FloatTensor(resize_video(video.cpu().numpy(), (512, 512))).permute(0,3,1,2)
        points = np.array(points)[..., :2] / np.array([[width, height]]) * 512
        queries = self.prepare_queries(512, 512, points.tolist(), n_frames, start_tracking_frames)
        input_video, queries_input, _ = self.align_format(input_video / 255.0, queries)
        input_video = input_video.cuda()
        queries_input = {k: v.cuda() for k, v in queries_input.items()}
        with torch.no_grad():
            with autocast():
                outputs, queries_input_ = self.model([input_video], [queries_input])
        pred_locations, pred_occluded = self.get_tracking_points(
            outputs,
            video,
            queries_input["num_real_pt"].cpu().item()
        )
        processed_video = self.plot_video(video, outputs, queries_input["num_real_pt"].cpu().item(), queries_input["query_frames"][:queries_input["num_real_pt"].cpu().item()].cpu())
        save_video_path = get_video_name(dir_save=self.args.output_dir)
        save_video(processed_video, save_video_path, fps)
        return save_video_path, pred_locations.detach().cpu().numpy(), pred_occluded.detach().cpu().numpy()

    def align_format(self, images, targets):
        if self.transforms is not None:
            images, targets = self.transforms(images, targets)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                targets, images = hack_runner(targets, img=images)
        seq_name = targets.pop("seq_name")
        targets = {
            "pt_boxes": targets.pop("boxes"),
            "pt_labels": targets.pop("labels"),
            "pt_tracking_mask": targets.pop("tracking_mask"),
            "num_real_pt": targets.pop("num_real_pt"),
            "query_frames": targets.pop("query_frames"),
        }
        return images, targets, seq_name

    def prepare_queries(self, width, height, points, n_frames, start_tracking_frames, num_padding=-1):
        """prepare the points to be tracked.

        Args:
            width (int): width of the video
            height (int):  height of the video
            points (list of list): N, 6(xy, ...)

        Returns:
            _type_: _description_
        """
        query_xy = np.array(points)[:, :2]
        num_query_ = query_xy.shape[0]
        if (self.num_query_per_video > 0) and (query_xy.shape[0] < self.num_query_per_video):
            np.random.seed(0)  # for reproducibility
            # xy
            margin_x = width // 16
            margin_y = height // 16
            min_distance = 1
            num_padding = self.num_query_per_video - query_xy.shape[0]
            # distant points
            def generate_coordinates(original_coords, N, lower_bound=0, upper_bound=255, distance=3):
                coordinates = original_coords.tolist()
                while len(coordinates) < N:
                    new_coordinate = np.random.randint(lower_bound, upper_bound+1, size=2)
                    if all(np.linalg.norm(new_coordinate - c) >= distance for c in coordinates):
                        coordinates.append(new_coordinate)
                return np.array(coordinates[-N:])
            padding_query_xy = generate_coordinates(query_xy, num_padding, lower_bound=margin_x, upper_bound=width-1-margin_x, distance=min_distance)
            padding_query_xy = padding_query_xy[:,None].repeat(n_frames, axis=1)
            query_xy = query_xy[:,None].repeat(n_frames, axis=1)
            query_xy = np.concatenate((query_xy, padding_query_xy), axis=0)
            # occ
            padding_query_occ = np.ones((num_padding, n_frames)) < 0
            # random restart all
            num_random_occ_frames = min(12, n_frames)
            padding_query_occ[:, :num_random_occ_frames] = \
                padding_query_occ[:, :num_random_occ_frames] | \
                np.arange(num_random_occ_frames)[None,:].repeat(padding_query_occ.shape[0], 0) < np.random.randint(0, num_random_occ_frames, padding_query_occ.shape[0])[:,None]
            query_occ = np.arange(n_frames)[None,:].repeat(num_query_, 0)
            query_occ = query_occ < np.array(start_tracking_frames)[:, None]
            query_occ = np.concatenate((query_occ, padding_query_occ), axis=0)
        start_tracking_frames = torch.argmax(torch.tensor(1-query_occ), axis=1)
        tracking_mask = torch.arange(n_frames)[None, :].repeat(query_xy.shape[0], 1)
        tracking_mask = tracking_mask >= start_tracking_frames[:, None]
        queries = {
                "points": torch.from_numpy(query_xy).float(),
                'occluded': torch.from_numpy(query_occ),
                'num_frames': n_frames,
                'sampled_frame_ids': torch.arange(n_frames),
                'tracking_mask': tracking_mask,
                'query_frames': start_tracking_frames,
                'sampled_point_ids': torch.arange(query_xy.shape[0]),
                "num_real_pt": torch.tensor([num_query_]),
                'seq_name': "demo_video",
            }
        return queries

    def get_tracking_points(self, outputs, video, num_queries, threshold_occ=0.5):
        len_temp, H, W, _ = video.shape
        # [len_temp, n_query, 3]
        pred_visibilities = outputs["full_seq_output"]["pred_logits"][0][:, :num_queries]
        # [n_query, len_temp]
        pred_occluded = pred_visibilities[..., 1].sigmoid() > threshold_occ
        # [len_temp, n_query, 2]
        pred_locations = outputs["full_seq_output"]["pred_boxes"][0, :, :num_queries, :2]
        # [len_temp, n_query, 2]
        pred_locations = pred_locations * torch.tensor([W, H]).float().cuda()[None, None, :]

        return pred_locations, pred_occluded

    def plot_video(self, video, outputs, num_queries, start_tracking_frames):
        """plot the outputs on the video.

        Args:
            video (torch.tensor): len h w 3
            outputs (_type_): _description_
            num_queries (_type_): _description_
        """
        threshold_occ = 0.5
        pred_locations, pred_occluded = self.get_tracking_points(outputs, video, num_queries, threshold_occ)
        video_plotted = draw_tracks_on_video(
            video,
            pred_locations, pred_occluded,
            start_tracking_frames,
            mode="rainbow",
            tracks_leave_trace=50
        )
        return video_plotted


def draw_tracks_on_video(
    video: torch.Tensor,
    tracks: torch.Tensor,
    visibility: torch.Tensor = None,
    query_frames: int = 0,
    compensate_for_camera_motion=False,
    mode="rainbow",
    tracks_leave_trace = -1,
):
    """_summary_

    Args:
        video (torch.Tensor): len_temp, H, W, 3
        tracks (torch.Tensor): len_temp, n_query, 2
        visibility (torch.Tensor, optional): len_temp, n_query
        segm_mask (torch.Tensor, optional): _description_. Defaults to None.
        query_frame (int, optional): _description_. Defaults to 0.
        compensate_for_camera_motion (bool, optional): _description_. Defaults to False.
        mode (str, optional): _description_. Defaults to "rainbow".
        tracks_leave_trace (int, optional): _description_. Defaults to -1.
    """

    def _draw_pred_tracks(
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x N x 2
        visibs: np.ndarray,  # T x N
        draw_flag: np.ndarray,  # N
        vector_colors: np.ndarray,
        alpha: float = 0.5,
        tracks_leave_trace: int = 0,
        point_size = 2,
    ):
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                if not draw_flag[i]:
                    continue
                if not visibs[s, i]:
                    continue
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        point_size,
                        cv2.LINE_AA,
                    )
            if tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    if mode == "rainbow":
        color_map = cm.get_cmap("gist_rainbow")
    elif mode == "cool":
        color_map = cm.get_cmap(mode)

    T, H, W, C = video.shape
    _, N, D = tracks.shape

    assert D == 2
    assert C == 3
    video = video.byte().detach().cpu().numpy()  # S, H, W, C
    tracks = tracks.long().detach().cpu().numpy()  # S, N, 2

    res_video = []

    # process input video
    for rgb in video:
        res_video.append(rgb.copy())

    vector_colors = np.zeros((T, N, 3))
    if mode == "rainbow":
        start_locations = torch.gather(torch.tensor(tracks), 0, query_frames[None, :, None].repeat(1,1,2))[0]
        y_min, y_max = (
            start_locations[..., 1].min(),
            start_locations[..., 1].max(),
        )
        norm = plt.Normalize(y_min, y_max)
        for n in range(N):
            color = color_map(norm(start_locations[n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)
    else:
        # color changes with time
        for t in range(T):
            color = np.array(color_map(t / T)[:3])[None] * 255
            vector_colors[t] = np.repeat(color, N, axis=0)

    point_size = 3
    #  draw tracks
    if tracks_leave_trace != 0:
        for t in range(1, T):
            first_ind = (
                max(0, t - tracks_leave_trace)
                if tracks_leave_trace >= 0
                else 0
            )
            curr_tracks = tracks[first_ind : t + 1]
            curr_visibs = visibility[first_ind : t + 1]
            curr_colors = vector_colors[first_ind : t + 1]
            if compensate_for_camera_motion:
                raise NotImplementedError("compensate for camera motion is not implemented.")
                # diff = (
                #     tracks[first_ind : t + 1, segm_mask <= 0]
                #     - tracks[t : t + 1, segm_mask <= 0]
                # ).mean(1)[:, None]

                # curr_tracks = curr_tracks - diff
                # curr_tracks = curr_tracks[:, segm_mask > 0]
                # curr_colors = curr_colors[:, segm_mask > 0]

            draw_flag = t >= query_frames
            res_video[t] = _draw_pred_tracks(
                res_video[t],
                curr_tracks,
                curr_visibs,
                draw_flag,
                curr_colors,
                tracks_leave_trace=tracks_leave_trace,
                point_size=point_size,
            )

    #  draw points
    for t in range(T):
        for i in range(N):
            if t < query_frames[i]:
                continue
            coord = (tracks[t, i, 0], tracks[t, i, 1])
            visibile = True
            if visibility is not None:
                visibile = visibility[t, i]
            if coord[0] != 0 and coord[1] != 0:
                if not compensate_for_camera_motion:
                # or (
                #     compensate_for_camera_motion and segm_mask[i] > 0
                # )
                    if visibile:
                        cv2.circle(
                            res_video[t],
                            coord,
                            int(point_size * 2),
                            vector_colors[t, i].tolist(),
                            thickness=-1 if visibile else point_size,
                        )
                else:
                    raise NotImplementedError("compensate for camera motion is not implemented.")

    #  construct the final rgb sequence
    # if self.show_first_frame > 0:
    #     res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
    return torch.from_numpy(np.stack(res_video)).byte()


def get_video_name(dir_save="./saved_videos", prefix="processed_video_"):
    import os
    os.makedirs(dir_save, exist_ok=True)
    files = os.listdir(dir_save)
    file_orders = [int(file.split(".")[0][len(prefix):]) for file in files if file.endswith(".mp4") and file.startswith(prefix)]
    if len(file_orders) == 0:
        return os.path.join(dir_save, prefix + "0.mp4")
    else:
        return os.path.join(dir_save, prefix + str(max(file_orders) + 1) + ".mp4")


def save_video(video, save_path, fps):
    """save a video into a specified file.

    Args:
        video (torch.tensor): n h w c
        filename (string): _description_
    """
    wide_list = list(video.unbind(0))
    wide_list = [wide.cpu().numpy() for wide in wide_list]
    clip = ImageSequenceClip(wide_list, fps=fps)
    clip.write_videofile(save_path, codec="libx264", fps=fps, logger=None)
    print(f"Video saved to {save_path}")
    return save_path


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames), fps


def video_to_interaction(input_video, frame_selector):
    """
    1. get the frame from input_video.
    2. get the interaction from the frame.
    3. return the interaction.
    Args:
        input_video (_type_): _description_
        frame_selector (_type_): _description_
        interaction (_type_): _description_
    """
    # frame_selector = int(700 * frame_selector)
    frame_selector = int(frame_selector)
    frames = cv2.VideoCapture(input_video)
    interaction = None
    # fps = int(frames.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    if not frames.isOpened():
        print("Error opening video file")
    else:
        while frames.isOpened():
            ret, frame = frames.read()
            print("Getting the interaction frame: ", frame_id, frame_selector)
            if frame_id > 700:
                print(f"Too long video ({input_video}) for the demo! - video_to_interaction")
                return None
            if frame_id == frame_selector:
                interaction = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                break
            frame_id += 1
        frames.release()
    if interaction is None:
        raise ValueError("Frame not found")
    return {"image": interaction, "points": []}


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    demo = Demo(args)

    # Read video
    video, fps = read_video_from_path(args.video_path)

    # Parse points
    points = [list(map(float, p.split(','))) for p in args.points.split(';')]
    start_tracking_frames = [args.start_frame] * len(points)

    # Process video
    output_video_fpath, pred_locations, pred_occluded = demo.process_one_video(
        video, fps, points, start_tracking_frames)
    print(f"Output video saved to {output_video_fpath}")

    # save outputs to a pickle file
    output_fpath = os.path.join(args.output_dir, "output.pkl")
    with open(output_fpath, "wb") as f:
        pickle.dump({
            'locations': pred_locations,
            'occluded': pred_occluded,
        }, f)


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0 python demo_cli.py \
#     --config_file /content/TAPTR/config/TAPTR.py \
#     --path_ckpt /content/drive/MyDrive/pretrained_models/TAPTR/taptr.pth \
#     --video_path /content/input.mp4 \
#     --points "800,200;800,400" \
#     --start_frame 0 \
#     --output_dir /content/output
