import collections
import json
import os
import os.path as osp
import hydra
import imageio
import numpy as np
import omegaconf
import torch
from PIL import Image
import random
from xskill.env.kitchen.v0 import KitchenAllV0
from collections import deque
import queue
from xskill.dataset.diffusion_bc_dataset import (
    normalize_data,
    unnormalize_data,
)
import plotly.graph_objects as go
from xskill.utility.transform import get_transform_pipeline
import cv2


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(
        eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_images(folder_path, resize_shape=None):
    images = []  # initialize an empty list to store the images

    # get a sorted list of filenames in the folder
    filenames = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    # loop through all PNG files in the sorted list
    for filename in filenames:
        # open the image file using PIL library
        img = Image.open(os.path.join(folder_path, filename))
        # convert the image to a NumPy array
        img_arr = np.array(img)
        if resize_shape is not None:
            img_arr = cv2.resize(img_arr, resize_shape)
        images.append(img_arr)  # add the image array to the list

    # convert the list of image arrays to a NumPy array
    images_arr = np.array(images)
    return images_arr


def convert_images_to_tensors(images_arr, pipeline=None):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    if pipeline is not None:
        images_tensor = pipeline(images_tensor)

    return images_tensor


def load_json(path):
    with open(path, "r") as f:
        file = json.load(f)
        file = np.array(file)
    return file


def load_pretrain_model(eval_cfg):
    device = torch.device("cuda")
    pretrain_cfg = omegaconf.OmegaConf.load(
        os.path.join(eval_cfg.pretrain_path, ".hydra/config.yaml"))
    model = hydra.utils.instantiate(pretrain_cfg.Model).to(device)
    loadpath = os.path.join(eval_cfg.pretrain_path,
                            f"epoch={eval_cfg.pretrain_ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


class visual_diffusion_bc_prediction_callback:

    def __init__(
        self,
        raw_representation=False,
        softmax_prototype=False,
        prototype=False,
        one_hot_prototype=False,
        snap_frames=100,
        task_progess_ratio=None,
        pretain_model_path=None,
        pretrain_model_ckpt=None,
    ) -> None:
        self.raw_representation = raw_representation
        self.softmax_prototype = softmax_prototype
        self.prototype = prototype
        self.one_hot_prototype = one_hot_prototype
        self.snap_frames = snap_frames
        self.task_progess_ratio = task_progess_ratio
        self.env = self.create_env()

        if self.task_progess_ratio is not None:
            self.model = self.load_pretrain_model(pretain_model_path,
                                                  pretrain_model_ckpt)
        else:
            self.model = None

    def load_pretrain_model(self, pretrain_model_path, pretrain_model_ckpt):
        device = torch.device("cuda")
        pretrain_cfg = omegaconf.OmegaConf.load(
            os.path.join(pretrain_model_path, ".hydra/config.yaml"))
        model = hydra.utils.instantiate(pretrain_cfg.Model).to(device)
        loadpath = os.path.join(pretrain_model_path,
                                f"epoch={pretrain_model_ckpt}.ckpt")
        checkpoint = torch.load(loadpath, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        print("pretrain model loaded")
        return model

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def sample_snap(self, proto_data):
        eps_len = len(proto_data)
        snap_idx = random.sample(list(range(eps_len)),
                                 k=min(self.snap_frames, eps_len))
        snap_idx.sort()
        snap = proto_data[snap_idx]  # (snap_frames,D)
        return snap

    def load_demo(self, eval_cfg):
        """
        demo images -> cv2 resize -> torch pipeline
        """
        eval_mask = load_json(eval_cfg.eval_mask_path)
        assert eval_cfg.demo_item in np.arange(len(eval_mask))[eval_mask]
        # load demo
        if self.task_progess_ratio is None:
            if eval_cfg.demo_type == "human":
                demo_emd = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "human_encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "affordance_state_embs.json",
                    ))
                demo_proto = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "human_encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "encode_protos.json",
                    ))
                demo_softmax_proto = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "human_encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "softmax_encode_protos.json",
                    ))
                demo_skill_rep = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "human_encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "traj_representation.json",
                    ))
            else:
                demo_emd = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "affordance_state_embs.json",
                    ))
                demo_proto = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "encode_protos.json",
                    ))
                demo_softmax_proto = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "softmax_encode_protos.json",
                    ))
                demo_skill_rep = load_json(
                    os.path.join(
                        eval_cfg.pretrain_path,
                        "encode_protos",
                        f"ckpt_{eval_cfg.pretrain_ckpt}",
                        f"{eval_cfg.demo_item}",
                        "traj_representation.json",
                    ))

            demo_emd = torch.tensor(demo_emd, dtype=torch.float32).cuda()

        else:
            # sample frames based on ratio
            pretrain_pipeline = get_transform_pipeline(
                eval_cfg.pretrain_pipeline)
            demo_videos = self.load_video(eval_cfg)
            demo_len = len(demo_videos)
            sample_index = (
                np.arange(int(demo_len * self.task_progess_ratio)) /
                self.task_progess_ratio).astype(np.int32)
            demo_videos = demo_videos[sample_index]
            # print("demo video shape", demo_videos.shape)
            images_tensor = convert_images_to_tensors(
                demo_videos, pretrain_pipeline).cuda()
            # print("images_tensor", images_tensor.shape)

            eps_len = images_tensor.shape[0]
            im_q = torch.stack([
                images_tensor[j:j + self.model.slide + 1]
                for j in range(eps_len - self.model.slide)
            ])  # (b,slide+1,c,h,w)
            z = self.model.encoder_q(im_q, None)
            softmax_z = torch.softmax(z / self.model.T, dim=1)
            affordance_emb = self.model.skill_prior(
                im_q[:, :self.model.stack_frames], None)
            state_representation = self.model.encoder_q.get_state_representation(
                im_q, None)
            traj_representation = self.model.encoder_q.get_traj_representation(
                state_representation)
            # demo_skill_rep
            traj_representation = repeat_last_proto(traj_representation,
                                                    eps_len)
            demo_skill_rep = traj_representation.detach().cpu().numpy()
            # demo_proto
            encode_protos = repeat_last_proto(z, eps_len)
            demo_proto = encode_protos.detach().cpu().numpy()
            # demo_softmax_proto
            softmax_encode_protos = repeat_last_proto(softmax_z, eps_len)
            demo_softmax_proto = softmax_encode_protos.detach().cpu().numpy()
            # demo_emd
            demo_emd = affordance_emb.detach().cpu().numpy()

        return demo_emd, demo_proto, demo_softmax_proto, demo_skill_rep

    def load_video(self, eval_cfg):
        # load demo video
        if eval_cfg.demo_type == "human":
            # /local/crv/mengda/air/datasets/kitchen_dataset_v2/human
            demo_videos = load_images(
                os.path.join(eval_cfg.demo_path, "human",
                             f"{eval_cfg.demo_item}"),
                resize_shape=eval_cfg.resize_shape,
            )
        else:
            demo_videos = load_images(
                os.path.join(eval_cfg.demo_path, "robot",
                             f"{eval_cfg.demo_item}"),
                resize_shape=eval_cfg.resize_shape,
            )

        return demo_videos

    def create_env(self):
        env = KitchenAllV0(use_abs_action=True)
        return env

    def eval(self, nets, noise_scheduler, stats, eval_cfg, save_path, seed):
        """
        pretrain resize doesn't matter here.
        use bc_resize to resize the env input to desired size
        """
        self.set_seed(seed)
        device = torch.device("cuda")

        # load demo
        _, demo_proto, demo_softmax_proto, demo_skill_rep = self.load_demo(
            eval_cfg)
        if self.raw_representation:
            proto_snap = self.sample_snap(demo_skill_rep)  # (snap_frames,D)
        elif self.softmax_prototype:
            proto_snap = self.sample_snap(
                demo_softmax_proto)  # (snap_frames,D)
        elif self.prototype:
            proto_snap = self.sample_snap(demo_proto)  # (snap_frames,D)
        elif self.one_hot_prototype:
            pass

        proto_snap = torch.from_numpy(proto_snap).to(device,
                                                     dtype=torch.float32)
        proto_snap = proto_snap.unsqueeze(0)  # (1,snap_frames,D)

        # recording
        imgs = []

        # get first observation
        max_steps = eval_cfg.max_steps
        obs = self.env.reset()
        # keep a queue of last 2 steps of observations
        obs_horizon = eval_cfg.obs_horizon
        img_obs_deque = collections.deque(
            [
                cv2.resize(self.env.render(width=384, height=384),
                           eval_cfg.bc_resize)
            ] * obs_horizon,
            maxlen=obs_horizon,
        )
        # only takes in the joint
        obs_deque = collections.deque([obs[:9]] * obs_horizon,
                                      maxlen=obs_horizon)

        done = False
        step_idx = 0
        rewards = list()
        B = 1

        # track completion order
        task_stack = deque(
            ["slide cabinet", "light switch", "kettle", "microwave"])
        complete_queue = queue.Queue()
        predict_protos = []
        while not done:
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            visual_seq = np.stack(img_obs_deque)
            # only convert to tensor.
            visual_seq = convert_images_to_tensors(visual_seq, None).cuda()
            # print('visual_seq shape', visual_seq.shape)
            visual_feature = nets["vision_encoder"](
                visual_seq)  # (T,visual_feature)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats["obs"])  # (T,obs)
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            # combine visual feature and low dim feature
            obs_feature = torch.cat(
                [visual_feature, nobs],
                dim=-1)  # (T,visual_feature+low_dim_feature)
            # feed in: (1,obs_feature*obs_horizon),(1,snap_frames,D)
            nproto = nets["proto_pred_net"](
                obs_feature.unsqueeze(0).flatten(start_dim=1),
                proto_snap)  # (1,D)
            predict_protos.append(nproto.squeeze(0).detach().cpu().numpy())

            if eval_cfg.upsample_proto:
                upsample_proto = nets["upsample_proto_net"](
                    nproto)  # (1,upsample_dim)

            # infer action
            with torch.no_grad():
                if eval_cfg.upsample_proto:
                    obs_cond = torch.cat(
                        [
                            obs_feature.unsqueeze(0).flatten(start_dim=1),
                            upsample_proto
                        ],
                        dim=1,
                    )
                else:
                    obs_cond = torch.cat([
                        obs_feature.unsqueeze(0).flatten(start_dim=1), nproto
                    ],
                                         dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, eval_cfg.pred_horizon, eval_cfg.action_dim),
                    device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(eval_cfg.num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = nets["noise_pred_net"](sample=naction,
                                                        timestep=k,
                                                        global_cond=obs_cond)

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(model_output=noise_pred,
                                                   timestep=k,
                                                   sample=naction).prev_sample

                # unnormalize action
                naction = naction.detach().to("cpu").numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats["actions"])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + eval_cfg.action_horizon
                action = action_pred[start:end, :]

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, info = self.env.step(action[i])

                    # record complete task
                    for task in info["completed_tasks"]:
                        if task not in list(complete_queue.queue):
                            complete_queue.put(task)

                    # save observations
                    obs_deque.append(obs[:9])
                    raw_env_image = self.env.render(width=384, height=384)
                    raw_env_image = np.array(raw_env_image)
                    raw_env_image = cv2.resize(raw_env_image,
                                               eval_cfg.bc_resize)

                    # save visual obs
                    img_obs_deque.append(raw_env_image.copy())

                    # reward/vis
                    rewards.append(reward)
                    imgs.append(raw_env_image)

                    # update progress bar
                    step_idx += 1
                    if step_idx > max_steps:
                        done = True

        predict_protos = np.array(predict_protos)
        eval_save_path = os.path.join(save_path, "evaluation")
        os.makedirs(eval_save_path, exist_ok=True)
        # save eval gif
        video_save_path = osp.join(eval_save_path, f"eval_{seed}.gif")
        imageio.mimsave(video_save_path, imgs)
        #
        fig = go.Figure()
        D = predict_protos.shape[1]
        for i in range(D):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(predict_protos[:, i])),
                    y=predict_protos[:, i],
                    mode="lines",
                    name=f"proto_{i}",
                ))
        fig.update_layout(title="predict proto",
                          xaxis_title="Iteration",
                          yaxis_title="Value")
        fig.write_image(
            os.path.join(eval_save_path, f"predict_proto_{seed}.png"))

        total_task_completed = set(info["completed_tasks"]).intersection(
            set(["kettle", "light switch", "microwave", "slide cabinet"]))
        order_task_completed_reward = 0
        while not complete_queue.empty() and task_stack:
            task = complete_queue.get()
            if task == task_stack[-1]:
                task_stack.pop()
                order_task_completed_reward += 1
        return len(total_task_completed), order_task_completed_reward
