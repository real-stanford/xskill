import os
import pickle
import uuid
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf
from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv
import random


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="skill_transfer_composing",
)
def train_diffusion_bc(cfg: DictConfig):
    # create save dir
    unique_id = str(uuid.uuid4())
    save_dir = os.path.join(cfg.save_dir, unique_id)
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")
    # Set up logger
    wandb.init(project=cfg.project_name)
    wandb.config.update(OmegaConf.to_container(cfg))

    #set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    dataset = hydra.utils.instantiate(cfg.dataset)
    # save training data statistics (min, max) for each dim
    stats = dataset.stats
    # open a file for writing in binary mode
    with open(os.path.join(save_dir, "stats.pickle"), "wb") as f:
        # write the dictionary to the file
        pickle.dump(stats, f)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=cfg.pin_memory,
        # don't kill worker process afte each epoch
        persistent_workers=cfg.persistent_workers,
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch["obs"].shape)
    print("batch['actions'].shape", batch["actions"].shape)
    print("batch['protos'].shape", batch["protos"].shape)
    print("batch['proto_snap'].shape", batch["proto_snap"].shape)
    print("batch['images'].shape", batch["images"].shape)

    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = cfg.vision_feature_dim

    # observation and action dimensions corrsponding to
    # the output of PushTEnv
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    proto_dim = cfg.proto_dim

    # create network object
    if cfg.upsample_proto:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon +
            proto_horizon * cfg.upsample_proto_net.out_size,
        )
    else:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon + proto_horizon * proto_dim,
        )
    proto_pred_net = hydra.utils.instantiate(
        cfg.proto_pred_net,
        input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
    )

    # the final arch has 3 parts
    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "proto_pred_net": proto_pred_net,
        "noise_pred_net": noise_pred_net,
    })

    if cfg.upsample_proto:
        upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
        nets["upsample_proto_net"] = upsample_proto_net

    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    # device transfer
    device = torch.device("cuda")
    _ = nets.to(device)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(model=nets, power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(params=nets.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * cfg.num_epochs,
    )

    # create eval callbacl
    eval_callback = hydra.utils.instantiate(cfg.eval_callback)

    for epoch_idx in range(cfg.num_epochs):
        epoch_loss = list()
        epoch_action_loss = list()
        epoch_proto_prediction_loss = list()

        # batch loop
        for nbatch in dataloader:
            # data normalized in dataset
            # device transfer

            # (B, obs_horizon, obs_dim)
            nobs = nbatch["obs"].to(device)
            B = nobs.shape[0]
            # (B, obs_horizon, 3,112,112)
            nimage = nbatch["images"].to(device)
            # (B, 1, model_dim)
            nproto = nbatch["protos"].to(device)
            proto_snap = nbatch["proto_snap"].to(device)
            proto_snap = proto_snap.reshape(B, dataset.snap_frames, -1)
            naction = nbatch["actions"].to(device)

            # encoder vision features
            image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
            image_features = image_features.reshape(
                *nimage.shape[:2], -1)  # (B,obs_horizon,visual_feature)

            obs_feature = torch.cat(
                [image_features, nobs],
                dim=-1)  # (B,obs_horizon,low_dim_feature+visual_feature)
            # predict the proto: (B,obs_horizon*(low_dim_feature+visual_feature))),(B,snap_frames,D)
            predict_proto = proto_pred_net(obs_feature.flatten(start_dim=1),
                                           proto_snap)

            # (B, proto_horizon, obs_dim)
            nobs = nobs[:, :obs_horizon, :]

            if cfg.upsample_proto:
                upsample_proto = upsample_proto_net(
                    nproto.flatten(start_dim=1))
                upsample_proto = upsample_proto.reshape(
                    B, cfg.proto_horizon, -1)  # (B,proto_horizon,upsample_dim)
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        upsample_proto.flatten(start_dim=1),
                    ],
                    dim=1,
                )
            else:
                # feed in: (B,obs_feature*obs_horizon),(B,snap_frame,D)
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        nproto.flatten(start_dim=1)
                    ],
                    dim=1,
                )

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps, (B, ),
                device=device).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(naction, noise,
                                                      timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(noisy_actions,
                                        timesteps,
                                        global_cond=obs_cond)

            # L2 loss
            action_loss = nn.functional.mse_loss(noise_pred, noise)
            proto_prediction_loss = nn.functional.mse_loss(
                predict_proto, nproto.squeeze(1))
            loss = action_loss + proto_prediction_loss

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets)

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            epoch_action_loss.append(action_loss.item())
            epoch_proto_prediction_loss.append(proto_prediction_loss.item())

        wandb.log({
            "epoch loss":
            np.mean(epoch_loss),
            "epoch action loss":
            np.mean(epoch_action_loss),
            "epoch proto prediction loss":
            np.mean(epoch_proto_prediction_loss),
        })

        if epoch_idx % cfg.ckpt_frequency == 0:
            torch.save(
                ema.averaged_model.state_dict(),
                os.path.join(save_dir, f"ckpt_{epoch_idx}.pt"),
            )
        # if epoch_idx%cfg.eval_cfg.eval_frequency==0 and epoch_idx>1:
        if epoch_idx % cfg.eval_cfg.eval_frequency == 0:
            for demo_type in cfg.demo_type_list:
                cfg.eval_cfg.demo_type = demo_type
                for task_progess_ratio in cfg.task_progess_ratio_list:
                    if demo_type == "robot":
                        task_progess_ratio = 1
                    print(f"evaluating {demo_type} ratio:{task_progess_ratio}")
                    # set task progress ratio
                    eval_callback.task_progess_ratio = task_progess_ratio
                    total_rewards = []
                    order_rewards = []
                    if epoch_idx == 0:
                        n_evaluations = 1
                    else:
                        n_evaluations = cfg.eval_cfg.n_evaluations
                    for seed in range(n_evaluations):
                        total_r, order_r = eval_callback.eval(
                            ema.averaged_model,
                            noise_scheduler,
                            stats,
                            cfg.eval_cfg,
                            save_dir,
                            seed,
                        )
                        total_rewards.append(total_r)
                        order_rewards.append(order_r)
                    wandb.log({
                        f"eval_score/{demo_type}_{task_progess_ratio}_total reward":
                        np.mean(total_rewards),
                        f"eval_score/{demo_type}_{task_progess_ratio}_order reward":
                        np.mean(order_rewards),
                    })
                    if demo_type == "robot":
                        break


if __name__ == "__main__":
    train_diffusion_bc()
