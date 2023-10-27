import numpy as np
import os
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import torch
import omegaconf
from tqdm import tqdm
from xskill.utility.transform import get_transform_pipeline
import zarr
import concurrent.futures
from pathlib import Path
import cv2


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(
        eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_model(cfg):
    exp_cfg = omegaconf.OmegaConf.load(
        os.path.join(cfg.exp_path, '.hydra/config.yaml'))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f'epoch={cfg.ckpt}.ckpt')
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(cfg.device)
    model.eval()
    print("model loaded")
    return model


def convert_images_to_tensors(images_arr, pipeline):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    images_tensor = pipeline(images_tensor)

    return images_tensor


def get_sequence_data(sample, image_zarr, eps_begin, resize_shape=None):
    sample_index = list(np.array(sample) + eps_begin)
    frames = [None for _ in range(len(sample))]

    def get_image(image_index, sample_index, frames, image_zarr, resize_shape):
        try:
            if resize_shape is not None:
                frame = image_zarr[sample_index]
                resized_frames = cv2.resize(frame, resize_shape)
                frames[image_index] = resized_frames
            else:
                frames[image_index] = image_zarr[sample_index]
            return True
        except Exception as e:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = set()
        for i, idx in enumerate(sample_index):
            futures.add(
                executor.submit(get_image, i, idx, frames, image_zarr,
                                resize_shape))

        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to get image!')

    sequence_data = np.stack(frames)  # Shape: (S * X, H, W, C)
    return sequence_data


@hydra.main(version_base=None,
            config_path="../../config/realworld",
            config_name="label_real_kitchen_dataset")
def label_dataset(cfg: DictConfig):

    model = load_model(cfg)
    pretrain_pipeline = get_transform_pipeline(cfg.augmentations)

    robot_dataset = hydra.utils.instantiate(cfg.robot_dataset)
    human_dataset = hydra.utils.instantiate(cfg.human_dataset)
    resize_shape = cfg.resize_shape

    # create zarr
    # save_path = os.path.join(cfg.exp_path, f'ckpt_{cfg.ckpt}',
    #                          'prototype.zarr')
    save_path = os.path.join(cfg.save_path, f'ckpt_{cfg.ckpt}',
                             'prototype.zarr')
    prototype_store = zarr.DirectoryStore(save_path)
    prototype_zarr = zarr.group(prototype_store)

    for embodiment in ['human', 'robot']:
        dataset_to_label = robot_dataset if embodiment == 'robot' else human_dataset
        for key, zarr_data in tqdm(dataset_to_label.in_replay_buffer.items(),
                                   desc="labelling task"):

            eps_end = zarr_data['/meta/episode_ends'][:]
            image_zarr = zarr_data['/data/camera_2']
            print(f"{key} image shape: {image_zarr.shape}")
            print(f"eps end: {eps_end}")
            z_store = []
            softmax_z_store = []
            raw_rep_store = []
            for i in tqdm(range(len(eps_end)), desc="labelling episode"):
                if i == 0:
                    eps_start_index = 0
                    eps_end_index = eps_end[i]
                else:
                    eps_start_index = eps_end[i - 1]
                    eps_end_index = eps_end[i]
                # Downsample!
                sample = np.arange(eps_end_index -
                                   eps_start_index)[::cfg.frequency]
                images_arr = get_sequence_data(sample, image_zarr,
                                               eps_start_index, resize_shape)
                images_tensor = convert_images_to_tensors(
                    images_arr, pretrain_pipeline).cuda()  #(T,c,h,w)
                eps_len = images_tensor.shape[0]
                im_q = torch.stack([
                    images_tensor[j:j + model.slide + 1]
                    for j in range(eps_len - model.slide)
                ])  # (b,slide+1,c,h,w)
                z = model.encoder_q(im_q, None)
                z = repeat_last_proto(z, eps_len)
                softmax_z = torch.softmax(z / model.T, dim=1)

                state_representation = model.encoder_q.get_state_representation(
                    im_q, None)
                traj_representation = model.encoder_q.get_traj_representation(
                    state_representation)
                traj_representation = repeat_last_proto(
                    traj_representation, eps_len)

                z_store.append(z.detach().cpu().numpy())
                softmax_z_store.append(softmax_z.detach().cpu().numpy())
                raw_rep_store.append(
                    traj_representation.detach().cpu().numpy())

            eps_end = np.cumsum([len(zs) for zs in z_store]).astype(np.int64)
            z_store = np.concatenate(z_store).astype(np.float32)
            softmax_z_store = np.concatenate(softmax_z_store).astype(
                np.float32)
            raw_rep_store = np.concatenate(raw_rep_store).astype(np.float32)
            save_key = Path(key).name

            prototype_zarr.require_dataset(
                f'{embodiment}/{save_key}/prototypes',
                shape=z_store.shape,
                dtype=np.float32)
            prototype_zarr.require_dataset(
                f'{embodiment}/{save_key}/softmax_prototypes',
                shape=softmax_z_store.shape,
                dtype=np.float32)
            prototype_zarr.require_dataset(f'{embodiment}/{save_key}/raw_rep',
                                           shape=raw_rep_store.shape,
                                           dtype=np.float32)
            prototype_zarr.require_dataset(f'{embodiment}/{save_key}/eps_end',
                                           shape=eps_end.shape,
                                           dtype=np.int64)

            prototype_zarr[f'{embodiment}/{save_key}/prototypes'] = z_store
            prototype_zarr[
                f'{embodiment}/{save_key}/softmax_prototypes'] = softmax_z_store
            prototype_zarr[f'{embodiment}/{save_key}/raw_rep'] = raw_rep_store
            prototype_zarr[f'{embodiment}/{save_key}/eps_end'] = eps_end


if __name__ == "__main__":
    label_dataset()
