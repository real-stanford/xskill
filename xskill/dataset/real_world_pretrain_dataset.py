import collections
import concurrent.futures
import random
from collections import namedtuple

import cv2
import numpy as np
import torch

from xskill.dataset.real_data_conversion import \
    real_data_to_replay_buffer

IndexBatch = namedtuple("IndexBatch", "im_q index info")


class RealWorldEpisodeTrajDataset(torch.utils.data.Dataset):
    # @profile
    def __init__(
        self,
        frame_sampler,
        _allowed_dirs=[],
        slide=None,
        seed=None,
        camera_name='camera_2',
        max_get_threads=4,
        read_top_n=None,
        resize_shape=[320, 240],
    ) -> None:
        super().__init__()
        self._frame_sampler = frame_sampler
        self.resize_shape = resize_shape
        self._seed = seed
        self.slide = slide
        self._allowed_dirs = _allowed_dirs
        print(self._allowed_dirs)
        self.in_replay_buffer = {
            dir: real_data_to_replay_buffer(dir,
                                            image_keys=[camera_name],
                                            read_top_n=read_top_n)
            for dir in self._allowed_dirs
        }

        self.seed_rng()
        self._indexfile = {}
        self._build_dir_tree()
        self.camera_name = camera_name
        self.max_get_threads = max_get_threads

    def seed_rng(self):
        if self._seed:
            random.seed(self._seed)

    # @profile
    def _build_dir_tree(self):
        """Build a dict of indices for iterating over the dataset."""
        self._dir_tree = collections.OrderedDict()
        num_vids = 0
        for i, path in enumerate(self._allowed_dirs):
            eps_ends = self.in_replay_buffer[path]['/meta/episode_ends']
            num_eps = len(eps_ends)
            for j in range(num_eps):
                self._indexfile[num_vids] = (path, j)
                num_vids += 1

    def _get_sequence_data(self,
                           sample,
                           image_zarr,
                           eps_begin,
                           eps_len,
                           resize_shape=None):
        sample_index = list(np.array(sample['ctx_idxs']).flatten() + eps_begin)
        frames = [None for _ in range(len(sample['ctx_idxs']))]

        def get_image(image_index, sample_index, frames, image_zarr,
                      resize_shape):
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

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_get_threads) as executor:
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

    def __len__(self):
        return len(self._indexfile)

    # @profile
    def __getitem__(self, idx):
        info = {}
        class_idx, vid_idx = self._indexfile[idx]
        info['class_idx'] = class_idx
        info['vid_idx'] = vid_idx
        cls_data = self.in_replay_buffer[class_idx]
        image_zarr = cls_data[f'/data/{self.camera_name}']
        eps_ends = cls_data['/meta/episode_ends']
        if vid_idx == 0:
            eps_len = eps_ends[0]
            eps_begin = 0
        else:
            eps_len = eps_ends[vid_idx] - eps_ends[vid_idx - 1]
            eps_begin = eps_ends[vid_idx - 1]
        info['eps_begin'] = eps_begin
        info['eps_len'] = eps_len
        sample = self._frame_sampler.sample(np.arange(eps_len))
        # get_time = time.time()
        sequence_data = self._get_sequence_data(
            sample,
            image_zarr,
            eps_begin,
            eps_len,
            resize_shape=self.resize_shape)  # (T,h,w,dim)
        # print("get_end_time",idx,time.time()-get_time)

        im_q = self.transform(sequence_data)
        return IndexBatch(im_q, idx, info)

    # # @profile
    def transform(self, sequence_data):
        # Horig, Worig = sequence_data.shape[1:3]
        sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(
            np.float32)  # (T,dim,h,w)
        sequence_data = sequence_data / 255

        return sequence_data
