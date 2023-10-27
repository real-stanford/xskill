from collections import namedtuple
import numpy as np
import torch
from xskill.utility.file_utils import get_subdirs
from xskill.utility.file_utils import load_image
import random
import collections
import torchvision.transforms as T
import pathlib
import json
import concurrent.futures
import cv2
IndexBatch = namedtuple("IndexBatch", "im_q index info")

class EpisodeTrajDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame_sampler,
        _allowed_dirs=[],
        slide=None,
        seed=None,
        sort_numerical=True,
        vid_mask = None,
        max_get_threads = 4,
        resize_shape=[135, 135],
    ) -> None:
        super().__init__()
        self._frame_sampler = frame_sampler
        self.max_get_threads=max_get_threads
        self.resize_shape=resize_shape
        self._seed = seed
        self.slide = slide
        self.sort_numerical = sort_numerical
        if vid_mask is not None:
            with open(vid_mask, 'r') as f:
                self.vid_mask = json.load(f)
        else:
            self.vid_mask = None

        self._allowed_dirs = _allowed_dirs
        print(self._allowed_dirs)

        self.seed_rng()
        self._indexfile = {}
        self._build_dir_tree()

    def seed_rng(self):
        if self._seed:
            random.seed(self._seed)

    # @profile
    def _build_dir_tree(self):
        """Build a dict of indices for iterating over the dataset."""
        self._dir_tree = collections.OrderedDict()
        num_vids = 0
        for i, path in enumerate(self._allowed_dirs):
            vids = get_subdirs(
                path,
                nonempty=False,
                sort_numerical=True if self.sort_numerical else False,
            )
            if vids:
                vids = np.array(vids)
                if self.vid_mask is not None:
                    vids = vids[self.vid_mask]
                self._dir_tree[path] = vids
                for j, v in enumerate(vids):
                    self._indexfile[num_vids] = (i, j)
                    num_vids += 1


    @property
    def class_names(self):
        """The stems of the allowed video class subdirs."""
        return [str(pathlib.Path(f).stem) for f in self._allowed_dirs]

    def _get_video_path(self, class_idx, vid_idx):
        """Return video paths given class and video indices.

        Args:
        class_idx: The index of the action class folder in the dataset directory
            tree.
        vid_idx: The index of the video in the action class folder to retrieve.

        Returns:
        A path to a video to sample in the dataset.
        """
        action_class = list(self._dir_tree)[class_idx]
        return self._dir_tree[action_class][vid_idx]

    def _get_sequence_data(self, sample,resize_shape=None):
        frame_paths = np.array([str(f) for f in sample["frames"]])
        frame_paths = np.take(frame_paths, sample["ctx_idxs"], axis=0)
        frame_paths = frame_paths.flatten()

        frames = [None for _ in range(len(frame_paths))]

        def get_image(image_index, image_path,frames,resize_shape):
            try:
                if resize_shape is not None:
                    frame = load_image(image_path)
                    resized_frames = cv2.resize(frame, resize_shape)
                    frames[image_index] = resized_frames
                else:
                    frames[image_index] = load_image(image_path)
                return True
            except Exception as e:
                print(image_index,image_path)
                return False

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_get_threads) as executor:
            futures = set()
            for i, idx in enumerate(frame_paths):
                futures.add(
                    executor.submit(get_image, i, idx, frames,resize_shape))

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
        vid_paths = self._get_video_path(class_idx, vid_idx)
        sample = self._frame_sampler.sample(vid_paths)
        sequence_data = self._get_sequence_data(sample,self.resize_shape)  # (T,h,w,dim)

        im_q = self.transform(sequence_data)
        return IndexBatch(im_q, idx, info)


    def transform(self, sequence_data):
        # Horig, Worig = sequence_data.shape[1:3]
        sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(
            np.float32)  # (T,dim,h,w)
        sequence_data = sequence_data / 255

        return sequence_data


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

