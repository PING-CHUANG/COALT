import os
import torch
import pandas
import csv


from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader, hyper_read_image, read_image_train

from lib.train.admin import env_settings

### add
# from lib.test.evaluation.data import SequenceList
import numpy as np

class Hyper16(BaseVideoDataset):

    """ IMEC16 dataset.
    """   

    def __init__(self, root=None, image_loader=read_image_train, split=None, seq_ids=None, data_fraction=None): 
        root = env_settings().hyper16_dir if root is None else root
        super().__init__('Hyper16', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        seqs = os.listdir(os.path.join(self.root))
        list.sort(seqs)
        return seqs

    
    def _read_bb_anno(self, seq_path):
        bb_anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        gt = np.loadtxt(bb_anno_path)
        return torch.tensor(gt, dtype=torch.float32)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:04}.mat'.format(frame_id+1))    # frames start from 1

    # 2024.6.13 lora 6-9bands
    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))


    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames
    
    def get_name(self):
        return 'hyper16'
        


