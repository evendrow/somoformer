import copy
import json
import math
import numpy as np
import random
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.utils import path_to_data
from utils.data import load_data_3dpw_multiperson, load_data_somof
from torchvision import transforms

def collate_batch(batch):
    joints_list = []
    masks_list = []
    num_people_list = []
    for joints, masks in batch:
        # Make sure first dimension is # people for single person case
        if len(joints.shape) == 3:
            joints = joints.unsqueeze(0)
            masks = masks.unsqueeze(0)

        joints_list.append(joints)
        masks_list.append(masks)
        num_people_list.append(torch.zeros(joints.shape[0]))

    joints = pad_sequence(joints_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    padding_mask = pad_sequence(num_people_list, batch_first=True, padding_value=1).bool()

    return joints, masks, padding_mask


def batch_process_joints(joints, masks, padding_mask, config,training=False, multiperson=True):
    joints = joints.to(config["DEVICE"])
    masks = masks.to(config["DEVICE"])
    
    if multiperson and len(joints.shape) == 4:
        joints = joints.unsqueeze(1)
        masks = masks.unsqueeze(1)

    in_F = config["TRAIN"]["input_track_size"]

    if multiperson:
        if config["DATA"]["joints"] == "somof" or config['DATA']['joints'] == 'posetrack':
            in_joints_pelvis = joints[:,:, (in_F-1):in_F, 6:7, :].clone()
        elif config["DATA"]["joints"] == "cmu":
            in_joints_pelvis = joints[:,:, (in_F-1):in_F , 12:13 ,:].clone()
    else:
        if config["DATA"]["joints"] == "somof" or config['DATA']['joints'] == 'posetrack':
            in_joints_pelvis = joints[:, (in_F-1):in_F , 6:7 ,:].clone()
        else:
            in_joints_pelvis = torch.zeros_like(in_joints_pelvis)
        
    joints -= in_joints_pelvis

    if multiperson:
        B, N, F, J, K = joints.shape
        joints = joints.transpose(1, 2).reshape(B, F, N*J, K)
        in_joints_pelvis = in_joints_pelvis.reshape(B, 1, N, K)
        masks = masks.transpose(1, 2).reshape(B, F, N*J)

    # If training, can do augmentations
    if training:
        if config["DATA"]["aug_rotate"]:
            joints = getRandomRotatePoseTransform(config)(joints)
        if config["DATA"]["aug_scale"]:
            joints = getRandomScaleTransform()(joints)
        if "aug_permute" in config["DATA"] and config["DATA"]["aug_permute"]:
            joints, masks, padding_mask = getRandomPermuteOrder(joints, masks, padding_mask)
    

    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]   
    in_joints = joints[:,:in_F].float()
    out_joints = joints[:,in_F:in_F+out_F].float()
    in_masks = masks[:,:in_F].float()
    out_masks = masks[:,in_F:in_F+out_F].float()

    return in_joints, in_masks, out_joints, out_masks, in_joints_pelvis.float(), padding_mask.float()




def getRandomScaleTransform(r1=0.8, r2=1.2):
    def do_scale(x):
        #scale = (r1 - r2) * torch.rand(1) + r2
        scale = (r1 - r2) * torch.rand(x.shape[0]).reshape(-1, 1, 1, 1) + r2
        return x * scale.to(x.device)
    return transforms.Lambda(lambda x: do_scale(x))


def getRandomPermuteOrder(joints, masks, padding_mask):
    """
    Randomly permutes persons across the input token dimension. This helps
    expose all learned embeddings to a variety of poses.
    """
    
    def do_permute(joints, masks, padding_mask):
        B, N = padding_mask.shape
        B, F, NJ, K = joints.shape
        J = NJ // N
        
        perm = torch.argsort(torch.rand(B, N), dim=-1).reshape(B, N)
        idx = torch.arange(B).unsqueeze(-1)
        
        joints = joints.view(B, F, N, J, K).transpose(1, 2)[idx, perm]
        joints = joints.transpose(1, 2).reshape(B, F, N*J, K)
        
        masks = masks.view(B, F, N, J).transpose(1, 2)[idx,perm]
        masks = masks.transpose(1, 2).reshape(B, F, N*J)
        
        padding_mask = padding_mask[idx, perm]
        
        return joints, masks, padding_mask

    return do_permute(joints, masks, padding_mask)


def getRandomRotatePoseTransform(config):
    """
    Performs a random rotation about the origin (0, 0, 0)
    """

    def do_rotate(pose_seq):
        """
        pose_seq: torch.Tensor of size (B, S, J, 3) where S is sequence length, J is number 
            of joints, and the last dimension is the coordinate
        """
        B, F, J, K = pose_seq.shape

        angles = torch.deg2rad(torch.rand(B)*360)

        rotation_matrix = torch.zeros(B, 3, 3).to(pose_seq.device)
        rotation_matrix[:,1,1] = 1
        rotation_matrix[:,0,0] = torch.cos(angles)
        rotation_matrix[:,0,2] = torch.sin(angles)
        rotation_matrix[:,2,0] = -torch.sin(angles)
        rotation_matrix[:,2,2] = torch.cos(angles)

        rot_pose = torch.bmm(pose_seq.reshape(B, -1, 3).float(), rotation_matrix)
        rot_pose = rot_pose.reshape(pose_seq.shape)
        return rot_pose

    return transforms.Lambda(lambda x: do_rotate(x))


class MultiPersonPoseDataset(torch.utils.data.Dataset):
    SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
    COCO_TO_SOMOF = [6, 12, 7, 13, 8, 14, 0, 3, 9, 4, 10, 5, 11]


    def __init__(self, name, split="train", track_size=30, track_cutoff=16, segmented=True,
                 add_flips=False, frequency=1):
        """
            name: The name of the dataset (e.g. "somof")
            split: one of ['train', 'valid', 'test']
            add_flips: whether to add flipped sequences to data as well (data augmentation)
            mode: one of ['inference', 'eval'] In eval mode, will not do train data augmentation.
            frequency: How often to take a frame (i.e. distance between frames). For example, if
                       frequency=2, will take every other frame.
        """
        self.name = name
        self.split = split
        self.track_size = track_size
        self.track_cutoff = track_cutoff
        self.segmented = segmented
        self.frequency = frequency
        self.add_flips = add_flips

        self.initialize()
        
    def load_data(self):
        raise NotImplementedError("Dataset load_data() method is not implemented.")

    def initialize(self):
        self.load_data()

        if self.segmented:
            tracks = []
            for scene in self.datalist:
                for seg, j in enumerate(range(0, len(scene[0][0]) - self.track_size * self.frequency + 1, self.track_size)):
                    people = []
                    for person in scene:
                        start_idx = j
                        end_idx = start_idx + self.track_size * self.frequency
                        J_3D_real, J_3D_mask = person[0][start_idx:end_idx:self.frequency], person[1][
                                                                                           start_idx:end_idx:self.frequency]
                        people.append((J_3D_real, J_3D_mask))
                    tracks.append(people)
            self.datalist = tracks

        # If we're on a train split, do some additional data augmentation as well
        if self.add_flips:
            print("doing some flips for " + self.name + ", " + self.split + " split")
            # for each sequence, we can also add a "flipped" sequence
            flipped_datalist = []
            for seq in self.datalist:
                flipped_seq = []
                for J_3D_real, J_3D_mask in seq:
                    J_3D_flipped = torch.flip(J_3D_real, dims=(0,)).clone()
                    J_3D_mask_flipped = torch.flip(J_3D_mask, dims=(0,)).clone()
                    flipped_seq.append((J_3D_flipped, J_3D_mask_flipped))
                flipped_datalist.append(flipped_seq)
        
            self.datalist += flipped_datalist

        if not self.segmented:
            # create a mapping from idx to which track/frame to look at
            # prevents extra computation at dataset time
            frame_count = 0
            self.track_map = []
            for i, scene in enumerate(self.datalist):
                track_frames = len(scene[0][0]) - self.track_size * self.frequency + 1
                for k in range(0, track_frames):
                    self.track_map.append((i, k))
                frame_count += track_frames

    def __len__(self):
        if self.segmented:
            return len(self.datalist)
        else:
            return sum([len(scene[0][0]) - self.track_size * self.frequency + 1 for scene in self.datalist])

    def __getitem__(self, idx):
        if self.segmented:
            scene = self.datalist[idx]

        else:
            # We want to count the idx-th valid frame in our list of tracks
            # A valid frame is any frame that has at least (track_size-1)
            # frames ahead of it (i.e. it can be used as the start frame)
            track_idx, frame_idx = self.track_map[idx]

            scene = []
            for person in self.datalist[track_idx]:
                J_3D_real = person[0][
                            frame_idx:frame_idx + self.track_size * self.frequency:self.frequency]
                J_3D_mask = person[1][
                            frame_idx:frame_idx + self.track_size * self.frequency:self.frequency]
                scene.append((J_3D_real, J_3D_mask))

        J_3D_real = torch.stack([s[0] for s in scene])
        J_3D_mask = torch.stack([s[1] for s in scene])

        return J_3D_real, J_3D_mask


class SoMoFDataset(MultiPersonPoseDataset):
    SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
    COCO_TO_SOMOF = [6, 12, 7, 13, 8, 14, 0, 3, 9, 4, 10, 5, 11]

    def __init__(self, **args):
        super(SoMoFDataset, self).__init__("somof", frequency=1, **args)

    def load_data(self):
        data_in, data_out, _, _ = load_data_somof(split=self.split)

        data = np.concatenate((data_in, data_out), axis=2)
        data = torch.from_numpy(data)
        data = data.reshape((*data.shape[:-1], 13, 3)) # (N, 30, 2, 13, 3)

        self.num_kps = 13
        self.datalist = [[(person, torch.ones(person.shape[:-1])) for person in track] for track in data]

class ThreeDPWDataset(MultiPersonPoseDataset):
    def __init__(self, **args):
        super(ThreeDPWDataset, self).__init__("3dpw", frequency=2, **args)

    def load_data(self):
        self.data = load_data_3dpw_multiperson(split=self.split)

        self.datalist = []
        for scene in self.data:
            people = [(torch.from_numpy(joints)[:,self.SOMOF_JOINTS],
                       torch.from_numpy(mask)[:,self.SOMOF_JOINTS]) for joints, mask in scene]
            self.datalist.append(people)


def create_dataset(dataset_name, logger, **args):
    logger.info("Loading dataset " + dataset_name)
    
    if dataset_name == "3dpw":
        dataset = ThreeDPWDataset(**args)
    elif dataset_name == "somof":
        dataset = SoMoFDataset(**args)
    else:
        raise ValueError(f"Dataset with name '{dataset_name}' not found.")
        
    logger.info(f"Loaded {len(dataset)} annotations for " + dataset_name)
    return dataset


def get_datasets(datasets_list, config, logger):
    """ 
    Returns a list of torch dataset objects
    datasets_list: [String]
    """
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    datasets = []
    for dataset_name in datasets_list:
        datasets.append(create_dataset(dataset_name, logger, split="train", track_size=(in_F+out_F), track_cutoff=in_F, segmented=config['DATA']['segmented'], add_flips=config['DATA']['add_flips']))
    return datasets