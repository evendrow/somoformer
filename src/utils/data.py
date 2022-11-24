import json
import glob
import os
import pickle
import torch

import numpy as np
from PIL import Image
from utils.utils import path_to_data

def tracks_for_sequence(annotations):
    """ For a list annotations, creates a dictionary of tracks, each
        corresponding to a list of annotations within that track
    """
    tracks = {}
    for frame in annotations:
        if frame['num_keypoints'] == 0:
            continue
        if frame['track_id'] not in tracks:
            tracks[frame['track_id']] = []
        tracks[frame['track_id']].append(frame)
    return tracks


def valid_tracks_for_sequence(annotations):
    """ For a list annotations, creates a dictionary of tracks, each
        corresponding to a list of annotations within that track.

        We additionally filter tracks to make sure that each track is
        strictly consecutive, so that there are no frame jumps.

    """
    tracks = tracks_for_sequence(annotations)
    valid_tracks = {}

    for track_id, track in tracks.items():
        # We check if the track id list is a list of consecutive integers
        # If not, we discard it. This is maybe not the most efficient thing
        # to do, but it's simple. Experimentally, 3773 / 3839 tracks are
        # consecutive, so we're not losing much anyway.
        all_image_ids = [frame['rel_image_id'] for frame in track]
        if (
            len(all_image_ids) == len(set(all_image_ids)) and          # ensure uniqueness
            all_image_ids == sorted(all_image_ids) and                 # ensure sortedness
            len(all_image_ids) == all_image_ids[-1]-all_image_ids[0]+1 # ensure consecutivity
           ):
            valid_tracks[track_id] = track
    return valid_tracks


def load_data_3dpw_multiperson(split):
    # TRAIN AND TEST SETS ARE REVERSED FOR SOMOF
    SPLIT_3DPW = {
        "train": "test",
        "val": "validation",
        "valid": "validation",
        "test": "train"
    }
    datalist = []

    for pkl in os.listdir(path_to_data('3dpw', 'sequenceFiles', SPLIT_3DPW[split])):
        with open(path_to_data('3dpw', 'sequenceFiles', SPLIT_3DPW[split], pkl), 'rb') as reader:
            annotations = pickle.load(reader, encoding='latin1')

        all_person_tracks = []
        for actor_index in range(len(annotations['genders'])):

            joints_2D = annotations['poses2d'][actor_index].transpose(0, 2, 1)
            joints_3D = annotations['jointPositions'][actor_index]
            
            track_joints = []
            track_mask = []

            for image_index in range(len(joints_2D)): # range(t1, t2):
                path = path_to_data('3dpw', 'imageFiles', os.path.splitext(pkl)[0], f"image_{str(image_index).zfill(5)}.jpg")
                J_3D_real = joints_3D[image_index].reshape(-1, 3)
                J_3D_mask = np.ones(J_3D_real.shape[:-1])
                track_joints.append(J_3D_real)
                track_mask.append(J_3D_mask)

            all_person_tracks.append((np.asarray(track_joints), np.asarray(track_mask)))

        datalist.append(all_person_tracks)

    return datalist


def load_data_somof(split="train", db="3dpw"):
    datalist = []
    masks_in = None
    masks_out = None

    with open(path_to_data('somof', f'{db}_{split}_in.json')) as f:
        frames_in = np.asarray(json.load(f))

    if db == "posetrack":
        with open(path_to_data('somof', f'{db}_{split}_masks_in.json')) as f:
            masks_in = np.asarray(json.load(f))

    if split == "test":
        frames_out = None
        mask = None
    else:
        with open(path_to_data('somof', f'{db}_{split}_out.json')) as f:
            frames_out = np.asarray(json.load(f))
        if db == "posetrack":
            with open(path_to_data('somof', f'{db}_{split}_masks_out.json')) as f:
                masks_out = np.asarray(json.load(f))

    return frames_in, frames_out, masks_in, masks_out
