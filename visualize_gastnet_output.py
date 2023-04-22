import numpy as np
import os
import cv2
import argparse
import toml
import re
import csv
from common.skeleton import Skeleton
from tools.visualization import render_animation
from tools.mpii_coco_h36m import coco_h36m

def get_joints_info():
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                joints_left=[4, 5, 6, 11, 12, 13],
                                joints_right=[1, 2, 3, 14, 15, 16])

    keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M',
                          'num_joints': 17}

    return joints_left, joints_right, h36m_skeleton, keypoints_metadata

def load_csv(file_path):
    """ Returns the two d keypoints in the csv, in the  "coco" format/order,
    as a Tx17x2 numpy array. No scores or labels are included.
    """
    with open(file_path, 'r') as fr:
        reader = csv.DictReader(fr)

        keypoints = []
        # COCO Order
        joint_names = [
            'Nose',
            'LEye',
            'REar',
            'REye',
            'LEar',
            'LShoulder',
            'RShoulder',
            'LElbow',
            'RElbow',
            'LWrist',
            'RWrist',
            'LHip',
            'RHip',
            'LKnee',
            'RKnee',
            'LAnkle',
            'RAnkle',
        ]
        dims = ['x', 'y']
        frame_index = 0
        for row in reader:
            # assume only one skeleton per frame
            joints = []  # list of joints within a frame
            for joint in joint_names:
                # get location
                joint_loc = []
                for dim in dims:
                    fieldname = f"{joint}_{dim}"
                    try:
                        joint_loc.append(float(row[fieldname]))
                    except ValueError:
                        joint_loc.append(np.nan)
                joints.append(joint_loc)

            keypoints.append(joints)
            frame_index += 1
        nparray = np.asarray(keypoints, dtype=np.float32)
        return nparray

def print_video_dims(vid_path):
    cap = cv2.VideoCapture(vid_path)
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(width, height)

def get_amb_id(walk_name, dataset):
    if dataset == "tri":
        regex = re.compile("ID_\d\d")
        match = re.match(regex, walk_name)
        if match:
            _id = int(walk_name[match.end()-2: match.end()])
            return _id - 2
        else:
            raise ValueError(f"Did not find match for {regex} in {walk_name}")
    elif dataset == "lakeside":
        regex = re.compile("AMB\d\d")
        match = re.match(regex, walk_name)
        if match:
            _id = int(walk_name[match.end()-2: match.end()])
            return _id
        else:
            raise ValueError(f"Did not find match for {regex} in {walk_name}")

def get_keypoints(path):
    keypoints = load_csv(path)
    keypoints, _ = coco_h36m(keypoints)
    _, _, _, keypoints_metadata = get_joints_info()
    return keypoints, keypoints_metadata

def orig_video_name(walk_name, dataset, base_path):
    if dataset == "tri":
        amb_id = get_amb_id(walk_name, dataset)
        return os.path.join(base_path, f"AMB{amb_id}/{walk_name}/RGB.avi")
    elif dataset == "lakeside":
        walk_name_no_id = walk_name[6:]
        amb_id = get_amb_id(walk_name, dataset)
        amb_dir = os.path.join(base_path, f"AMB{amb_id}")
        movie_file = os.path.join(amb_dir, f"{walk_name_no_id}.avi")
        if not os.path.isfile(movie_file):
            movie_file = os.path.join(amb_dir, walk_name_no_id, "RGB.avi")
        return movie_file
    elif dataset == "belmont":
        return os.path.join(base_path, f"{walk_name}.mp4.mkv")
    elif dataset == "mdc":
        return os.path.join(base_path, f"{walk_name}.avi")

def visualize(config, walk_name, dataset, base):
    print(dataset, walk_name)
    config = toml.load(f"ambient_configs/{dataset}.toml")
    out_dir = base + config["gastnet_vis_dir"]
    csv_dir = base + config["alphapose_csvs"]
    keypoints, keypoints_metadata = get_keypoints(os.path.join(csv_dir, f"{walk_name}.csv"))
    os.makedirs(out_dir, exist_ok=True)
    gastnet_output = base + config["gastnet_dir"] + f"{walk_name}.npy"
    output_video =  os.path.join(out_dir, f"{walk_name}.mp4")
    orig_video_dir = base + config["orig_video_dir"]
    orig_video = orig_video_name(walk_name, dataset, orig_video_dir)
    #print_video_dims(orig_video)
    width, height = config["vid_dims"]


    prediction = np.load(gastnet_output)
    h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                 joints_left=[4, 5, 6, 11, 12, 13],
                                 joints_right=[1, 2, 3, 14, 15, 16])

    print('Rendering ...')
    anim_output = {'Reconstruction': prediction}
    render_animation(keypoints, keypoints_metadata, anim_output, h36m_skeleton, 25, 3000,
                     np.array(70., dtype=np.float32), output_video, limit=-1, downsample=1, size=5,
                     input_video_path=orig_video, viewport=(width, height), input_video_skip=0)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--belmont", action="store_true")
    parser.add_argument("-t", "--tri", action="store_true")
    parser.add_argument("-l", "--lakeside", action="store_true")
    parser.add_argument("-m", "--mdc", action="store_true")
    args = parser.parse_args()
    base = "/mnt/win_share/AMBIENT/"

    if args.tri:
        walk_names = [
            "2018_03_11__01_58_39_ID_44_state_3",
            ]
        config = toml.load("ambient_configs/tri.toml")
        for walk_name in walk_names:
            visualize(config, walk_name, "tri", base)

    if args.lakeside:
        config = toml.load("ambient_configs/lakeside.toml")
        walk_names = [
            "AMB77.2019_06_12__13_58_33_ID_78_state_4",
            "AMB83.2020_01_02__14_58_29_ID_91_state_3",
            ]
        for walk_name in walk_names:
            visualize(config, walk_name, "lakeside", base)

    if args.belmont:
        config = toml.load("ambient_configs/belmont.toml")
        walk_names = [
            "BELMONT05-top.mp4_backward_1",
        ]
        for walk_name in walk_names:
            visualize(config, walk_name, "belmont", base)

    if args.mdc:
        config = toml.load("ambient_configs/mdc.toml")
        walk_names = [
            "vid0031_0333-backward_1",
            "vid0413_4197 (Visit 1- Meds OFF DBS OFF)-forward_1",
        ]
        for walk_name in walk_names:
            visualize(config, walk_name, "mdc", base)