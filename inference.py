import torch
import numpy as np
import cv2
import os
import argparse
import csv
import toml

from tools.mpii_coco_h36m import coco_h36m
from common.camera import normalize_screen_coordinates, camera_to_world
from common.generators import *
from model.gast_net import SpatioTemporalModel
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
# GOAL: Read csvs with 2d keypoints, run GastNet, write 3D keypoints to npy files
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
# Note: this rotation is not correct for our camera angles, but we don't care right now



def parse_args():
    parser = argparse.ArgumentParser(description='Inference Script')

    # General arguments: these default values are good for us
    parser.add_argument('-f', '--frames', type=int, default=27, metavar='NAME',
                        help='The number of receptive fields')
    parser.add_argument('-ca', '--causal', action='store_true',
                        help='Using real-time model with causal convolution')
    parser.add_argument('-w', '--weight', type=str, default='27_frame_model.bin', metavar='NAME',
                        help='The name of model weight')
    parser.add_argument('-n', '--num-joints', type=int, default=17, metavar='NAME',
                        help='The number of joints')
    # parser.add_argument('-k', '--keypoints-file', type=str, default='./data/keypoints/baseball.json', metavar='NAME',
                       #  help='The path of keypoints file')
    # parser.add_argument('-vi', '--video-path', type=str, default='./data/video/baseball.mp4', metavar='NAME',
    #                     help='The path of input video')
    #parser.add_argument('-vo', '--viz-output', type=str, default='./output/baseball.mp4', metavar='NAME',
    #                    help='The path of output video')
    parser.add_argument('-kf', '--kpts-format', type=str, default='csv', metavar='NAME',
                        help='The format of 2D keypoints. "csv" for custom csv')

    return parser

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

def evaluate(test_generator, model_pos, joints_left, joints_right, return_predictions=False):
    """Actually run inference on a set of 2D keypoints, otpionally returning the first each result"""
    with torch.no_grad():
        model_pos.eval()

        for _, _, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

def reconstruction(args, csv_dir, output_dir, video_dims, overwrite=True):
    """
    Generate 3D poses from 2D keypoints detected from video, and write to npy file
    Default width and height are correct for TRI dataset.
    If overwrite is False, check if file already exists and don't recompute
    """
    # prepare model
    filter_widths = [3, 3, 3]
    channels = 128
    video_width, video_height = video_dims

    h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                 joints_left=[4, 5, 6, 11, 12, 13],
                                 joints_right=[1, 2, 3, 14, 15, 16])
    adj = adj_mx_from_skeleton(h36m_skeleton)
    model_pos = SpatioTemporalModel(adj, args.num_joints, 2, args.num_joints, filter_widths=filter_widths,
                                    channels=channels, dropout=0.05, causal=args.causal)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    # load pretrained model
    print('Loading checkpoint', args.weight)
    chk_file = os.path.join('./checkpoint/gastnet', args.weight)
    checkpoint = torch.load(chk_file, map_location=lambda storage, _: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    if args.causal:
        causal_shift = pad
    else:
        causal_shift = 0

    os.makedirs(output_dir, exist_ok=True)
    # Getting joint information
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            walk_name = os.path.splitext(file)[0]
            output_file = os.path.join(output_dir, walk_name + ".npy")
            if (not overwrite) and os.path.isfile(output_file):
                print(f"Skipping {file} as results already exists.")
                continue
            print(f'Loading 2D keypoints at {file}...')
            keypoints = load_csv(os.path.join(csv_dir, file))
            assert len(keypoints.shape) == 3

            # Transform the keypoints format from COCO to h36m format
            keypoints, valid_frames = coco_h36m(keypoints)

            # Get the width and height of video
            '''
            cap = cv2.VideoCapture(args.video_path)
            width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))'''

            # normalize keypoints
            input_keypoints = normalize_screen_coordinates(keypoints[..., :2], w=video_width, h=video_height)

            print(f'Reconstructing...')
            gen = UnchunkedGenerator(None, None, [input_keypoints[valid_frames]],
                            pad=pad, causal_shift=causal_shift, augment=False,
                            )
            prediction  = evaluate(gen, model_pos, None, None, return_predictions=True)
            prediction = camera_to_world(prediction, R=rot, t=0)

            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

            prediction_new = np.zeros((*input_keypoints.shape[:-1], 3), dtype=np.float32)
            prediction_new[valid_frames] = prediction


            print(f'Writing to file {output_file}...')
            with open(output_file, 'w') as f:
                np.save(output_file, prediction_new)

if __name__ == '__main__':
    parser = parse_args()
    parser.add_argument("-b", "--belmont", action="store_true")
    parser.add_argument("-t", "--tri", action="store_true")
    parser.add_argument("-l", "--lakeside", action="store_true")
    parser.add_argument("-m", "--mdc", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_false")
    args = parser.parse_args()

    base = "/mnt/win_share/AMBIENT/"
    if args.belmont:
        print("Belmont")
        config = toml.load('ambient_configs/belmont.toml')
        belmont_csvs = base + config["alphapose_csvs"]
        belmont_output_dir = base + config["gastnet_dir"]
        belmont_vid_dims = config["vid_dims"]
        reconstruction(args, belmont_csvs, belmont_output_dir, belmont_vid_dims, overwrite=args.overwrite)

    if args.mdc:
        print("MDC")
        config = toml.load('ambient_configs/mdc.toml')
        mdc_csvs = base + config["alphapose_csvs"]
        mdc_output_dir = base + config["gastnet_dir"]
        mdc_vid_dims = config["vid_dims"]
        reconstruction(args, mdc_csvs, mdc_output_dir, mdc_vid_dims, overwrite=args.overwrite)

    if args.lakeside:
        print("Lakeside")
        config = toml.load('ambient_configs/lakeside.toml')
        lakeside_csvs = base + config["alphapose_csvs"]
        lakeside_output_dir = base + config["gastnet_dir"]
        lakeside_vid_dims = config["vid_dims"]
        reconstruction(args, lakeside_csvs, lakeside_output_dir, lakeside_vid_dims, overwrite=args.overwrite)

    if args.tri:
        print("TRI")
        config = toml.load('ambient_configs/tri.toml')
        tri_csvs = base + config["alphapose_csvs"]
        tri_output_dir = base + config["gastnet_dir"]
        tri_vid_dims = config["vid_dims"]
        reconstruction(args, tri_csvs, tri_output_dir, tri_vid_dims, overwrite=args.overwrite)