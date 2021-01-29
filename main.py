import argparse
import cv2
import os
import sys
import json

from camera_calibration import calibrate_camera
from rectify_images import rectify_and_undistort, downsample_images
from disparity_map import calculate_disparity
from reconstruction_3d import calculate_3d_cloud

from os.path import join as osjoin

from stereo_calibration import stereo_calibrate


def read_downsampled_rectified_images(image_number, rectified_images_path):
    left_path = osjoin(rectified_images_path,
                       f'downsampled/{image_number}_L_rectified.png')
    right_path = osjoin(rectified_images_path,
                        f'downsampled/{image_number}_R_rectified.png')

    return cv2.imread(left_path), cv2.imread(right_path)


def read_rectified_images(image_number, rectified_images_path, downsample_image=False):
    left_path = osjoin(rectified_images_path, f'{image_number}_L_rectified.png')
    right_path = osjoin(rectified_images_path, f'{image_number}_R_rectified.png')

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    if downsample_image:
        if left is None or right is None:
            left, right = read_downsampled_rectified_images(image_number,
                                                            rectified_images_path)
        else:
            left, right = downsample_images(left, right)

    return left, right


def read_parameters(image_number, algorithm, bm_parameters_path):
    specific_parameters = os.path.join(bm_parameters_path,
                                       f'parameters_{image_number}_{algorithm}.json')
    default_parameters = os.path.join(bm_parameters_path,
                                      f'parameters_default_{algorithm}.json')

    algorithm_name = "Stereo SGBM" if algorithm == 'stereo-bm' else "Stereo BM"

    if os.path.isfile(specific_parameters):
        print(f"Image specific {algorithm_name} parameters found: {specific_parameters}")
        with open(specific_parameters) as f:
            parameters = json.load(f)
    else:
        print(f"Image specific {algorithm_name} parameters not found in directory {bm_parameters_path}")
        print(f"Opening default {algorithm_name} parameters")
        if os.path.isfile(default_parameters):
            with open(default_parameters) as f:
                parameters = json.load(f)
        else:
            print(f"Error: Default {algorithm_name} parameters not found.")
            sys.exit(1)
    return parameters


def calculate_disparity_map(left, right, image_number, algorithm,
                            bm_parameters_path, disparity_maps_path):
    parameters = read_parameters(image_number, algorithm, bm_parameters_path)
    disparity_map, roi = calculate_disparity(left, right, image_number,
                                             algorithm, parameters,
                                             disparity_maps_path)
    return disparity_map, roi


def check_rectified_images(image_number, rectified_images_path, downsample_image):
    left, right = read_rectified_images(image_number, rectified_images_path,
                                        downsample_image=downsample_image)
    return left is None or right is None


def check_camera_calibration(camera_params_path):
    K_exists = os.path.exists(osjoin(camera_params_path, "K.npy"))
    dists_exists = os.path.exists(osjoin(camera_params_path, "dist.npy"))
    return K_exists and dists_exists


def check_stereo_calibration(camera_params_path):
    K1_exists = os.path.exists(osjoin(camera_params_path, "K1.npy"))
    D1_exists = os.path.exists(osjoin(camera_params_path, "D1.npy"))
    K2_exists = os.path.exists(osjoin(camera_params_path, "K2.npy"))
    D2_exists = os.path.exists(osjoin(camera_params_path, "D2.npy"))
    R1_exists = os.path.exists(osjoin(camera_params_path, "R1.npy"))
    R2_exists = os.path.exists(osjoin(camera_params_path, "R2.npy"))
    P1_exists = os.path.exists(osjoin(camera_params_path, "P1.npy"))
    P2_exists = os.path.exists(osjoin(camera_params_path, "P2.npy"))

    return K1_exists and D1_exists and K2_exists and D2_exists and R1_exists and R2_exists and P1_exists and P2_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 3D mapping from stereo images.')
    parser.add_argument('image_number', metavar='N', type=int,
                        help=f'image number (integer)')
    parser.add_argument('stereo_algorithm', choices=['stereo-bm', 'stereo-sgbm'],
                        help="stereo block matching algorithm, either "
                             "'stereo-bm' or 'stereo-sgbm'")
    parser.add_argument('-ss', '--square_size', type=int, default=2.5,
                        help='chessboard square size, in centimeters')
    parser.add_argument('-sw', '--sensor_width', type=int, default=5.5,
                        help="camera sensor width, in millimeters")
    parser.add_argument('-bs', '--board_size', nargs=2, metavar=('rows', 'cols'),
                        default=(9, 6),
                        help="chessboard size")
    parser.add_argument('-rr', '--read_rectified_images',
                        help="Read rectified images from local folder instead of calculating them from raw images",
                        action="store_true")
    parser.add_argument('-or', '--only_rectify_images',
                        help="Only calculate rectified images from raw images, without calculating 3D reconstruction",
                        action="store_true")
    parser.add_argument('-ds', '--downsample_images',
                        help="Use downsampled images (1/8 factor) ",
                        action="store_true")

    args = parser.parse_args()

    board_size = args.board_size
    square_size = args.square_size
    sensor_width = args.sensor_width
    img_number = args.image_number
    downsample = args.downsample_images
    stereo_algorithm = args.stereo_algorithm
    only_rectify = args.only_rectify_images

    CAMERA_PARAMS_PATH = "./camera_params"
    CALIBRATION_PATH = "./imgs/calibration"
    SINGLE_CALIBRATION_PATH = osjoin(CALIBRATION_PATH, "single_camera")
    STEREO_CALIBRATION_PATH = osjoin(CALIBRATION_PATH, "stereo_camera")
    RAW_IMAGES_PATH = "./imgs/raw"
    RECTIFIED_IMAGES_PATH = "./imgs/rectified"
    DISPARITY_MAPS_PATH = "./imgs/disparity_maps"
    RECONSTRUCTION_3D_PATH = "./3d_reconstruction"
    BM_PARAMETERS_PATH = './bm_parameters'

    if not check_camera_calibration(CAMERA_PARAMS_PATH):
        print("Single camera parameters not found. Calculating...")
        try:
            if not os.path.isdir(CAMERA_PARAMS_PATH):
                os.makedirs(CAMERA_PARAMS_PATH)
            calibrate_camera(board_size, SINGLE_CALIBRATION_PATH, CAMERA_PARAMS_PATH)
            print("Single camera parameters calculated.")
        except TypeError:
            print(f"Error: Camera calibration images not found in directory {SINGLE_CALIBRATION_PATH}")
            sys.exit(1)

    if not check_stereo_calibration(CAMERA_PARAMS_PATH):
        print("Stereo calibration parameters not found. Calculating...")
        try:
            if not os.path.isdir(CAMERA_PARAMS_PATH):
                os.makedirs(CAMERA_PARAMS_PATH)
            stereo_calibrate(board_size, square_size, STEREO_CALIBRATION_PATH,
                             CAMERA_PARAMS_PATH)
            print("Stereo calibration parameters calculated.")
        except IndexError:
            print("Error: Couldn't find any stereo calibration image")
            sys.exit(1)
        except AssertionError:
            print("Error: Unequal number of left and right stereo calibration images.")
            sys.exit(1)

    saved_rectified_dir = osjoin(
        RECTIFIED_IMAGES_PATH, 'downsampled') if downsample else RECTIFIED_IMAGES_PATH
    if not args.read_rectified_images or check_rectified_images(img_number,
                                                                RECTIFIED_IMAGES_PATH,
                                                                downsample):
        try:
            print(f'Calculating rectified images...')
            left_rectified, right_rectified = rectify_and_undistort(img_number, RAW_IMAGES_PATH, RECTIFIED_IMAGES_PATH, CAMERA_PARAMS_PATH, downsample_image=downsample, save_local=True)

            print(f'{img_number}_L_rectified.png and'
                  f' {img_number}_R_rectified.png saved in directory {saved_rectified_dir}')
        except AssertionError:
            print(f"Error: Couldn't find stereo pair {img_number}_L.jpg"
                  f" and {img_number}_R.jpg in directory {RAW_IMAGES_PATH}")
            sys.exit(1)
    else:
        left_rectified, right_rectified = read_rectified_images(img_number, RECTIFIED_IMAGES_PATH, downsample_image=downsample)
        print(f'Rectified images {img_number}_L_rectified.png and'
              f' {img_number}_R_rectified.png retrieved from directory {saved_rectified_dir}')

    if not only_rectify:
        DISPARITY_MAPS_PATH = osjoin(DISPARITY_MAPS_PATH, 'downsampled') if downsample else DISPARITY_MAPS_PATH
        if not os.path.isdir(DISPARITY_MAPS_PATH):
            os.makedirs(DISPARITY_MAPS_PATH)
        print(f'Calculating disparity map...')
        disp_map, roi = calculate_disparity_map(left_rectified, right_rectified,
                                                img_number, stereo_algorithm,
                                                BM_PARAMETERS_PATH,
                                                DISPARITY_MAPS_PATH)
        print(f'Disparity map calculated')
        print(f'disparity_map_{stereo_algorithm}_{img_number}.png saved in directory {DISPARITY_MAPS_PATH}')

        RECONSTRUCTION_3D_PATH = osjoin(RECONSTRUCTION_3D_PATH, 'downsampled') if downsample else RECONSTRUCTION_3D_PATH
        if not os.path.isdir(RECONSTRUCTION_3D_PATH):
            os.makedirs(RECONSTRUCTION_3D_PATH)
        print(f'Calculating 3D cloud...')
        calculate_3d_cloud(disp_map, left_rectified, img_number, stereo_algorithm,
                           roi, sensor_width, RECONSTRUCTION_3D_PATH,
                           CAMERA_PARAMS_PATH)
        print(f'3D cloud calculated')
        print(f'reconstructed_{img_number}_{stereo_algorithm}.ply saved in directory {RECONSTRUCTION_3D_PATH}')
