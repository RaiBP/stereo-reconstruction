import os
import cv2
import glob
import numpy as np

from os.path import join as osjoin

from camera_calibration import find_board_corners


def stereo_calibrate(board_size, square_size, calibration_path,
                     camera_params_path):
    object_pts, left_image_pts, right_image_pts = calculate_image_and_object_points(board_size, calibration_path,
                                                                                    square_size)

    K = np.load(osjoin(camera_params_path, "K.npy"))
    D = np.load(osjoin(camera_params_path, "dist.npy"))
    imgs = glob.glob(osjoin(calibration_path, '*.jpg'))

    image_size = cv2.imread(imgs[0]).shape[:2]

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        object_pts, left_image_pts, right_image_pts, K, D, K, D, image_size)
    print("RMS of stereo calibration: ", ret)

    R1, R2, P1, P2, Q, left_ROI, right_ROI = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9)
    save_stereo_calibration_params(D1, D2, E, F, K1, K2, P1, P2, Q,
                                   R, R1, R2, T, camera_params_path)


def calculate_image_and_object_points(board_size, calibration_path,
                                      square_size,
                                      visualize_corners=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)
    chessboard_finder_flag = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS

    obj_pts_instance = np.zeros((np.prod(board_size), 3),
                                dtype=np.float32)
    obj_pts_instance[:, :2] = np.mgrid[0:board_size[0],
                              0:board_size[1]].T.reshape(-1, 2)

    obj_pts_instance *= square_size

    object_pts = []
    left_image_pts = []
    right_image_pts = []

    pair_images = read_stereo_calibration_images(calibration_path)

    for left_path, right_path in pair_images:
        corners_right, right_image_gray, right_image, ret_right = find_board_corners(board_size, right_path, chessboard_finder_flag)
        corners_left, left_image_gray, left_image, ret_left = find_board_corners(board_size, left_path, chessboard_finder_flag)

        if ret_left and ret_right:
            object_pts.append(obj_pts_instance)

            subpix_right = cv2.cornerSubPix(right_image_gray, corners_right,
                                            (5, 5), (-1, -1), criteria)
            right_image_pts.append(subpix_right)

            subpix_left = cv2.cornerSubPix(left_image_gray, corners_left,
                                           (5, 5), (-1, -1), criteria)
            left_image_pts.append(subpix_left)

            if visualize_corners:
                visualize_stereo_corners(board_size, subpix_left, subpix_right,
                                         left_image, left_path, ret_left,
                                         ret_right, right_image,
                                         calibration_path)
        else:
            print("Chessboard couldn't be detected in image pair: ", left_path,
                  " and ", right_path)

    return object_pts, left_image_pts, right_image_pts


def read_stereo_calibration_images(calibration_path):
    left_images = glob.glob(osjoin(calibration_path, '*_L.jpg'))
    right_images = glob.glob(osjoin(calibration_path, '*_R.jpg'))

    assert len(left_images) == len(
        right_images), "Error: Unequal number of left and right images."

    left_images.sort()
    right_images.sort()
    pair_images = zip(left_images, right_images)
    return pair_images


def visualize_stereo_corners(board_size, corners2_left, corners2_right,
                             left, left_im, ret_left, ret_right, right,
                             calibration_path):
    im1 = cv2.drawChessboardCorners(left, board_size, corners2_left,
                                    ret_left)
    im2 = cv2.drawChessboardCorners(right, board_size, corners2_right,
                                    ret_right)
    im = cv2.hconcat([im1, im2])
    corner_path = osjoin(calibration_path, 'corners')

    if not os.path.isdir(corner_path):
        os.makedirs(corner_path)

    cv2.imwrite(osjoin(corner_path,
                       f'{left_im.split("/")[-1].split("_")[0]}.jpg'), im)


def save_stereo_calibration_params(D1, D2, E, F, K1, K2, P1, P2, Q, R,
                                   R1, R2, T, camera_params_path):
    np.save(osjoin(camera_params_path, "K1"), K1)
    np.save(osjoin(camera_params_path, "D1"), D1)
    np.save(osjoin(camera_params_path, "K2"), K2)
    np.save(osjoin(camera_params_path, "D2"), D2)
    np.save(osjoin(camera_params_path, "R"), R)
    np.save(osjoin(camera_params_path, "T"), T)
    np.save(osjoin(camera_params_path, "E"), E)
    np.save(osjoin(camera_params_path, "F"), F)
    np.save(osjoin(camera_params_path, "R1"), R1)
    np.save(osjoin(camera_params_path, "R2"), R2)
    np.save(osjoin(camera_params_path, "P1"), P1)
    np.save(osjoin(camera_params_path, "P2"), P2)
    np.save(osjoin(camera_params_path, "Q"), Q)
