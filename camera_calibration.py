import cv2
import numpy as np
import glob
import PIL.ExifTags
import PIL.Image

from tqdm import tqdm
from os.path import join as osjoin


def calibrate_camera(board_size, calibration_path, camera_params_path):
    object_pts = []
    image_pts = []

    obj_pts_instance = np.zeros((np.prod(board_size), 3), dtype=np.float32)
    obj_pts_instance[:, :2] = np.mgrid[0:board_size[0],
                              0:board_size[1]].T.reshape(-1, 2)

    calibration_paths = glob.glob(osjoin(calibration_path, '*'))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_gray = np.array(0)
    for image_path in tqdm(calibration_paths):
        corners, img_gray, img, ret = find_board_corners(board_size, image_path, None)
        if ret:
            print(f"Chessboard detected at {image_path}!")
            cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), criteria)
            object_pts.append(obj_pts_instance)
            image_pts.append(corners)
        else:
            print(f"Couldn't detect chessboard at {image_path}")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_pts, image_pts,
                                                     img_gray.shape[::-1],
                                                     None, None)
    save_focal_length(calibration_paths, camera_params_path)
    save_calibration_parameters(K, dist, image_pts, object_pts, ret, rvecs,
                                tvecs, camera_params_path)


def save_focal_length(calibration_paths, camera_params_path):
    img = PIL.Image.open(calibration_paths[0])
    exif = {PIL.ExifTags.TAGS[key]: value
                 for key, value in img._getexif().items()
                 if key in PIL.ExifTags.TAGS}

    focal_length_exif = exif['FocalLength']  # mm

    np.save(osjoin(camera_params_path, "FocalLengthMm"), focal_length_exif)


def find_board_corners(board_size, image_path, flags):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_gray, board_size, flags)
    return corners, img_gray, img, ret


def save_calibration_parameters(K, dist, image_pts, object_pts, ret, rvecs,
                                tvecs, camera_params_path):
    np.save(osjoin(camera_params_path, "ret"), ret)
    np.save(osjoin(camera_params_path, "K"), K)
    np.save(osjoin(camera_params_path, "dist"), dist)
    np.save(osjoin(camera_params_path, "rvecs"), rvecs)
    np.save(osjoin(camera_params_path, "tvecs"), tvecs)
    np.save(osjoin(camera_params_path, "img_points"), image_pts)
    np.save(osjoin(camera_params_path, "obj_points"), object_pts)
