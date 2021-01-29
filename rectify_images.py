import numpy as np
import cv2
import matplotlib.pyplot as plt

from os.path import join as osjoin


def rectify_and_undistort(img_number, raw_images_path, rectified_images_path,
                          camera_params_path, downsample_image=True,
                          save_local=True):
    K1, D1, K2, D2, R1, R2, P1, P2 = load_stereo_calibration_parameters(camera_params_path)

    left = cv2.imread(osjoin(raw_images_path, f'{img_number}_L.jpg'))
    right = cv2.imread(osjoin(raw_images_path, f'{img_number}_R.jpg'))

    assert left is not None and right is not None

    height, width = left.shape[:2]

    left_rectified = rectify_and_undistort_single_image(left, K1, D1,
                                                        R1, P1, (width, height))
    right_rectified = rectify_and_undistort_single_image(right, K2, D2,
                                                         R2, P2, (width, height))

    return save_rectified_images(downsample_image, img_number, left_rectified,
                                 right_rectified, rectified_images_path,
                                 save_local)


def load_stereo_calibration_parameters(camera_params_path):
    K1 = np.load(osjoin(camera_params_path, "K1.npy"))
    D1 = np.load(osjoin(camera_params_path, "D1.npy"))
    K2 = np.load(osjoin(camera_params_path, "K2.npy"))
    D2 = np.load(osjoin(camera_params_path, "D2.npy"))
    R1 = np.load(osjoin(camera_params_path, "R1.npy"))
    R2 = np.load(osjoin(camera_params_path, "R2.npy"))
    P1 = np.load(osjoin(camera_params_path, "P1.npy"))
    P2 = np.load(osjoin(camera_params_path, "P2.npy"))
    return K1, D1, K2, D2, R1, R2, P1, P2


def rectify_and_undistort_single_image(img, K, D, R, P, size):
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
    img_rectified = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                              cv2.BORDER_CONSTANT)
    return img_rectified


def save_rectified_images(downsample_image, img_number, left_rectified,
                          right_rectified, rectified_images_path,
                          save_local):
    if downsample_image:
        left_downsampled, right_downsampled = downsample_images(left_rectified,
                                                                right_rectified)

        if save_local:
            cv2.imwrite(osjoin(rectified_images_path,
                               f'downsampled/{img_number}_L_rectified.png'),
                        left_downsampled)
            cv2.imwrite(osjoin(rectified_images_path,
                               f'downsampled/{img_number}_R_rectified.png'),
                        right_downsampled)
        return left_downsampled, right_downsampled
    if save_local:
        cv2.imwrite(osjoin(rectified_images_path,
                           f'{img_number}_L_rectified.png'), left_rectified)
        cv2.imwrite(osjoin(rectified_images_path,
                           f'{img_number}_R_rectified.png'), right_rectified)
    return left_rectified, right_rectified


def downsample_images(left_rectified, right_rectified):
    h, w = left_rectified.shape[:2]

    w_down = int(w / 8)
    h_down = int(h / 8)
    dim = (w_down, h_down)
    left_downsampled = cv2.resize(left_rectified, dim,
                                  interpolation=cv2.INTER_AREA)
    right_downsampled = cv2.resize(right_rectified, dim,
                                   interpolation=cv2.INTER_AREA)
    return left_downsampled, right_downsampled


def plot_rectified_images(h_new, left_rectified, right_rectiied):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(left_rectified)
    axes[1].imshow(right_rectiied)
    for i in range(0, h_new, 400):
        axes[0].axhline(i)
        axes[1].axhline(i)
    plt.suptitle("Rectified images")
    plt.show()
