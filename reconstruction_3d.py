import numpy as np
import cv2

from os.path import join as osjoin


def calculate_3d_cloud(disparity_map, img, img_number, bm_algorithm, roi,
                       sensor_width, reconstruction_path, camera_params_path):
    if roi is not None:
        (x, y, w, h) = roi
        img = img[y:y + h, x:x + w]
    h, w = disparity_map.shape[:2]
    f = calculate_focal_length(w, sensor_width, camera_params_path)
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, -f],
                    [0, 0, 1, 0]])
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    output_file = osjoin(reconstruction_path,
                         f'reconstructed_{img_number}_{bm_algorithm}.ply')
    create_output(output_points, output_colors, output_file)


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def calculate_focal_length(image_width, sensor_width, camera_params_path):
    focal_length = np.load(osjoin(camera_params_path, 'FocalLengthMm.npy'),
                           allow_pickle=True)
    focal_length_pixel = (focal_length / sensor_width) * image_width
    return focal_length_pixel


