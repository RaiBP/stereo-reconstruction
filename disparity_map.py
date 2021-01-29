import cv2

from filtering import filtering
from os.path import join as osjoin


def calculate_disparity(left, right, img_number, bm_algorithm, parameters,
                        disparity_maps_path, save_disp=True):
    if bm_algorithm == "stereo-sgbm":
        stereo = get_stereo_sgbm_object(parameters)
    else:
        stereo = get_stereo_bm_object(parameters)

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    wls_filtering = parameters["wls_filtering"]

    roi = None
    if wls_filtering:
        lmbda = parameters["lmbda"]
        sigma = parameters["sigma"]
        disparity_map, (x, y, w, h) = filtering(stereo, left_gray, right_gray,
                                                lmbda=lmbda, sigma=sigma)
        disparity_map = disparity_map[y:y + h, x:x + w]
        roi = (x, y, w, h)

    else:
        disparity_map = stereo.compute(left_gray, right_gray)

    disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)

    if save_disp:
        cv2.imwrite(osjoin(
            disparity_maps_path, f'disparity_map_{bm_algorithm}_{img_number}.png'),
            disparity_map)

    return disparity_map, roi


def get_stereo_sgbm_object(bm_parameters):
    min_disp = bm_parameters["min_disp"]
    num_disp = bm_parameters["num_disp"]
    block_size = bm_parameters["block_size"]
    disp_12_max_diff = bm_parameters["disp12maxdiff"]
    speckle_range = bm_parameters["speckle_range"]
    speckle_window_size = bm_parameters["speckle_windows_size"]
    uniqueness = bm_parameters["uniqueness_ratio"]
    prefilter_cap = bm_parameters["prefilter_cap"]
    p1 = bm_parameters["p1"]
    p2 = bm_parameters["p2"]
    use_mode = bm_parameters["use_dynamic_programming"]

    if use_mode == "dp":
        mode = cv2.StereoSGBM_MODE_HH
    elif use_mode == "3way":
        mode = cv2.StereoSGBM_MODE_SGBM_3WAY
    else:
        mode = cv2.StereoSGBM_MODE_SGBM

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=block_size,
                                   P1=p1,
                                   P2=p2,
                                   preFilterCap=prefilter_cap,
                                   disp12MaxDiff=disp_12_max_diff,
                                   uniquenessRatio=uniqueness,
                                   speckleWindowSize=speckle_window_size,
                                   speckleRange=speckle_range,
                                   mode=mode
                                   )
    return stereo


def get_stereo_bm_object(bm_parameters):
    min_disp = bm_parameters["min_disp"]
    num_disp = bm_parameters["num_disp"]
    block_size = bm_parameters["block_size"]
    disp_12_max_diff = bm_parameters["disp12maxdiff"]
    speckle_range = bm_parameters["speckle_range"]
    speckle_window_size = bm_parameters["speckle_windows_size"]
    uniqueness = bm_parameters["uniqueness_ratio"]
    prefilter_cap = bm_parameters["prefilter_cap"]
    prefilter_size = bm_parameters["prefilter_size"]
    texture_threshold = bm_parameters["texture_threshold"]

    prefilter_type = cv2.STEREO_BM_PREFILTER_XSOBEL if bm_parameters["use_xsobel"] else cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    stereo.setPreFilterCap(prefilter_cap)
    stereo.setMinDisparity(min_disp)
    stereo.setDisp12MaxDiff(disp_12_max_diff)
    stereo.setUniquenessRatio(uniqueness)
    stereo.setSpeckleWindowSize(speckle_window_size)
    stereo.setSpeckleRange(speckle_range)
    stereo.setPreFilterSize(prefilter_size)
    stereo.setTextureThreshold(texture_threshold)
    stereo.setPreFilterType(prefilter_type)

    return stereo
