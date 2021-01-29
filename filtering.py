import cv2
import numpy as np


def filtering(left_matcher, left, right, lmbda=8000, sigma=1.0):
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_disp = left_matcher.compute(left, right)
    right_disp = right_matcher.compute(left, right)

    left_disp_int16 = np.int16(left_disp)
    right_disp_int16 = np.int16(right_disp)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(left_disp_int16, left, disparity_map_right=right_disp_int16)

    return (filtered_disp / 16.0).astype(np.uint8), wls_filter.getROI()
