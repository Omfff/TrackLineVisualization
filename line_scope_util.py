import cv2
import numpy as np


def get_line_scope(start,end,height, width, img=None,isShow=False):
    """ Obtained pixel col coordinate of each row between the two endpoints of the line.

    Args:
        start: The top end pixel of the line
        end: The bottom end pixel of the line
        height: Height of the frame resolution.
        width: Width of the frame resolution.
        img: Frame.
        isShow: Whether to show the line on windows, which is used for testing.

    Returns:
         A list whose size are the height of the frame.
        The i-th value is the col value of the pixel of line in the i-th row.If the i-th value is zero,
        it means that the line doesn't reach the i-th row.
    """
    if img is None:
        img = np.zeros((height, width), np.uint8)
    cv2.line(img, start, end, color=(255, 255, 255), thickness=1)
    if isShow is True:
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return np.argmax(img, axis=1)


def get_line(start_l, end_l, start_r, end_r, height, width):
    """Obtained pixel col coordinate of each row  between the two endpoints of left and right line.

    Args:
        start_l: The top end pixel of the left line
        end_l: The bottom end pixel of the left line
        start_r: The top end pixel of the right line
        end_r: The bottom end pixel of the right line
        height: Height of the frame resolution.
        width: Width of the frame resolution.

    Returns:
        Two list whose size are the height of the frame.
        The i-th value is the col value of the pixel of line in the i-th row.
    """
    line_left = get_line_scope(start_l, end_l, height, width)
    line_right = get_line_scope(start_r, end_r, height, width)
    check_line_correctness(line_left,line_right)
    return line_left,line_right


def check_line_correctness(line_left,line_right):
    """ To check if two lines' pixel are correct.

    """
    for i in range(len(line_right)):
        if (line_left[i] > line_right[i] and line_left[i] != 0)or \
                (line_right[i] -line_left[i] < 2 and line_left[i] != 0):
            print(i,"line range detect error(",line_left[i] ,line_right[i],")")


def get_curve_by_fitted(curve_left, curve_right, bottom_y, height,cross_t = 2):
    curve_right_param = np.polyfit(curve_right[1], curve_right[0], 2)
    curve_left_param = np.polyfit(curve_left[1], curve_left[0], 2)
    i = height -2
    y = np.arange(0,height-1)
    curve_left_func = np.poly1d(curve_left_param)
    curve_left_x = curve_left_func(y).astype(np.int32)
    curve_right_func = np.poly1d(curve_right_param)
    curve_right_x = curve_right_func(y).astype(np.int32)
    cross_mark = False
    while i >= 0:
        if i > bottom_y or cross_mark:
            curve_left_x[i] = 0
            curve_right_x[i] = 0
            i -= 1
            continue
        else:
            if curve_right_x[i] == curve_left_x[i] or curve_right_x[i] - curve_left_x[i] < cross_t:
                cross_mark =True
            i -= 1

    return curve_left_x,curve_right_x


def get_curve(curve_left, curve_right, height, width, bottom_y, curve_point_count_left, curve_point_count_right, cross_t = 2):
    """ Obtain the coordinates of continuous pixel points by discrete pixel points on the curve

    Args:
        curve_left: Pixel col coordinate of scattered points on the left curve.
        curve_right: Pixel col coordinate of scattered points on the right curve.
        height: Height of the frame resolution.
        width: Width of the frame resolution.
        bottom_y: The bottom boundary of curves.
        cross_t: The threshold at which two lines intersect on the same row.

    Returns:
         line_left, line_right: Two lists whose size are the height of the frame.
                The i-th value is the col value of the pixel of left/right line in the i-th row. If the i-th value is zero,
                it means that the line doesn't reach the i-th row.
    """
    line_left = get_curve_scope(np.delete(curve_left, 2, axis = 0)[:, 0:curve_point_count_left], height, width)
    line_right = get_curve_scope(np.delete(curve_right, 2, axis = 0)[:, 0:curve_point_count_right], height, width)

    cross_mark = False
    i = height -2
    while i >= 0:
        if i > bottom_y or cross_mark or (line_right[i] == 0 and line_left[i] == 0):
            line_left[i] = 0
            line_right[i] = 0
            i -= 1
            continue
        elif line_right[i] == 0 or line_left[i] == 0:
            if line_left[i] == 0 :
                line_left[i] = 1
            else:
                line_right[i] = width - 2
        else:
            if line_right[i] == line_left[i] or line_right[i] - line_left[i] < cross_t:
                print(line_right[i],line_left[i] )
                cross_mark = True
            i -= 1

    check_line_correctness(line_left, line_right)
    return line_left, line_right


def get_curve_scope(curve, height, width):
    """

    Args:
        curve: np.array shape (2,n), n is the number of points on the line
        height: Height of the frame resolution.
        width: Width of the frame resolution.

    Returns:
        A list whose size are the height of the frame.
        The i-th value is the col value of the pixel of line in the i-th row.
    """

    # Reshape  (2,n) to (n,2)
    curve = curve.T
    curve = curve.astype(np.int32)
    img = np.zeros((height, width), np.uint8)
    cv2.polylines(img, [curve],  color=(255, 255, 255), thickness=1 ,isClosed = False)
    return np.argmax(img, axis=1)