import cv2
import numpy as np
import math
from yaml_reader import BaseParam
from ransac_line import fit_line_by_ransac
from line_scope_util import get_line,get_curve
from parse_args import parse_args
LEFT = 1
RIGHT = -1
MID = 0


class NewTrackLineGenerator:
    """This class is used for generate pixel points on the track line.
        How to use: An example:
            base_param = BaseParam(tread, wheelbase, head_height, front_wheel_to_head_d, param_yaml_path)
            track_line_generator = NewTrackLineGenerator(base_param)
            left_line,right_line = track_line_generator.add_track_line(0)
    Attributes:
        base_param: A BaseParam class contains all the necessary car and camera parameters.
        steer_angle: Steering angle of front wheel.
        dir: Go straight or turn left or turn left.
        line_color: (B,G,R) color, the line color of track lines on the frame when testing
        curve_point_color: (B,G,R) color, the points color of track lines on the frame when testing
        x_end: The furthest distance of the point on track in the real world.
    """
    def __init__(self, base_param):
        self.base_param = base_param
        self.steer_angle = 0
        self.dir = MID
        self.line_color = (0, 255, 0)
        self.curve_point_color = (0, 0, 255)
        self.x_end = 100

    def add_track_line(self, steer_angle, frame=None):
        """Used for getting the coordinates of each pixel on track lines

        Args:
            steer_angle: Current steering angle of front wheel.
            frame: Current video frame.

        Returns:
            curve_pixel_left: A list whose size is the height of the frame.
                The i-th value is the col value of the pixel of left line in the i-th row. If the i-th value is zero,
                it means that the line doesn't reach the i-th row.
            curve_pixel_right: A list whose size is the height of the frame.
                The i-th value is the col value of the pixel of right line in the i-th row. If the i-th value is zero,
                it means that the line doesn't reach the i-th row.
        """
        # Set scatter's xyz position on the line in real world
        x_start = self.base_param.head_to_back_wheel_d
        x_end = self.x_end
        y_range = self.base_param.tread / 2.0
        z_pos = self.base_param.head_height
        point_num = int(x_end) - int(x_start)
        # Returns num evenly spaced samples, calculated over the interval [x_start, x_end]
        line_world_y = np.linspace(x_start, x_end, point_num)
        line_world_left_x = np.ones(point_num)
        line_world_right_x = np.ones(point_num)
        curve_point_count_left = 0
        curve_point_count_right = 0

        # Calculate track line point's x in real world if the steer angle is non-zero
        # In this version, steer_angle is always 0
        self.steer_angle = self.steer_angle_rectify(steer_angle)
        if self.dir != MID:
            length = line_world_y.shape[0]
            left_end_mark = False
            right_end_mark = False
            for i in range(length):
                try:
                    if not left_end_mark:
                        line_world_left_x[i] = self.get_line_left_x_real_world(line_world_y[i])
                        curve_point_count_left += 1
                except ValueError:
                    # This exception will happen when line_world_y[i] exceeds the largest y on the trajectory.And the remain
                    # line_world_y is keep its' default value 1, and these points would be filtered in get_curve() by the
                    # boundary bottom_y.
                    # print("value error")
                    left_end_mark = True
                    pass
                try:
                    if not right_end_mark :
                        line_world_right_x[i] = self.get_line_right_x_real_world(line_world_y[i])
                        curve_point_count_right += 1
                except ValueError:
                    # This exception will happen when line_world_y[i] exceeds the largest y on the trajectory.And the remain
                    # line_world_y is keep its' default value 1, and these points would be filtered in get_curve() by the
                    # boundary bottom_y.
                    # print("value error")
                    right_end_mark = True
                    pass
        else:
            line_world_left_x = y_range * line_world_left_x
            line_world_right_x = (-y_range) * line_world_right_x

        # Transform left and right line real world coordinates to pixel coordinates on frame by transform matrix in
        # calibration which ignores camera distortion. [x,y,z,1]
        line_left = np.stack((line_world_y, line_world_left_x, z_pos * np.ones(point_num), np.ones(point_num)), 0)
        line_pixel_left = np.dot(self.base_param.tf_matrix, line_left)
        line_pixel_left[0] = np.divide(line_pixel_left[0], line_pixel_left[2])
        line_pixel_left[1] = np.divide(line_pixel_left[1], line_pixel_left[2])
        line_right = np.stack((line_world_y, line_world_right_x, z_pos * np.ones(point_num), np.ones(point_num)), 0)
        line_pixel_right = np.dot(self.base_param.tf_matrix, line_right)
        line_pixel_right[0] = np.divide(line_pixel_right[0], line_pixel_right[2])
        line_pixel_right[1] = np.divide(line_pixel_right[1], line_pixel_right[2])
        # Get the pixel coordinate of the end of line, which is closed to the bottom of the frame.
        line_bottom_y = int(line_pixel_left[1][0])
        if line_bottom_y > self.base_param.screen_h:
            line_bottom_y = self.base_param.screen_h - 2

        if self.dir == MID:
            # Using ransac algorithm to fit the line
            aL, bL = fit_line_by_ransac(line_pixel_left, sigma=3)
            line_pixel_left_y = np.arange(0, int(self.base_param.screen_h) - 1, 1)
            line_pixel_left_x = line_pixel_left_y * aL + bL
            # print(aL, bL)
            line_pixel_left = np.stack((line_pixel_left_x, line_pixel_left_y), 0)
            aR, bR = fit_line_by_ransac(line_pixel_right, sigma=3)
            line_pixel_right_y = np.arange(0, int(self.base_param.screen_h) - 1, 1)
            line_pixel_right_x = line_pixel_right_y * aR + bR
            # print(aR, bR)
            line_pixel_right = np.stack((line_pixel_right_x, line_pixel_right_y), 0)
            # Get the intersection point of two lines
            cross_p = cross_point(aL, bL, aR, bR)  #[colIndex, rowIndex]
            cross_p = [int(cross_p[1]), int(cross_p[0])]
            line_left_bottom_p = [int(line_pixel_left[0][line_bottom_y]), int(line_pixel_left[1][line_bottom_y])]
            line_right_bottom_p = [int(line_pixel_right[0][line_bottom_y]), int(line_pixel_right[1][line_bottom_y])]
            curve_pixel_left, curve_pixel_right = get_line(tuple(line_left_bottom_p),tuple(cross_p),
                                                           tuple(cross_p), tuple(line_right_bottom_p),
                                                           self.base_param.screen_h, self.base_param.screen_w )
        else:
            curve_pixel_left, curve_pixel_right = get_curve(line_pixel_left, line_pixel_right,
                                                            self.base_param.screen_h, self.base_param.screen_w ,
                                                            line_bottom_y, curve_point_count_left, curve_point_count_right)

        # This part is used for testing convenience
        if frame is not None:
            # for i in range(int(self.base_param.screen_h)-1):
            #     cv2.circle(frame,(int(line_pixel_left[0][i]), int(line_pixel_left[1][i])),radius=3, color=(0,0,255),thickness=-1)
            #     cv2.circle(frame,(int(line_pixel_right[0][i]), int(line_pixel_right[1][i])),radius=3, color=(0,0,255),thickness=-1)
            if self.dir == MID:
                cv2.line(frame, (int(cross_p[0]), int(cross_p[1])),
                         (line_left_bottom_p[0], line_left_bottom_p[1]), color=self.line_color, thickness=2)
                cv2.line(frame, (int(cross_p[0]), int(cross_p[1])),
                         (line_right_bottom_p[0], line_right_bottom_p[1]), color=self.line_color, thickness=2)
            else:
                # for i in range(len(line_pixel_right[0])):
                #     if i < curve_point_count_right :
                #         cv2.circle(frame, (int(line_pixel_right[0][i]), int(line_pixel_right[1][i])), radius=2, color=self.curve_point_color, thickness=-1)
                #     if i < curve_point_count_left :
                #         cv2.circle(frame,  (int(line_pixel_left[0][i]), int(line_pixel_left[1][i])), radius=2, color=self.curve_point_color , thickness=-1)
                for i in range(self.base_param.screen_h):
                    if i < line_bottom_y and curve_pixel_left[i] != 0:
                        cv2.circle(frame,(curve_pixel_left[i],i),radius= 1, color=self.line_color, thickness= -1)
                        cv2.circle(frame, (curve_pixel_right[i], i), radius=1, color=self.line_color, thickness=-1)
            cv2.imwrite('/Users/oumingfeng/Documents/lab/HW/world_to_image/test.jpg', frame)

        return curve_pixel_left, curve_pixel_right

    def get_line_left_x_real_world(self, y):
        """Used for getting real world x coordinate at vertical distance of the left line

        Args:
            y: Point's vertical distance in the real world, the same as x coordinate in vehicle coordinate system.

        Returns:
            x: float value, the point's horizontal distance in the real world, the same as y coordinate
            in vehicle coordinate system.
        """
        r2 = math.pow((self.base_param.wheelbase * self.cot(self.steer_angle) - self.dir * self.base_param.tread / 2),
                      2) \
             + math.pow((self.base_param.front_wheel_to_head_d + self.base_param.wheelbase), 2)
        return self.cal_x(r2, y)

    def get_line_right_x_real_world(self, y):
        """Used for getting real world x coordinate at vertical distance of the right line

          Args:
              y: Point's vertical distance in the real world, the same as x coordinate in vehicle coordinate system.

          Returns:
              x: float value, the point's horizontal distance in the real world, the same as y coordinate
              in vehicle coordinate system.
          """
        r2 = math.pow((self.base_param.wheelbase * self.cot(self.steer_angle) + self.dir * self.base_param.tread / 2),
                      2) \
             + math.pow((self.base_param.front_wheel_to_head_d + self.base_param.wheelbase), 2)
        return self.cal_x(r2, y)

    def cal_x(self, r2, y):
        first = r2 - math.pow(y, 2)
        # math.pow(wheelbase*cot(steer_angle)+tread/2) - math.pow(y+front_wheel_to_head_d+wheelbase)
        return (-self.dir) * math.pow(first, 0.5) + self.dir * self.base_param.wheelbase * self.cot(self.steer_angle)

    def steer_angle_rectify(self, steer_angle):
        """Rectify the steer angle because steer angle usually has a offset

        Args:
            steer_angle: Steer angle of font wheel

        Returns:
            Steer angle(radian)
        """
        steer_angle = steer_angle #+ 0.2
        if steer_angle < 0:
            self.dir = RIGHT
        else:
            self.dir = LEFT
        steer_angle = math.fabs(steer_angle)
        if steer_angle <= 0.01:
            # print(steer_angle)
            steer_angle = 0
            self.dir = MID
        return steer_angle

    def cot(self, x):
        return 1 / math.tan(x)



def cross_point(k1, b1, k2, b2):
    """Calculate intersection function
    Args:
        k1: Slope of straight line1
        b1: Line1 intercept
        k2: Slope of straight line2
        b2: Line2 intercept

    Returns:
        [x, y]: intersection point
    """
    x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def test():
    """
    This function just write for testing.

    Returns: None

    """
    # param test 1
    # screen_w = 1920.0
    # screen_h = 1080.0
    # camera_h = 1.2
    # tread = 1.6
    # wheelbase = 2.3
    # front_wheel_to_head_d = 1.2
    # video_path = '/Users/oumingfeng/Documents/lab/HW/data/1575338030730750.mp4'

    # para test 2 -- audi a6l
    # tread = 1.88
    # wheelbase = 3.02
    # front_wheel_to_head_d = 0.919
    # head_height = 0.676
    # param_yaml_path = "/Users/oumingfeng/Documents/lab/HW/data/1440Bonnet.yaml"
    # video_path = "/Users/oumingfeng/Documents/lab/HW/data/1440Bonnet.avi"

    # para test 3 -- audi a8l
    # tread = 1.95
    # wheelbase = 3.128
    # front_wheel_to_head_d = 0.989
    # head_height = 0.697
    # param_yaml_path = "/Users/oumingfeng/Documents/lab/HW/data/1920Bonnent.yaml"
    # video_path = "/Users/oumingfeng/Documents/lab/HW/data/1920Bonnent.avi"

    # para test 4 -- Crusie magotan 2
    # tread = 1.832
    # wheelbase = 2.871
    # front_wheel_to_head_d = 0.89
    # head_height = 0.68
    # param_yaml_path = "/Users/oumingfeng/Documents/lab/HW/data/1920NoBonnet.yaml"
    # video_path = "/Users/oumingfeng/Documents/lab/HW/data/1920NoBonnet.avi"

    # para test 5 -- Command line Crusie magotan 2
    # python track_line_generator.py  --tread 1.832 --wheelbase 2.871 --front_wheel_to_head_d 0.89 --head_height 0.68 --camera_yaml_path /Users/oumingfeng/Documents/lab/HW/data/1920NoBonnet.yaml --video_path /Users/oumingfeng/Documents/lab/HW/data/1920NoBonnet.avi
    args = parse_args()
    tread = args.tread
    wheelbase = args.wheelbase
    front_wheel_to_head_d = args.front_wheel_to_head_d
    head_height = args.head_height
    param_yaml_path = args.camera_yaml_path
    video_path = args.video_path

    base_param = BaseParam(tread, wheelbase, head_height, front_wheel_to_head_d, param_yaml_path)
    track_line_generator = NewTrackLineGenerator(base_param)

    vc = cv2.VideoCapture(video_path)
    rval = vc.isOpened()
    c = 0
    while rval:
        c = c + 1
        rval, frame = vc.read()
        if c == 100:
            result = track_line_generator.add_track_line(0.3, frame)
            #print(result)
            break
    vc.release()


if __name__ == "__main__":
    test()



