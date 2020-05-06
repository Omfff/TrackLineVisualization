import yaml
import numpy as np


class BaseParam:
    """A  class contains all the necessary car and camera parameters.

    Attributes:
        cam_param: A BaseParam class contains all the necessary car and camera parameters.
        screen_w: Steering angle of front wheel.
        screen_h: Go straight or turn left or turn left.
        camera_h: Camera z position in vehicle coordinate.(meters)
        tread: Car width (meters).
        wheelbase: The distance between the front and rear axles of a vehicle
        head_height: The z position of the point in the car head. And the diameter of the wheel is temporarily used
        front_wheel_to_head_d: Distance between front wheel center and car head
        param_yaml_path: Yaml file path for camera configuration.
        tf_matrix: Transform matrix that could convert real world coordinates to image plane's pixel coordinates.
    """
    def __init__(self, tread,wheelbase,head_height,front_wheel_to_head_d,param_yaml_path):
        self.alpha = 0
        self.beta = 0
        self.cam_param = CameraParam(param_yaml_path)
        self.screen_w = self.cam_param.resolution['width']
        self.screen_h = self.cam_param.resolution['height']
        self.camera_h = self.cam_param.cam_cord['z']
        self.tread = tread
        self.wheelbase = wheelbase
        self.front_wheel_to_head_d = front_wheel_to_head_d
        self.head_height = head_height #+ 0.05
        self.head_to_back_wheel_d = self.wheelbase + self.front_wheel_to_head_d
        self.tf_matrix = self.cam_param.transform_veh2image_matrix
        self.camera_to_head_d = front_wheel_to_head_d


class CameraParam:
    def __init__(self, param_yaml_path):
        with open(param_yaml_path) as f:
            para_dic = yaml.load(f, Loader=yaml.FullLoader)
            para_dic = para_dic['roof_cam_2']
            self.intrinsics = convert_to_intrinsics_matrix(para_dic['intrinsics']['data'])
            self.resolution = para_dic['resolution']
            self.cam_cord = para_dic['translation_veh_cam']
            self.distortion_coeffs = np.array(para_dic['distortion_coeffs']['data'])
            rotation_matrix = np.array(para_dic['rotation_veh2cam_matrix']['data'])
            self.rotation_matrix = np.reshape(rotation_matrix, (3, 3))
            translation_vec = np.array(para_dic['tanslation_veh2cam_matrix']['data'])
            self.translation_vec = np.reshape(translation_vec, (3, 1))
            transform_veh2image_matrix = np.array(para_dic['transform_veh2image_matrix']['data'])
            self.transform_veh2image_matrix = np.reshape(transform_veh2image_matrix, (3, 4))

    def get_tf_matrix(self):
        """
        Calculate transform matrix that could convert real world coordinates to image plane's pixel coordinates( without
        camera distortion.

        Returns:
            transform matrix: ndarray, whose shape is 3x4
        """
        return np.dot(self.intrinsics, np.hstack((self.rotation_matrix, self.translation_vec)))

def convert_to_intrinsics_matrix(intrinsics):
    # intrinsics 1x4
    intrinsics_matrix = np.array([[intrinsics[0], 0, intrinsics[2]],[0, intrinsics[1], intrinsics[3]],[0, 0, 1]])
    return intrinsics_matrix

if __name__ == "__main__":
    file_path = "/Users/oumingfeng/Documents/lab/HW/data/1440Bonnet.yaml"