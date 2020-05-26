from yaml_reader import CameraParam
import cv2
import pandas as pd
import numpy as np


def draw_objects_per_frame(frame, obj_pixel_pos, obj_info):
    """ Draw all radar objects for one frame

    Args:
        frame: Frame image array
        obj_pixel_pos: Pixel coordinates for radar objects, array likes [[x,y] ,..., [x,y]].
        obj_info: The information of radar objects, including [id, x, y, z]

    Returns:
        Frame with radar objects information text.
    """
    width = frame.shape[1]
    height = frame.shape[0]
    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    font_color = (0, 0, 255)
    font_thickness = 2
    point_color = (0, 255, 255)
    poinr_radius = 5
    for index in range(len(obj_info)):
        if obj_pixel_pos[index][0] > width or obj_pixel_pos[index][0] <0 or \
                obj_pixel_pos[index][1] > height or obj_pixel_pos[index][1] <0:
            continue
        info_id = ' id:' + str(int(obj_info[index][0]))
        info_XY = 'X:' + ('%.2f' % obj_info[index][1]) + ' Y:' + ('%.2f' % obj_info[index][2])

        #print(info,infoXY int(obj_pixel_pos[index][0]), int(obj_pixel_pos[index][1]), width, height)
        text_id_size = cv2.getTextSize(info_id, font_face, font_scale, font_thickness)
        text_id_x = int(obj_pixel_pos[index][0])
        text_id_y = int(obj_pixel_pos[index][1] + text_id_size[0][1] / 2)
        text_XY_size = cv2.getTextSize(info_XY, font_face, font_scale, font_thickness)
        text_XY_x = int(obj_pixel_pos[index][0] - text_XY_size[0][0]/2)
        text_XY_y = int(obj_pixel_pos[index][1] + text_id_size[0][1] + text_XY_size[0][1]/2) + 5

        cv2.circle(frame, (int(obj_pixel_pos[index][0]), int(obj_pixel_pos[index][1])), radius=poinr_radius,
                   color=point_color,thickness=font_thickness)
        frame = cv2.putText(frame, info_id, (text_id_x, text_id_y),fontFace = font_face,
                            fontScale =font_scale, color= font_color, thickness = font_thickness)
        frame = cv2.putText(frame, info_XY, (text_XY_x, text_XY_y), fontFace=font_face,
                            fontScale=font_scale, color=font_color, thickness=font_thickness)
    return frame


def draw_radar_objects_on_video(video_path, radar_object_path, yaml_path, save_path):
    """Draw all radar objects info on videos.

    Args:
        video_path: Video file path.
        radar_object_path: Radar objects csv path.
        yaml_path: Camera yaml configure file path.
        save_path: Outpue video save path

    """
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    radar_objects = pd.read_csv(radar_object_path)
    camera_con = CameraParam(yaml_path)
    output_video = cv2.VideoWriter(save_path, fourcc, fps, size)

    rval = vc.isOpened()
    frame_index = 0
    second = 1
    objects_second = radar_objects[radar_objects['sec'] == second]
    while rval:
        frame_index = frame_index + 1
        rval, frame = vc.read()
        objects_frame = objects_second[objects_second['fps'] == frame_index]
        if objects_frame is None or len(objects_frame)==0:
            continue
        obj_vec_pos = objects_frame[['obj_x','obj_y','obj_z']].values
        obj_vec_pos = np.vstack((obj_vec_pos.T, np.ones(obj_vec_pos.shape[0])))
        obj_pixel_pos = np.dot(camera_con.transform_veh2image_matrix, obj_vec_pos)
        obj_pixel_pos[0] = np.divide(obj_pixel_pos[0], obj_pixel_pos[2])
        obj_pixel_pos[1] = np.divide(obj_pixel_pos[1], obj_pixel_pos[2])
        # [[x,y]...[x,y]]
        obj_pixel_pos = np.delete(obj_pixel_pos.T, 2, axis=1)
        frame = draw_objects_per_frame(frame, obj_pixel_pos, objects_frame[['obj_id','obj_x','obj_y','obj_z']].values)
        output_video.write(frame)
        if frame_index % fps == 0:
            second+=1
            frame_index = 0
            objects_second = radar_objects[radar_objects['sec'] == second]

    vc.release()
    output_video.release()


if __name__ == '__main__':
    basic_path = '/Users/oumingfeng/Documents/lab/HW/data/'
    video_path = basic_path +'1561343763996000.mp4'
    radar_object_path = basic_path + 'fusion_5seconds.csv'
    yaml_path = basic_path + 'roof_cam_2.yaml'
    save_path = basic_path + 'test.mp4'
    draw_radar_objects_on_video(video_path, radar_object_path, yaml_path, save_path)
