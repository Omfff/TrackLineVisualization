import argparse


def parse_args():
    """ Parse args from command line

    Returns:
        args: all required params from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tread", help="car width (meters)",
                        metavar='\b', type=float, required=True)
    parser.add_argument("-w", "--wheelbase", help="distance between the front and rear axles of a vehicle",
                        metavar='\b', type=float, required=True)
    parser.add_argument("-f", "--front_wheel_to_head_d", help="distance between front wheel center and car head",
                        metavar='\b', type=float, required=True)
    parser.add_argument("-e", "--head_height", help="z position of the point in the car head",
                        metavar='\b', type=float, required=True)
    parser.add_argument("-c", "--camera_yaml_path", help="yaml file path for camera configuration",
                        metavar='\b', type=str, required=True)
    parser.add_argument("-v", "--video_path", help="video path",
                        metavar='\b', type=str, required=True)
    args = parser.parse_args()
    return args