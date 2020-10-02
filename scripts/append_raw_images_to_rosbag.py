import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from os.path import join
import rospy
import argparse
import shutil
import os
import glob


def read_bayer12p(data):
    fst_uint8, mid_uint8, lst_uint8 = data.reshape(-1, 3).astype(np.uint16).T
    fst_uint12 = fst_uint8  + ((mid_uint8 % 16) << 8)
    snd_uint12 = (lst_uint8 << 4) + (mid_uint8 >> 4)
    stack = np.stack([fst_uint12, snd_uint12],-1)
    return stack.reshape((-1,))

def get_demosaiced_image(filepath):
    rows = 1080
    cols = 1440
    with open(filepath, 'rb') as raw_file:
        raw_data = np.fromfile(raw_file, dtype="uint8", count=rows*cols*12//8)
    img_raw = read_bayer12p(raw_data)
    img_raw = np.reshape(img_raw, (rows, cols))
    img = cv2.cvtColor(img_raw, cv2.COLOR_BAYER_RG2RGB_EA)
    return img

def load_img(path):
    if path.endswith(".raw"):
        return get_demosaiced_image(path), True
    elif path.endswith(".png"):
        return cv2.imread(path), False
    else:
        raise ValueError


def is_img(path):
    return path.endswith(".raw") or path.endswith(".png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", default="/home/dani/Documents/projects/dual_setup_calibration/data/calibration_data/200831/log.bag", type=str,
                        help="Delimited list of datasets")
    parser.add_argument("--image_folder", default="/home/dani/Documents/projects/dual_setup_calibration/data/calibration_data/200831/images",
                        type=str, help="Path to the base folder containing the rosbags")
    parser.add_argument("--trigger_topic",
                        type=str, default="/dvs/image_raw", help="What timestamps should be used for the added images")
    parser.add_argument("--image_topic", default='/dvs/image_raw',
                        type=str, help="Path to the output folder")

    args = parser.parse_args()

    bridge = CvBridge()

    # Now append the reconstructed images to the copied rosbag
    # only keep timestamps with positive polarity
    print("Loading timestamps")
    with rosbag.Bag(args.rosbag, "r") as bag:
        timestamps = [t for topic, msg, t in bag.read_messages(topics=[args.trigger_topic]) if msg.polarity]


    image_files = sorted([f for f in glob.glob(join(args.image_folder, "*")) if is_img(f)])
    print("Appending %s images with %s triggers" % (len(image_files), len(timestamps)))

    assert len(image_files) > 0
    assert len(timestamps) > 0

    # Copy the source bag to the output folder
    print("Copying bag...")
    temp_bag_name = args.rosbag + ".appended"
    shutil.copyfile(args.rosbag, temp_bag_name)

    # list all images in the folder
    with rosbag.Bag(temp_bag_name, 'a') as outbag:
        for i, (t, image_path) in enumerate(zip(timestamps, image_files)):

            img, is_hdr = load_img(image_path)

            try:
                if is_hdr:
                    # save 16 bit image
                    img_msg = bridge.cv2_to_imgmsg(img, encoding='passthrough')
                    img_msg.header.stamp = t
                    img_msg.header.seq = i
                    outbag.write(args.image_topic+"_16bit", img_msg, img_msg.header.stamp)
                    # also save 8 bit image
                    img = (img.astype("float64") / (2**12-1) * (2**8-1)).astype("uint8")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_msg = bridge.cv2_to_imgmsg(img, encoding='mono8')
                img_msg.header.stamp = t
                img_msg.header.seq = i
                outbag.write(args.image_topic, img_msg, img_msg.header.stamp)

            except CvBridgeError, e:
                print e

            print("Progress [%5d/%5d]" % (i,len(timestamps)))