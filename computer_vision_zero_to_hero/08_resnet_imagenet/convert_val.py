import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="dir with the images", required=True)
parser.add_argument("-l", "--labels", help="file with image name to class label mapping", required=True)

args = parser.parse_args()

processed_classes = set()

with open(args.labels, "r") as file:
    # skip header
    next(file)
    for line in file:
        img_name, labels = line.split(",")
        class_name = labels.split(" ")[0]
        # create a dir for this classname
        if class_name not in processed_classes:
            dir_path = args.dir + "/" + class_name
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        shutil.mv(args.dir + "/" + img_name + ".JPEG", args.dir + "/" + class_name+ "/" + img_name + ".JPEG")