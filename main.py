import os
from PIL import Image, ImageStat
import math
import random
import shutil
import Augmentor

"""
Helper function definitions
"""


def perceived_brightness(im_file):
    """
    Get the perceived brightness
    :param im_file:
    :return:
    """
    try:
        im = Image.open(im_file)
        stat = ImageStat.Stat(im)
        r, g, b = stat.mean
        return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
    except Exception as e:
        return 128


"""
Main code
"""

original_path = './TrainVal'
output_path = './TrainVal_output'
validation_path = './TrainVal_validation'
if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(validation_path):
    os.mkdir(validation_path)

classes = os.listdir(original_path)
classes = [c for c in classes if c != '.DS_Store']
class_numbers = []
number_by_class = {}
for c in classes:
    class_number = len(os.listdir(os.path.join(original_path, c)))
    class_numbers.append((int(c), class_number))
    number_by_class[c] = class_number
sorted_classes = sorted(class_numbers, key=lambda x: x[1])
min_class_label, min_class_num = sorted_classes[0]
max_class_label, max_class_num = sorted_classes[-1]

test_target_class_num = 1
target_class_num = 1200  # TODO: change this to 10k later

# take out random images of train set to be the validation set
validation_set_size_per_class = 7
for c in classes[:test_target_class_num]:
    files = [f for f in os.listdir(os.path.join(original_path, c)) if f != '.DS_Store']
    random.shuffle(files)
    validation_files = files[:7]
    validation_class_path = os.path.join(validation_path, c)
    if not os.path.exists(validation_class_path):
        os.mkdir(validation_class_path)
    for vf in validation_files:
        src_file = os.path.join(original_path, c, vf)
        target_file = os.path.join(validation_class_path, vf)
        # TODO: change to shutil.move later
        shutil.copy(src_file, target_file)

for c in classes[:test_target_class_num]:
    print('processing class {}'.format(c))
    class_num = number_by_class[c]
    # number of images needs to be generated
    target_class_num = target_class_num - class_num - validation_set_size_per_class
    print("target output file = {}".format(target_class_num))
    class_path = os.path.join(original_path, c)
    class_output_path = os.path.join('../../', output_path, c)
    p = Augmentor.Pipeline(class_path, output_directory=class_output_path)
    p.skew_tilt(probability=0.5, magnitude=0.2)
    p.random_brightness(probability=0.5, min_factor=0.2, max_factor=0.8)
    p.random_erasing(probability=0.5, rectangle_area=0.15)
    p.random_distortion(probability=0.5, grid_width=300, grid_height=300, magnitude=4)
    p.sample(target_class_num)
