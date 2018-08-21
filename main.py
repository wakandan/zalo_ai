import os
from PIL import Image, ImageStat
import math
import random
import shutil
import Augmentor
import imgaug
from Augmentor.Operations import Operation
from imgaug import augmenters as iaa
import numpy as np

"""
Helper function definitions
"""

TEST_MODE = True


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


class GaussianBlurOperation(Operation):
    def __init__(self, probability, magnitude=3.0):
        self.probability = probability
        self.blurer = iaa.GaussianBlur(magnitude)

    def perform_operation(self, images):
        return_images = []
        for image in images:
            image_array = np.array(image).astype('uint8')
            if random.random() < self.probability:
                processed_image = image_array
            else:
                self.blurer.augment_image(image_array)
            processed_image = Image.fromarray(image_array)
            return_images.append(processed_image)
        return return_images


"""
Main code
"""

original_path = os.path.abspath('./TrainVal')
output_path = os.path.abspath('./TrainVal_output')
validation_path = os.path.abspath('./TrainVal_validation')
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

if TEST_MODE:
    target_class_num = 3000  # TODO: change this to 10k later
    test_target_class_num = 2
else:
    target_class_num = 10000
    test_target_class_num = len(classes)

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
        if TEST_MODE:
            shutil.copy(src_file, target_file)
        else:
            shutil.move(src_file, target_file)

for c in classes[:test_target_class_num]:
    print('processing class {}'.format(c))
    class_num = number_by_class[c]
    # number of images needs to be generated
    target_class_num = target_class_num - class_num - validation_set_size_per_class
    print("target output file = {}".format(target_class_num))
    class_path = os.path.join(original_path, c)
    print("class path = {}".format(class_path))
    class_output_path = os.path.join(output_path, c)
    print("class path output = {}".format(class_output_path))
    p = Augmentor.Pipeline(class_path, output_directory=class_output_path)
    p.skew_tilt(probability=0.5, magnitude=0.2)
    # change a bit here 
    p.random_brightness(probability=1, min_factor=0.5, max_factor=1.5)
    
    # may be add multiple box
    p.random_erasing(probability=0.5, rectangle_area=0.15)
    p.add_operation(GaussianBlurOperation(probability=0.5, magnitude=4))
    p.sample(target_class_num)
