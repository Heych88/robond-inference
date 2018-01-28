import cv2
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#label_data = []  # stores the label of each image
image_data = []  # store the image data
#label_data.extend(0)


football_filenames = [img for img in glob.glob("images/football/*.jpg")]
tennis_ball_filenames = [img for img in glob.glob("images/tennisball/*.jpg")]
football_pitch_filenames = [img for img in glob.glob("images/footballpitch/*.jpg")]
tennis_court_filenames = [img for img in glob.glob("images/tenniscourt/*.jpg")]

football_filenames.sort()
tennis_ball_filenames.sort()
football_pitch_filenames.sort()
tennis_court_filenames.sort()

# Initialize photo count
global number
number = 0


def save_address_file(file_name, list):

    file = open(file_name, 'w')

    for item, name in enumerate(list):
        file.write(name + '\n')

    file.close()


def save_image(img, set_dir, name_type, label_type):
    global number

    filename = 'Data/' + set_dir + '/' + name_type + "_" + str(number) + ".png"
    cv2.imwrite(filename, img)

    image_data.append(filename + "<" + str(label_type) + ">")

    number += 1


# Translates image data to create shifted training data
# img : image data
# label_type : label data for the image
# distance : max distance of transformed images
# step : steps between transformed images
# return : list of new images and corresponding label data
def translate_image(img, set_dir, name_type, label_type, distance=0, step=20):

    rows, cols = img.shape[:2]

    # add original un touched image
    save_image(img, set_dir, name_type, label_type)

    if(step <= 0):
        step = 1

    # add step to include the distance in the image transform
    for offset in range(step, distance+step, step):
        M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
        M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

        # shift the image to the right and append the process image to the list
        save_image(cv2.warpAffine(img, M_right, (cols, rows)), set_dir, name_type, label_type)
        # shift the image to the left and append the process image to the list
        save_image(cv2.warpAffine(img, M_left, (cols, rows)), set_dir, name_type, label_type)


# Zooms an image in
# img : image data
# label_type : label data for the image
# ratio : ratio to zoom image. ratio > 1 will zoom in & ratio < 1 zoom out
# step : steps between zoom ratio images
# return : list of new images and corresponding label data
def zoom_image(img, set_dir, name_type, label_type, ratio=2, steps=2):

    rows, cols = img.shape[:2]

    # prevent negative and zero values from the input
    ratio = max(ratio, 1)
    steps = max(steps, 1)

    # zoom into the image by finding the middle image size that is the ratio of the image
    # width, divide the excess of this value by two to find the start and end position
    # for the image regions.
    row_max = int((rows * (1 - (1 / ratio))) // 2)
    col_max = int((cols * (1 - (1 / ratio))) // 2)

    row_step = int(row_max // steps) # prevent divide by zero
    col_step = int(col_max // steps)

    col_offset = 0

    # add step to include the distance in the image transform
    for row_offset in range(row_step, row_max+1, row_step):
        col_offset += col_step

        # make the image larger
        larger_img = img[row_offset:rows - row_offset, col_offset:cols - col_offset]
        save_image(cv2.resize(larger_img, (rows, cols), interpolation=cv2.INTER_CUBIC), set_dir, name_type, label_type)


## Original function sourced from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# adjusts the gamma brightness of an image
# image : original image to be adjusted
# gamma : value to ajust the image gamma by
# return : gamma corrected image
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# calls image manipulation function to create multiple images of the one supplied image
# img : image data
# label_type : label data for the image
# return : list of new images and corresponding label data
def more_images(img, set_dir, name_type, label_type):
    shif_dist = 0
    zoom_ratio = 1.5

    # translate base image
    translate_image(img, set_dir, name_type, label_type, step=shif_dist // 2, distance=shif_dist)

    # scale the flipped image
    zoom_image(img, set_dir, name_type, label_type, ratio=zoom_ratio, steps=2)


# loads all the images in the file directory and creates variations of the image
# for extra image training data for neural networks.
# img : image data
# label : label data for the image
# return : list of new images and corresponding label data
def make_data(filenames, set_dir, name_type, label_type, img_size=256):

    for name in filenames:
        image = cv2.imread(name)

        if image == None:  # skip corrupt image data
            continue

        # adjust image gamma for more data
        for i in range(1, 2):

            if i <= 1: # 0.5 and 1.0
                gamma = 0.5 + i * 0.5
            else:
                gamma = 1 + (i - 1) * 1. # 2.0

            img = adjust_gamma(image, gamma)

            # resize to the GoogLenet input size of 256x256x3
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

            # translate base image
            more_images(img, set_dir, name_type, label_type)

            # translate horizontally flipped image
            flip_img = cv2.flip(img, 1)
            more_images(flip_img, set_dir, name_type, label_type)

            # translate vertically flipped image
            flip_img = cv2.flip(img, 0)
            #more_images(flip_img, set_dir, name_type, label_type)

            # translate vertically and horizontally flipped image
            flip_img = cv2.flip(img, -1)
            #more_images(flip_img, set_dir, name_type, label_type)


if __name__ == "__main__":
    global number
    # Give image a name type
    name_type = 'Football'

    # Specify the name of the directory that has been premade and be sure that it's the name of your class
    # Remember this directory name serves as your data label for that particular class
    set_dir = 'Football'

    # numerical label value
    label_keys = {'Football':0, 'Tennisball':1, 'Nothing':2}

    number = 0  # current image number to be saved

    print("Making data for directory {}".format(set_dir))
    make_data(football_filenames, set_dir, name_type, label_keys['Football'])

    set_dir = 'Tennisball'
    name_type = 'Tennisball'
    number = 0
    print("Making data for directory {}".format(set_dir))
    make_data(tennis_ball_filenames, set_dir, name_type, label_keys['Tennisball'])

    set_dir = 'None'
    name_type = 'Nothing'
    number = 0
    print("Making data for directory {}".format(set_dir))
    make_data(football_pitch_filenames, set_dir, name_type, label_keys['Nothing'])
    make_data(tennis_court_filenames, set_dir, name_type, label_keys['Nothing'])

    print("Finished making data")
    print("Total data {}".format(len(image_data)))


    #shuffle the data for train validation
    shuf_img = shuffle(image_data)

    train_data, val_data = train_test_split(shuf_img, test_size=0.15)

    print("Train data {}, Validation data {}".format(len(train_data), len(val_data)))

    print("saving data file")
    save_address_file('train_file.txt', train_data)
    save_address_file('validation_file.txt', val_data)
    print("Finished")
