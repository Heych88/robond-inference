import cv2
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os, errno
from termcolor import colored

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

    # check if the directory path exists otherwise create it
    if not os.path.exists('Data/'):
        try:
            os.makedirs('Data/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Oops!  Can not make the 'Data/' directory. Given error {}".format(e.errno))
                print(e.args)
                raise
    elif not os.path.exists('Data/' + set_dir):
        try:
            os.makedirs('Data/' + set_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Oops!  Can not make the 'Data/{}' directory. Given error {}".format(set_dir, e.errno))
                print(e.args)
                raise

    filename = 'Data/' + set_dir + '/' + name_type + "_" + str(number) + ".png"
    cv2.imwrite(filename, img)

    #image_data.append(filename + "<" + str(label_type) + ">")

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
            #flip_img = cv2.flip(img, 0)
            #more_images(flip_img, set_dir, name_type, label_type)

            # translate vertically and horizontally flipped image
            #flip_img = cv2.flip(img, -1)
            #more_images(flip_img, set_dir, name_type, label_type)



def load_filenames(directory):
    if not os.path.exists(directory):
        print(colored("Directory {} Doesn't exist".format(directory), 'yellow'))
        return None
    else:
        return [img for img in glob.glob(directory + "*.jpg")]


def get_target_data(label_keys):
    ######################### Make target data ################################
    # Give image a name type
    name_type = 'Football'
    # Specify the name of the directory that has been premade and be sure that it's the name of your class
    # Remember this directory name serves as your data label for that particular class
    set_dir = 'Football'
    number = 0  # current image number to be saved
    print("Making data for directory {}".format(set_dir))
    football_filenames = load_filenames("images/football/")
    if football_filenames is not None:
        make_data(football_filenames, set_dir, name_type, label_keys['Football'])

    set_dir = 'Tennis_ball'
    name_type = 'Tennis_ball'
    number = 0
    print("Making data for directory {}".format(set_dir))
    tennis_ball_filenames = load_filenames("images/tennisball/")
    if tennis_ball_filenames is not None:
        make_data(tennis_ball_filenames, set_dir, name_type, label_keys['Tennis_ball'])

    set_dir = 'Basketball'
    name_type = 'Basketball'
    number = 0
    print("Making data for directory {}".format(set_dir))
    basketball_filenames = load_filenames("images/basketball/")
    if basketball_filenames is not None:
        make_data(basketball_filenames, set_dir, name_type, label_keys['Basketball'])

    set_dir = 'American_football'
    name_type = 'American_football'
    number = 0
    print("Making data for directory {}".format(set_dir))
    american_football_filenames = load_filenames("images/american_football/")
    if american_football_filenames is not None:
        make_data(american_football_filenames, set_dir, name_type, label_keys['American_football'])

    set_dir = 'Rugby_ball'
    name_type = 'Rugby_ball'
    number = 0
    print("Making data for directory {}".format(set_dir))
    rugby_ball_filenames = load_filenames("images/rugby_ball/")
    if rugby_ball_filenames is not None:
        make_data(rugby_ball_filenames, set_dir, name_type, label_keys['Rugby_ball'])

    set_dir = 'Volleyball'
    name_type = 'Volleyball'
    number = 0
    print("Making data for directory {}".format(set_dir))
    volleyball_filenames = load_filenames("images/volleyball/")
    if volleyball_filenames is not None:
        make_data(volleyball_filenames, set_dir, name_type, label_keys['Volleyball'])


def get_nothing_data(label_keys):
    ##################### Make background noise data ##########################
    set_dir = 'Nothing'
    name_type = 'Nothing'
    number = 0
    print("Making data for directory {}".format(set_dir))

    football_pitch_filenames = load_filenames("images/footballpitch/")
    if football_pitch_filenames is not None:
        make_data(football_pitch_filenames, set_dir, name_type, label_keys['Nothing'])

    tennis_court_filenames = load_filenames("images/tenniscourt/")
    if tennis_court_filenames is not None:
        make_data(tennis_court_filenames, set_dir, name_type, label_keys['Nothing'])

    basketball_court_filenames = load_filenames("images/basketball_court/")
    if basketball_court_filenames is not None:
        make_data(basketball_court_filenames, set_dir, name_type, label_keys['Nothing'])

    nothing_american_football_filenames = load_filenames("images/nothing_american_football/")
    if nothing_american_football_filenames is not None:
        make_data(nothing_american_football_filenames, set_dir, name_type, label_keys['Nothing'])

    nothing_volleyball_filenames = load_filenames("images/nothing_volleyball/")
    if nothing_volleyball_filenames is not None:
        make_data(nothing_volleyball_filenames, set_dir, name_type, label_keys['Nothing'])

    rugby_field_filenames = load_filenames("images/rugby_field/")
    if rugby_field_filenames is not None:
        make_data(rugby_field_filenames, set_dir, name_type, label_keys['Nothing'])


if __name__ == "__main__":
    global number

    # numerical label value
    label_keys = {'Football': 0,
                  'Tennis_ball': 1,
                  'Basketball': 2,
                  'American_football': 3,
                  'Rugby_ball': 4,
                  'Volleyball': 5,
                  'Nothing': 6}

    get_target_data(label_keys)

    get_nothing_data(label_keys)

    print("Finished making data")

    #print("Total data {}".format(len(image_data)))


    #shuffle the data for train validation
    '''shuf_img = shuffle(image_data)

    train_data, val_data = train_test_split(shuf_img, test_size=0.15)

    print("Train data {}, Validation data {}".format(len(train_data), len(val_data)))

    print("saving data file")
    save_address_file('train_file.txt', train_data)
    save_address_file('validation_file.txt', val_data)
    print("Finished")'''
