import cv2
import glob
import numpy as np


label = []  # stores the label of each image
image_data = []  # store the image data


football_filenames = [img for img in glob.glob("images/football/*.jpg")]
tennis_ball_filenames = [img for img in glob.glob("images/tennisball/*.jpg")]
football_pitch_filenames = [img for img in glob.glob("images/footballpitch/*.jpg")]
tennis_court_filenames = [img for img in glob.glob("images/tenniscourt/*.jpg")]

football_filenames.sort()
tennis_ball_filenames.sort()
football_pitch_filenames.sort()
tennis_court_filenames.sort()


def translate_image(img, label, distance=0, step=20):
    # Translation of image data to create more training data
    # img : image data
    # label : label data for the image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # return : list of new images and corresponding label data

    rows, cols = img.shape[:2]
    img_list = []
    label_list = []

    # add original un touched image
    img_list.append(img)
    label_list.append(label)

    if(step <= 0):
        step = 1

    # add step to include the distance in the image transform
    for offset in range(step, distance+step, step):
        M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
        M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

        # shift the image to the right and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_right, (cols, rows)))
        # shift the image to the left and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_left, (cols, rows)))

        label_list.append(label)  # for image shifted to the right
        label_list.append(label)  # left

    return img_list, label_list


# Zooms an image in
# img : image data
# label : label data for the image
# ratio : ratio to zoom image. ratio > 1 will zoom in & ratio < 1 zoom out
# step : steps between zoom ratio images
# return : list of new images and corresponding label data
def zoom_image(img, label, ratio=2, steps=2):

    rows, cols = img.shape[:2]
    img_list = []
    label_list = []

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
        img_list.append(cv2.resize(larger_img, (rows, cols), interpolation=cv2.INTER_CUBIC))

        label_list.append(label)

    return img_list, label_list

## Original function sourced from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def more_images(img, label_name):
    shif_dist = 0
    zoom_ratio = 1.5

    # translate base image
    img_list, label_list = translate_image(img, label_name, step=shif_dist // 2,
                                           distance=shif_dist)
    image_data.extend(img_list)
    label.extend(label_list)

    # scale the flipped image
    img_list, label_list = zoom_image(img, label_name, ratio=zoom_ratio, steps=2)
    image_data.extend(img_list)
    label.extend(label_list)

def make_data(filenames, label_name, img_size=256):

    for name in filenames:
        image = cv2.imread(name)

        if image == None:  # skip corrupt image data
            continue

        # adjust image gamma for more data
        for i in range(0, 3):

            if i <= 1: # 0.5 and 1.0
                gamma = 0.5 + i * 0.5
            else:
                gamma = 1 + (i - 1) * 1. # 2.0

            img = adjust_gamma(image, gamma)

            # resize to the GoogLenet input size of 256x256x3
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

            # translate base image
            more_images(img, label_name)

            # translate horizontally flipped image
            flip_img = cv2.flip(img, 1)
            more_images(flip_img, label_name)

            # translate vertically flipped image
            flip_img = cv2.flip(img, 0)
            more_images(flip_img, label_name)

            # translate vertically and horizontally flipped image
            flip_img = cv2.flip(img, -1)
            more_images(flip_img, label_name)


if __name__ == "__main__":
    make_data(football_filenames, "football")
    make_data(tennis_ball_filenames, "tennis_ball")
    make_data(football_pitch_filenames, "nothing")
    make_data(tennis_court_filenames, "nothing")

    print(len(image_data))


