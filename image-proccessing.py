import cv2
import glob
import numpy as np


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


# Zooms an image in or out
# img : image data
# label : label data for the image
# ratio : ratio to zoom image. ratio > 1 will zoom in & ratio < 1 zoom out
# step : steps between transformed images
# return : list of new images and corresponding label data
def zoom_image(img, label, ratio=2, steps=2):

    rows, cols = img.shape[:2]
    img_list = []
    label_list = []

    # zoom into the image by finding the middle image size that is the ratio of the image
    # width, divide the excess of this value by two to find the start and end position
    # for the image regions.
    row_max = int((rows * (1 - (1 / ratio))) // 2)
    col_max = int((cols * (1 - (1 / ratio))) // 2)

    row_step = int(row_max // max(steps, 1)) # prevent divide by zero
    col_step = int(col_max // max(steps, 1))

    col_offset = 0

    #print("row_max ", row_max, "   row_step ", row_step)

    # add step to include the distance in the image transform
    for row_offset in range(row_step, row_max+1, row_step):
        col_offset += col_step

        # make the image larger
        larger_img = img[row_offset:img_size - row_offset, col_offset:img_size - col_offset]
        print(larger_img.shape)
        larger_img = cv2.resize(larger_img, (rows, cols), interpolation=cv2.INTER_CUBIC)

        cv2.imshow("norm", img)
        cv2.imshow("larger_img", larger_img)

        cv2.waitKey(0)



football_filenames = [img for img in glob.glob("images/football/*.jpg")]
tennis_ball_filenames = [img for img in glob.glob("images/tennisball/*.jpg")]
football_pitch_filenames = [img for img in glob.glob("images/footballpitch/*.jpg")]
tennis_court_filenames = [img for img in glob.glob("images/tenniscourt/*.jpg")]

football_filenames.sort()
tennis_ball_filenames.sort()
football_pitch_filenames.sort()
tennis_court_filenames.sort()

label = []  # stores the label of each image
image_data = []  # store the image data


shif_dist = 0
img_size = 256

for name in football_filenames:
    img = cv2.imread(name)
    name = "football"

    # resize to the GoogLenet input size of 256x256x3
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # translate base image
    img_list, label_list = translate_image(img, name, step=shif_dist//2,
                                           distance=shif_dist)
    image_data.extend(img_list)
    label.extend(label_list)

    # translate horizontally flipped image
    img_list, label_list = translate_image(cv2.flip(img, 1), name, step=shif_dist // 2,
                                           distance=shif_dist)
    image_data.extend(img_list)
    label.extend(label_list)

    # translate vertically flipped image
    img_list, label_list = translate_image(cv2.flip(img, 0), name, step=shif_dist // 2,
                                           distance=shif_dist)
    image_data.extend(img_list)
    label.extend(label_list)

    # translate vertically and horizontally flipped image
    img_list, label_list = translate_image(cv2.flip(img, -1), name, step=shif_dist // 2,
                                           distance=shif_dist)
    image_data.extend(img_list)
    label.extend(label_list)

    zoom_image(img, name, ratio=1.5, steps=2)



print(len(image_data))






'''# Create normalised images for the training data
print('Creating the data')
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        angle = float(line[3])

        # only keep data with a steer angle
        if angle >= 0.01 or angle <= -0.01:
            for camera in range(0, 3):
                name = './data/IMG/' + line[camera].split('/')[-1]
                image = cv2.imread(name)

                # Left of image is a negative angle, Right positive
                if camera == 1:
                    # Left camera, make angle smaller to help return to center
                    angle += 0.3
                    distance = 0
                elif camera == 2:
                    # Right camera, make angle smaller to help return to center
                    angle -= 0.3
                    distance = 0
                else:
                    distance = 0

                img_list, angle_list = translate_image(image, angle,  step=20,
                                                       distance=distance)
                drive_data.extend(img_list)
                steer_data.extend(angle_list)

                img_list, angle_list = translate_image(np.fliplr(image), -angle,
                                                       distance=distance)
                drive_data.extend(img_list)
                steer_data.extend(angle_list)

print('data count: ', np.shape(drive_data))
train_data, val_data, train_labels, val_labels = train_test_split(drive_data,
                                                                  steer_data,
                                                                  test_size=0.15)'''