import cv2
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np

# Argument parsing variable declared
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                required=True,
                help="Path to folder")
args = vars(ap.parse_args())

# Find all the images in the provided images folder
mypath = args["image"]
onlyFiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = np.empty(len(onlyFiles), dtype=object)

# Iterate through every image and resize all the images
for n in range(0, len(onlyFiles)):
    path = join(mypath, onlyFiles[n])
    images[n] = cv2.imread(join(mypath, onlyFiles[n]), cv2.IMREAD_UNCHANGED)

    # Load the image in img variable
    img = cv2.imread(path, 1)

    # Define a resizing Scale to declare how much to resize
    # resize_scaling = 50
    # resize_width = int(img.shape[1] * resize_scaling / 100)
    # resize_hieght = int(img.shape[0] * resize_scaling / 100)
    # resized_dimensions = (resize_width, resize_hieght)
    # resized_dimensions = (512, 384)
    resized_dimensions = (640, 480)

    # Create resized image using the calculated dimensions
    resized_image = cv2.resize(img, resized_dimensions, interpolation=cv2.INTER_AREA)

    # Save the image in Output Folder
    # cv2.imwrite('output/' + str(onlyFiles[n].split(".")[0]) + '_resized.jpg', resized_image)
    cv2.imwrite('output/' + str(onlyFiles[n].split(".")[0]) + '.jpg', resized_image)

print("Images resized Successfully")

