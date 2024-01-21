import time
import cv2
import os


# ​save pixel numbers of skin and non-skin pixels
skinPixelNumber = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]
nonskinPixelNumber = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]
# global skinPixels
# global nonSkinPixels
skin = 0
nonSkin = 0


def take_input():
    images = []
    mask_images = os.listdir("./ibtd/Mask")
    for i in range(555):
        tempList = mask_images[i].split(".")
        temp = tempList[0] + ".jpg"
        images.append(temp)
    return images, mask_images


nonSkinPixels = 0
skinPixels = 0


def train(images, mask_images):
    global skinPixels, nonSkinPixels
    for i in range(0, len(images) - 56):
        # print("eita normal image ", images[i], "ar eita masked ",mask_images[i])
        image = cv2.imread("./ibtd/" + images[i])
        mask_image = cv2.imread("./ibtd/Mask/" + mask_images[i])
        width, height, _ = image.shape
        # print(width, " ", height)

        for h in range(height):
            for w in range(width):
                blue = image[w][h][0]
                green = image[w][h][1]
                red = image[w][h][2]
                maskBlue = mask_image[w][h][0]
                maskGreen = mask_image[w][h][1]
                maskRed = mask_image[w][h][2]
                if maskRed > 250 and maskGreen > 250 and maskBlue > 250:
                    nonskinPixelNumber[red][green][blue] += 1
                    # print("#")
                    nonSkinPixels += 1
                else:
                    skinPixelNumber[red][green][blue] += 1
                    # print(".")
                    skinPixels += 1

    print(nonSkinPixels, " ", skinPixels)


# # write final probability of skin/non-skin to file
def write_ratio(T):
    print(nonSkinPixels, " ", skinPixels)
    fp = open("./ratio.txt", "w")
    for r in range(256):
        for g in range(256):
            for b in range(256):
                if skinPixels != 0 and nonSkinPixels != 0:
                    # print(".")
                    skin = skinPixelNumber[r][g][b] / skinPixels  # P(C|S)
                    nonSkin = nonskinPixelNumber[r][g][b] / nonSkinPixels  # P(C|NS)
                    if nonSkin != 0:
                        T = skin / nonSkin
                    else:
                        T = 0.0
                # print(skinPixelNumber[r][g][b]," ",nonskinPixelNumber[r][g][b])
                fp.write(f"{T}\n")
                # print(T)
    fp.close()


import numpy as np


def calculate_accuracy():
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    trained_value = np.zeros(shape=(256, 256, 256))
    ratio = open("./ratio.txt", "r")
    for i in range(256):
        for j in range(256):
            for k in range(256):
                val = ratio.readline()
                if val:
                    trained_value[i][j][k] = float(val)

    # print(trained_value)
    for i in range(500, len(images) - 1):
        new_img = cv2.imread("./ibtd/" + images[i])
        new_mask_img = cv2.imread("./ibtd/mask/" + mask_images[i])
        height, width, _ = new_img.shape
    if new_img is not None and new_mask_img is not None: 
      for x in range(height):
            for y in range(width):
                red = new_img[x, y, 0]
                green = new_img[x, y, 1]
                blue = new_img[x, y, 2]
                if trained_value[red, green, blue] <= 0.4:
                    new_img[x, y, 0] = 255
                    new_img[x, y, 1] = 255
                    new_img[x, y, 2] = 255
                    if new_img[x, y, 0] == new_mask_img[x, y, 0]:
                        fn += 1
                    else:
                        tn += 1
                else:
                    if new_img[x, y, 0] == new_mask_img[x, y, 0]:
                        tp += 1
                    else:
                        fp += 1
    res = 0
    hor = tp + fn
    lob = tp + fp + tn + fn
    print(tp, " ", tn, " ", fp, " ", fn)
    if lob != 0:
        res = hor / lob
    print(res)


start = time.time()
images = []
mask_images = []
for i in range(1):
    images, mask_images = take_input()
#   train(images,mask_images)

# write_ratio(0.0)
calculate_accuracy()
end = time.time()
#print(end - start)