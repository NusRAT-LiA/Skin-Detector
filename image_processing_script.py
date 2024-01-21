import random
import cv2
import os
import numpy as np
import zipfile

def extract_zip(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# def remove_unwanted_files(directory):
#     os.remove(os.path.join(directory, 'Thumbs.db'))
#     os.remove(os.path.join(directory, 'Mask/Thumbs.db'))
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".ipynb_checkpoints"):
#                 os.remove(os.path.join(root, file))
def take_input():
    images = os.listdir("./ibtd_unzip/ibtd")
        
    mask_images = os.listdir("./ibtd_unzip/ibtd/Mask")
    
    if len(mask_images) >= 555:
        images = random.sample(mask_images, 555)
    else:
        print("Not enough images for sampling.")
        # Handle the situation based on your requirements
        
    return images, mask_images

def train(images, mask_images, skinPixelNumber, nonskinPixelNumber):
    skinPixels = 0
    nonSkinPixels = 0

    for i in range(0, len(images) - 56):
        image_path = os.path.join('./ibtd_unzip/ibtd', images[i])
        mask_image_path = os.path.join('./ibtd_unzip/ibtd/Mask', mask_images[i])

        # Check if both image and mask exist
        if os.path.exists(image_path) and os.path.exists(mask_image_path):
            image = cv2.imread(image_path)
            mask_image = cv2.imread(mask_image_path)

            # Check if the images have valid shapes
            if image is not None and mask_image is not None:
                width, height, _ = image.shape

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
                            nonSkinPixels += 1
                        else:
                            skinPixelNumber[red][green][blue] += 1
                            skinPixels += 1
            else:
                print(f"Skipped invalid image: {images[i]}")

    return skinPixels, nonSkinPixels

def write_ratio(skinPixels, nonSkinPixels):
    with open("./ibtd_unzip/ratio.txt", "w") as fp:
        for r in range(256):
            for g in range(256):
                for b in range(256):
                    if skinPixels != 0 and nonSkinPixels != 0:
                        skin = skinPixelNumber[r][g][b] / skinPixels
                        nonSkin = nonskinPixelNumber[r][g][b] / nonSkinPixels
                        T = skin / nonSkin if nonSkin != 0 else 0.0
                        fp.write(f"{T}\n")
                    else:
                        fp.write("0.0\n")  # Handle the case where skinPixels or nonSkinPixels is zero
def calculate_accuracy(images, mask_images):
    tp, tn, fp, fn = 0, 0, 0, 0
    trained_value = np.zeros(shape=(256, 256, 256))
    
    with open("./ibtd_unzip/ratio.txt", "r") as ratio:
        for i in range(256):
            for j in range(256):
                for k in range(256):
                    val = ratio.readline().strip()  # Strip leading and trailing whitespaces
                    if val:
                        trained_value[i][j][k] = float(val)

    for i in range(500, len(images) - 1):
        # Load the new_img and new_mask_img
        new_img = cv2.imread(os.path.join('./ibtd', images[i]))
        new_mask_img = cv2.imread(os.path.join('./ibtd/Mask', mask_images[i]))

        # Check if the image loading was successful
        if new_img is None or new_mask_img is None:
            print(f"Skipped invalid image: {images[i]}")
            continue

        height, width, _ = new_img.shape

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
                        tn += 1
                    else:
                        fn += 1
                else:
                    if new_img[x, y, 0] == new_mask_img[x, y, 0]:
                        tp += 1
                    else:
                        fp += 1

    res = 0
    hor = tp + tn
    lob = tp + fp + tn + fn
    
    print('True positive', tp, 'True Negative', tn, 'False Positive', fp, 'False Negative', fn)
    
    if lob != 0:
        res = hor / lob
    
    print('Accuracy is:', res)




zip_file_path = './ibtd.zip'
extract_path = './ibtd_unzip/ibtd'

# Extract the zip file
extract_zip(zip_file_path, extract_path)

# Remove unwanted files
# remove_unwanted_files(extract_path)

# Initialize pixel counters
skinPixelNumber = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]
nonskinPixelNumber = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]

# Loop for training and evaluating
for _ in range(10):
    images = os.listdir(extract_path)
    images, mask_images = take_input()
    

    skinPixels, nonSkinPixels = train(images, mask_images, skinPixelNumber, nonskinPixelNumber)
    write_ratio(skinPixels, nonSkinPixels)
    calculate_accuracy(images, mask_images)
