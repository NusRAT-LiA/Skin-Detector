import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    mask_images = []
    for i in range(555):
        image_path = f"{folder}/{str(i).zfill(4)}.jpg"
        mask_path = f"{folder}/Mask/{str(i).zfill(4)}.bmp"

        images.append(cv2.imread(image_path))
        mask_images.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))

    return images, mask_images

def calculate_probability(images, mask_images):
    skin_array = np.zeros((256, 256, 256), dtype=int)
    nonskin_array = np.zeros((256, 256, 256), dtype=int)
    
    for i in range(len(images)):
        mask = mask_images[i]
        img = images[i]

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                b, g, r = img[x, y]
                if mask[x, y] > 250:
                    nonskin_array[r, g, b] += 1
                else:
                    skin_array[r, g, b] += 1
   
        
    # Learning Probability
    skin_sum = np.sum(skin_array)
    nonskin_sum = np.sum(nonskin_array)
    print(skin_sum)
    skin_array = skin_array.astype(float)
    nonskin_array = nonskin_array.astype(float)
    epsilon = 1e-10  # a small value to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        learn_array = np.divide(skin_array, skin_sum + epsilon, out=np.zeros_like(skin_array), where=(skin_sum + epsilon)!=0)
        learn_array /= np.divide(nonskin_array, nonskin_sum + epsilon, out=np.zeros_like(nonskin_array), where=(nonskin_sum + epsilon)!=0)

    learn_array = np.nan_to_num(learn_array, nan=0.0)  # replace NaN with 0

    return learn_array

def apply_skin_detection(image, learn_array, threshold=0.35):
    result_image = image.copy()

    height, width, _ = image.shape

    for x in range(height):
        for y in range(width):
            b, g, r = image[x, y]
            if abs(learn_array[r, g, b]) > threshold:
                result_image[x, y] = [255, 255, 255]

    return result_image

def main():
    images, mask_images = load_images_from_folder("ibtd")
    print("eita normal image")
    learn_array = calculate_probability(images, mask_images)
    # Testing
    # test_image = cv2.imread("path/to/your/test/image.jpg")

    # if test_image is not None:
    #     result_image = apply_skin_detection(test_image, learn_array)

    #     # Visualization
    #     plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    #     plt.show()

if __name__ == "__main__":
    main()
