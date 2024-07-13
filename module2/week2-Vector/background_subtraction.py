import numpy as np
import cv2
import os
# from google.colab.patches import cv2_imshow

class BackgroundSubtraction:
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder

    def load_and_resize_image(self, filename, size=(678, 381)):
        img_path = os.path.join(self.data_folder, filename)
        if not os.path.exists(img_path):
            print(f"Image file '{img_path}' does not exist.")
            return None
        
        img = cv2.imread(img_path, 1)        
        img = cv2.resize(img, size)
        return img

    def store_images(self, bg1_filename, bg2_filename, ob_filename, size=(678, 381)):
        self.bg1_image = self.load_and_resize_image(bg1_filename, size)
        self.bg2_image = self.load_and_resize_image(bg2_filename, size)
        self.ob_image = self.load_and_resize_image(ob_filename, size)
        
        if self.bg1_image is None or self.bg2_image is None or self.ob_image is None:
            raise ValueError("One or more images failed to load.")

    @staticmethod
    def compute_difference(bg_img, input_img):
        diff_img = cv2.absdiff(bg_img, input_img)
        gray_diff = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        return gray_diff

    @staticmethod
    def compute_binary_mask(diff_img, threshold=30):
        _, binary_mask = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)
        # np.where converts uint8 -> uint64
        # binary_mask = np.where(diff_img >= threshold, 255, 0)
        # print(binary_mask.dtype)
        # binary_mask = binary_mask.astype('uint8')

        return binary_mask

    def replace_background(self):
        difference_single_channel = self.compute_difference(self.bg1_image, self.ob_image)
        binary_mask = self.compute_binary_mask(difference_single_channel)
        output = np.where(binary_mask[:, :, np.newaxis] == 255, self.ob_image, self.bg2_image)

        return output
