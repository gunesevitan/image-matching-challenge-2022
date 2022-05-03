from tqdm import tqdm
from glob import glob
import numpy as np
import cv2

import settings


STATISTICS = {'mean': [0.50201953, 0.50477692, 0.49493494], 'std': [0.26986689, 0.27720259, 0.30068508]}

if __name__ == '__main__':

    filenames = glob(f'{settings.DATA}/train/*/images/*.jpg')

    pixel_count = 0
    pixel_sum = 0
    pixel_squared_sum = 0

    for image in tqdm(filenames):

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.

        pixel_count += (image.shape[0] * image.shape[1])
        pixel_sum += np.sum(image, axis=(0, 1))
        pixel_squared_sum += np.sum(image ** 2, axis=(0, 1))

    mean = pixel_sum / pixel_count
    var = (pixel_squared_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    print(f'{len(filenames)} Images - Mean: {mean} - Standard Deviation: {std}')
