import logging
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2

import settings


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
    dataset_statistics = {'mean': list(mean), 'std': list(std)}

    logging.info(f'{len(filenames)} Images - Mean: {mean} - Standard Deviation: {std}')
    with open(settings.DATA / 'dataset_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_statistics, f, ensure_ascii=False, indent=4)
