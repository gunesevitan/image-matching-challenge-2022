import os
import pathlib
import numpy as np
import pandas as pd
import cv2
import torch
import kornia
from kornia.feature import laf_from_center_scale_ori
from kornia_moons.feature import draw_LAF_matches
import matplotlib.pyplot as plt

import settings


def visualize_image(image, path=None):

    """
    Visualize image along with its mask

    Parameters
    ----------
    image (str or numpy.ndarray of shape (height, width, 3)): Image path relative to data directory or image array
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image, pathlib.Path) or isinstance(image, str):
        # Read image from the given path
        image_path = image
        image = cv2.imread(str(settings.DATA / image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        title = image_path

    elif isinstance(image, np.ndarray):
        title = f'Image {image.shape}'

    else:
        # Raise TypeError if image argument is not an array-like object or a path-like string
        raise TypeError('Image is not an array or path.')

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_image_pair(image_path1, image_path2, covisibility, path=None):

    """
    Visualize raw image pairs and display their covisibility

    Parameters
    ----------
    image_path1 (str): First image path relative to data path
    image_path2 (str): Second image path relative to data path
    covisibility (float): Covisibility value of image pair (0 <= covisibility <= 1)
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image1 = cv2.imread(f'{settings.DATA}/{image_path1}')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(f'{settings.DATA}/{image_path2}')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(figsize=(16, 32), ncols=2)
    axes[0].imshow(image1)
    axes[1].imshow(image2)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)
        if i == 0:
            axes[i].set_title(f'{image_path1.split("/")[-1]} - {image1.shape}', size=20, pad=15)
        elif i == 1:
            axes[i].set_title(f'{image_path2.split("/")[-1]} - {image2.shape}', size=20, pad=15)

    plt.suptitle(f'Scene: {image_path1.split("/")[1]} - Covisibility: {covisibility}', fontsize=20)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_covisibility_histogram(df_pair_covisibility, scene, path=None):

    """
    Visualize histograms of covisibilities

    Parameters
    ----------
    df_pair_covisibility (pandas.DataFrame of shape (n_pairs, 3)): Dataframe with pair, covisibility and fundamental_matrix columns
    scene (str): Name of the scene
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    mean_covisibility = df_pair_covisibility['covisibility'].mean()
    median_covisibility = df_pair_covisibility['covisibility'].median()
    std_covisibility = df_pair_covisibility['covisibility'].std()
    min_covisibility = df_pair_covisibility['covisibility'].min()
    max_covisibility = df_pair_covisibility['covisibility'].max()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(df_pair_covisibility['covisibility'], bins=255)
    ax.axvline(mean_covisibility, label='Mean', color='r', linewidth=2, linestyle='--')
    ax.axvline(median_covisibility, label='Median', color='b', linewidth=2, linestyle='--')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.legend(prop={'size': 15})
    ax.set_title(f'Covisibility {scene} - Mean: {mean_covisibility:.2f} Median: {median_covisibility:.2f} Std: {std_covisibility:.2f} Min: {min_covisibility:.2f} Max: {max_covisibility:.2f}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_matches(image1, image2, keypoints1, keypoints2, inliers, path=None):

    """
    Visualize matched keypoints of an image pair

    Parameters
    ----------
    image1 (torch.Tensor of shape (1, 3, height, width)): First image tensor
    image2 (torch.Tensor of shape (1, 3, height, width)): Second image tensor
    keypoints1 (numpy.ndarray of shape (n_keypoints, 2)): Keypoints from first image
    keypoints2 (numpy.ndarray of shape (n_keypoints, 2)): Keypoints from second image
    inliers (numpy.ndarray of shape (n_keypoints)): Inlier mask
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    draw_LAF_matches(
        lafs1=laf_from_center_scale_ori(
            torch.from_numpy(keypoints1).view(1, -1, 2),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1)
        ),
        lafs2=laf_from_center_scale_ori(
            torch.from_numpy(keypoints2).view(1, -1, 2),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1)
         ),
        tent_idxs=torch.arange(keypoints1.shape[0]).view(-1, 1).repeat(1, 2),
        img1=kornia.tensor_to_image(image1),
        img2=kornia.tensor_to_image(image2),
        inlier_mask=inliers,
        draw_dict={
            'inlier_color': (0.2, 1, 0.2),
            'tentative_color': None,
            'feature_color': (0.2, 0.5, 1),
            'vertical': False
        }
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


if __name__ == '__main__':

    VISUALIZE_COVISIBILITY_HISTOGRAMS = False

    if VISUALIZE_COVISIBILITY_HISTOGRAMS:
        # Visualize covisibility histograms of every training set scene
        train_scenes = [f for f in os.listdir(settings.DATA / 'train') if '.' not in f]
        for scene in train_scenes:
            df_pair_covisibility = pd.read_csv(f'../data/train/{scene}/pair_covisibility.csv')
            visualize_covisibility_histogram(
                df_pair_covisibility=df_pair_covisibility,
                scene=scene,
                path=settings.EDA / f'{scene}_covisibility_histogram.png'
            )
