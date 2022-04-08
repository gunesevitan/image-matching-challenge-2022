import pandas as pd

import settings


def read_pair_covisibility(scene, covisibility_threshold=0.1, shuffle=True):

    """
    Read pair_covisibility.csv file of the given scene

    Parameters
    ----------
    scene (str): Scene name
    covisibility_threshold (float): Threshold for filtering out low covisible pairs
    shuffle (bool): Whether to shuffle pairs or not

    Returns
    -------
    df_pair_covisibility [pandas.DataFrame of shape (n_pairs, 3)]: Dataframe with pair, covisibility and fundamental_matrix columns
    """

    df_pair_covisibility = pd.read_csv(settings.DATA / 'train' / scene / 'pair_covisibility.csv')

    if covisibility_threshold >= 0.:
        df_pair_covisibility = df_pair_covisibility.loc[df_pair_covisibility['covisibility'] >= covisibility_threshold, :]

    if shuffle:
        df_pair_covisibility = df_pair_covisibility.sample(frac=1).reset_index(drop=True)

    return df_pair_covisibility


def read_calibration(scene):

    """
    Read calibration.csv file of the given scene

    Parameters
    ----------
    scene (str): Scene name

    Returns
    -------
    df_calibration [pandas.DataFrame of shape (n_images, 4)]: Dataframe with image_id, camera_intrinsics, rotation_matrix and translation_vector columns
    """

    df_calibration = pd.read_csv(settings.DATA / 'train' / scene / 'calibration.csv')

    return df_calibration
