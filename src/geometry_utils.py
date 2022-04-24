import numpy as np
import cv2


def get_camera_calibration_from_dataframe(df_calibration, image_id):

    """
    Get camera calibration of the given image_id

    Parameters
    ----------
    df_calibration (pandas.DataFrame of shape (n_images, 4)): Dataframe with image_id, camera_intrinsics, rotation_matrix and translation_vector columns
    image_id (str): Image ID

    Returns
    -------
    camera_intrinsics (numpy.ndarray of shape (3, 3)): Array of camera properties that determine the transformation between 3D points and 2D (pixel) coordinates
    rotation_matrix (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates
    translation_vector (numpy.ndarray of shape (3)): Array of position of the world origin in camera coordinates
    """

    row = df_calibration.loc[df_calibration['image_id'] == image_id]

    if len(row) >= 1:
        # Get image calibration if given image ID is valid
        camera_intrinsics = np.fromstring(row['camera_intrinsics'].values[0], sep=' ', dtype=np.float32).reshape(3, 3)
        rotation_matrix = np.fromstring(row['rotation_matrix'].values[0], sep=' ', dtype=np.float32).reshape(3, 3)
        translation_vector = np.fromstring(row['translation_vector'].values[0], sep=' ', dtype=np.float32)
    else:
        camera_intrinsics = None
        rotation_matrix = None
        translation_vector = None

    return camera_intrinsics, rotation_matrix, translation_vector


def get_fundamental_matrix_from_dataframe(df_pair_covisibility, pair):

    """
    Get fundamental matrix of the given pair

    Parameters
    ----------
    df_pair_covisibility (pandas.DataFrame of shape (n_pairs, 3)): Dataframe with pair, covisibility and fundamental_matrix columns
    pair (str): Image ID pair

    Returns
    -------
    fundamental_matrix (numpy.ndarray of shape (3, 3)): Array of fundamental matrix
    """

    row = df_pair_covisibility.loc[df_pair_covisibility['pair'] == pair]

    if len(row) >= 1:
        # Get fundamental matrix if given pair is valid
        fundamental_matrix = np.fromstring(row['fundamental_matrix'].values[0], sep=' ', dtype=np.float32).reshape(3, 3)
    else:
        fundamental_matrix = None

    return fundamental_matrix


def get_camera_extrinsics(rotation_matrix, translation_vector):

    """
    Create camera extrinsics by stacking rotation matrix and translation vector

    Parameters
    ----------
    rotation_matrix (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates
    translation_vector (numpy.ndarray of shape (3)): Array of position of the world origin in camera coordinates

    Returns
    -------
    camera_extrinsics (numpy.ndarray of shape (4, 4)): Array of world properties
    """

    camera_extrinsics = np.hstack([rotation_matrix, translation_vector.reshape(-1, 1)])
    camera_extrinsics = np.vstack([camera_extrinsics, [0, 0, 0, 1]])
    return camera_extrinsics


def get_essential_matrix(fundamental_matrix, camera_intrinsics1, camera_intrinsics2, keypoints1, keypoints2):

    """
    Get essential matrix

    Parameters
    ----------
    fundamental_matrix (numpy.ndarray of shape (3, 3)): Array of fundamental matrix
    camera_intrinsics1 (numpy.ndarray of shape (3, 3)): Array of camera properties of the first image
    camera_intrinsics2 (numpy.ndarray of shape (3, 3)): Array of camera properties of the second image
    keypoints1 (tuple of shape (n_keypoints)): Keypoints detected on the first image
    keypoints2 (tuple of shape (n_keypoints)): Keypoints detected on the second image

    Returns
    -------
    rotation_matrix (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates
    translation_vector (numpy.ndarray of shape (3, 3)): Array of position of the world origin in camera coordinates

    """

    essential_matrix = np.matmul(np.matmul(camera_intrinsics2.T, fundamental_matrix), camera_intrinsics1).astype(np.float64)
    normalized_keypoints1 = normalize_keypoints(keypoints=keypoints1, camera_intrinsics=camera_intrinsics1)
    normalized_keypoints2 = normalize_keypoints(keypoints=keypoints2, camera_intrinsics=camera_intrinsics2)
    n_inliers, rotation_matrix, translation_vector, inlier_mask = cv2.recoverPose(essential_matrix, normalized_keypoints1, normalized_keypoints2)

    return rotation_matrix, translation_vector.flatten(), inlier_mask.flatten().astype(bool)


def get_projection_matrix(camera_intrinsics, rotation_matrix, translation_vector):

    """
    Get calibration of the given image_id

    Parameters
    ----------
    camera_intrinsics (numpy.ndarray of shape (3, 3)): Array of camera properties that determine the transformation between 3D points and 2D (pixel) coordinates
    rotation_matrix (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates
    translation_vector (numpy.ndarray of shape (3, 3)): Array of position of the world origin in camera coordinates

    Returns
    -------
    projection_matrix (numpy.ndarray of shape (3, 4)): Array of projection matrix
    """

    camera_extrinsics = np.hstack([rotation_matrix, translation_vector.reshape(-1, 1)])
    projection_matrix = np.matmul(camera_intrinsics, camera_extrinsics)

    return projection_matrix


def normalize_keypoints(keypoints, camera_intrinsics):

    """
    Normalize detected keypoints using camera intrinsics

    Parameters
    ----------
    keypoints (np.ndarray of shape (n_keypoints, 2)): Keypoints detected on the source image
    camera_intrinsics (numpy.ndarray of shape (3, 3)): Array of camera properties that determine the transformation between 3D points and 2D (pixel) coordinates

    Returns
    -------
    normalized_keypoints (numpy.ndarray of shape (n_keypoints, 2)): Normalized keypoints
    """

    focal_length_x = camera_intrinsics[0, 0]
    focal_length_y = camera_intrinsics[1, 1]
    principal_point_offset_u = camera_intrinsics[0, 2]
    principal_point_offset_v = camera_intrinsics[1, 2]

    normalized_keypoints = (keypoints - np.array([[principal_point_offset_u, principal_point_offset_v]])) / np.array([[focal_length_x, focal_length_y]])
    return normalized_keypoints
