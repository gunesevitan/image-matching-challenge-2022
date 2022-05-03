import numpy as np


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


def decompose_fundamental_matrix_with_camera_intrinsics(fundamental_matrix, camera_intrinsics1, camera_intrinsics2):

    """
    Decompose fundamental matrix into rotation matrices and translation vector

    Parameters
    ----------
    fundamental_matrix (numpy.ndarray of shape (3, 3)): Array of fundamental matrix
    camera_intrinsics1 (numpy.ndarray of shape (3, 3)): Array of camera intrinsics from the first image
    camera_intrinsics2 (numpy.ndarray of shape (3, 3)): Array of camera intrinsics from the second image

    Returns
    -------
    rotation_matrix1 (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates for the first image
    rotation_matrix2 (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates for the second image
    translation_vector (numpy.ndarray of shape (3)): Array of position of the world origin in camera coordinates
    """

    essential_matrix = np.matmul(camera_intrinsics2.T, np.matmul(fundamental_matrix, camera_intrinsics1)).astype(np.float64)
    u, s, v = np.linalg.svd(essential_matrix)

    if np.linalg.det(u) < 0:
        u *= -1
    if np.linalg.det(v) < 0:
        v *= -1

    w = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotation_matrix1 = np.matmul(u, np.matmul(w, v))
    rotation_matrix2 = np.matmul(u, np.matmul(w.T, v))
    translation_vector = u[:, -1]

    return rotation_matrix1, rotation_matrix2, translation_vector


def rotation_matrix_to_quaternion(rotation_matrix):

    """
    Convert rotation matrix to quaternion

    Parameters
    ----------
    rotation_matrix (numpy.ndarray of shape (3, 3)): Array of directions of the world-axes in camera coordinates

    Returns
    -------
    quaternion (numpy.ndarray of shape (4)): Array of quaternion
    """

    r00 = rotation_matrix[0, 0]
    r01 = rotation_matrix[0, 1]
    r02 = rotation_matrix[0, 2]
    r10 = rotation_matrix[1, 0]
    r11 = rotation_matrix[1, 1]
    r12 = rotation_matrix[1, 2]
    r20 = rotation_matrix[2, 0]
    r21 = rotation_matrix[2, 1]
    r22 = rotation_matrix[2, 2]

    k = np.array(
        [[r00 - r11 - r22, 0.0, 0.0, 0.0],
         [r01 + r10, r11 - r00 - r22, 0.0, 0.0],
         [r02 + r20, r12 + r21, r22 - r00 - r11, 0.0],
         [r21 - r12, r02 - r20, r10 - r01, r00 + r11 + r22]]
    )
    k /= 3.0

    # Quaternion is the eigenvector of k that corresponds to the largest eigenvalue
    w, v = np.linalg.eigh(k)
    quaternion = v[[3, 0, 1, 2], np.argmax(w)]

    if quaternion[0] < 0:
        np.negative(quaternion, quaternion)

    return quaternion
