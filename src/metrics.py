import numpy as np


def calculate_rotation_and_translation_error(q_ground_truth, t_ground_truth, q_prediction, t_prediction, scaling_factor, epsilon=1e-15):

    """
    Calculate rotation and translation error for an image pair

    Parameters
    ----------
    q_ground_truth (numpy.ndarray of shape (4)): Array of quaternion derived from ground truth rotation matrix
    t_ground_truth (numpy.ndarray of shape (3)): Array of ground truth translation vector
    q_prediction (numpy.ndarray of shape (4)): Array of quaternion derived from estimated rotation matrix
    t_prediction (numpy.ndarray of shape (3)): Array of estimated translation vector
    scaling_factor (float): Scaling factor of the scene
    epsilon (float): Small number for preventing zero division

    Returns
    -------
    rotation_error (float): Rotation error in terms of degrees
    translation_error (float): Translation error in terms of meters
    """

    q_ground_truth_norm = q_ground_truth / (np.linalg.norm(q_ground_truth) + epsilon)
    q_prediction_norm = q_prediction / (np.linalg.norm(q_prediction) + epsilon)
    loss_q = np.maximum(epsilon, (1.0 - np.sum(q_prediction_norm * q_ground_truth_norm) ** 2))
    rotation_error = np.arccos(1 - (2 * loss_q)) * 180 / np.pi

    t_ground_truth_scaled = t_ground_truth * scaling_factor
    t_prediction_scaled = t_prediction * np.linalg.norm(t_ground_truth) * scaling_factor / (np.linalg.norm(t_prediction) + epsilon)
    translation_error = min(np.linalg.norm(t_ground_truth_scaled - t_prediction_scaled), np.linalg.norm(t_ground_truth_scaled + t_prediction_scaled))

    return rotation_error, translation_error


def calculate_mean_average_accuracy(rotation_errors, translation_errors):

    """
    Calculate mean average accuracies for the scene

    Parameters
    ----------
    rotation_errors (numpy.ndarray of shape (n_samples)): Array of calculated rotation errors
    translation_errors (numpy.ndarray of shape (n_samples)): Array of calculated translation errors

    Returns
    -------
    mean_average_accuracy (float): Mean average accuracy calculated on both rotation and translation errors
    mean_average_accuracy_rotation (float): Mean average accuracy calculated on both rotation errors
    mean_average_accuracy_translation (float): Mean average accuracy calculated on both translation errors
    """

    rotation_error_thresholds = np.linspace(1, 10, 10)
    translation_error_thresholds = np.geomspace(0.2, 5, 10)

    accuracy, accuracy_rotation, accuracy_translation = [], [], []
    for rotation_error_threshold, translation_error_threshold in zip(rotation_error_thresholds, translation_error_thresholds):
        accuracy += [(np.bitwise_and(rotation_errors < rotation_error_threshold, translation_errors < translation_error_threshold)).sum() / len(rotation_errors)]
        accuracy_rotation += [(rotation_errors < rotation_error_threshold).sum() / len(rotation_errors)]
        accuracy_translation += [(translation_errors < translation_error_threshold).sum() / len(translation_errors)]

    mean_average_accuracy = np.mean(accuracy)
    mean_average_accuracy_rotation = np.mean(accuracy_rotation)
    mean_average_accuracy_translation = np.mean(accuracy_translation)

    return mean_average_accuracy, mean_average_accuracy_rotation, mean_average_accuracy_translation
