import cv2
import matplotlib.pyplot as plt


def get_matcher(matcher_type, matcher_parameters):

    """
    Create feature matcher with specified parameters

    Parameters
    ----------
    matcher_type (str): Number of best features to keep (brute_force or flann)
    matcher_parameters (str): Keyword arguments passed to matcher constructor

    Returns
    -------
    matcher (cv2.BFMatcher or cv2.FlannBasedMatcher): Feature matcher
    """

    if matcher_type == 'brute_force':
        matcher = cv2.BFMatcher(**matcher_parameters)
    elif matcher_type == 'flann':
        matcher = cv2.FlannBasedMatcher(**matcher_parameters)
    else:
        matcher = None

    return matcher


def visualize_matches(image1, keypoints1, image2, keypoints2, matches, path=None):

    """
    Visualize feature matches drawn on two images

    Parameters
    ----------
    image1 (numpy.ndarray of shape (height, width, channel)): First image
    keypoints1 (tuple of shape (n_keypoints)): Keypoints detected on the first image
    image2 (numpy.ndarray of shape (height, width, channel)): Second image
    keypoints2 (tuple of shape (n_keypoints)): Keypoints detected on the second image
    matches (list of shape (n_matches)): Matched features
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image_matches = cv2.drawMatches(
        img1=image1,
        keypoints1=keypoints1,
        img2=image2,
        keypoints2=keypoints2,
        matches1to2=matches,
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image_matches)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Image SIFT Keypoints {image_matches.shape}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
