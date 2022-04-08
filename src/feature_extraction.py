import cv2
import matplotlib.pyplot as plt


def get_sift_detector(n_features, n_octave_layers=3, contrast_threshold=0.09, edge_threshold=0.1, sigma=1.6):

    """
    Create SIFT feature detector with specified parameters

    Parameters
    ----------
    n_features (int): Number of best features to keep
    n_octave_layers (int): Number of layers in each octave
    contrast_threshold (float): Threshold to filter out weak features in low-contrast regions
    edge_threshold (float): Threshold to filter out edge-like features
    sigma (float): Sigma of the gaussian filter

    Returns
    -------
    detector (cv2.SIFT): SIFT feature detector
    """

    detector = cv2.SIFT_create(
        nfeatures=n_features,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    return detector


def extract_sift_features(image, detector):

    """
    Extract SIFT features on given image with given detector

    Parameters
    ----------
    image [numpy.ndarray of shape (height, width, channel)]: Image
    detector (cv2.SIFT): SIFT feature detector

    Returns
    -------
    keypoints [tuple of shape (n_features)]: Keypoints detected on the image
    descriptors [np.ndarray of shape (n_features, 128)]: Descriptors extracted on the image
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = detector.detectAndCompute(grayscale_image, None)

    return keypoints, descriptors


def visualize_keypoints(source_image, keypoints, output_image, path=None):

    """
    Visualize keypoints drawn on image

    Parameters
    ----------
    source_image [numpy.ndarray of shape (height, width, channel)]: Source image on which the keypoints are detected
    keypoints [tuple of shape (n_keypoints)]: Keypoints detected on the source image
    output_image [numpy.ndarray of shape (height, width, channel)]: Output image to draw keypoints
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image_keypoints = cv2.drawKeypoints(
        image=source_image,
        keypoints=keypoints,
        outImage=output_image,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image_keypoints)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Image Keypoints {image_keypoints.shape}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
