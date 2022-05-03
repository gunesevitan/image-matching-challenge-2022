import cv2


def get_fundamental_matrix(keypoints1, keypoints2, method=cv2.USAC_MAGSAC, ransac_reproj_threshold=0.25, confidence=0.99999, max_iters=100000):

    """
    Estimate fundamental matrix using predicted keypoints from an image pair with RANSAC

    Parameters
    ----------
    keypoints1 (numpy.ndarray of shape (n_keypoints, 2)): Keypoints from first image
    keypoints2 (numpy.ndarray of shape (n_keypoints, 2)): Keypoints from second image
    method (int): OpenCV method flag for computing the fundamental matrix
    ransac_reproj_threshold (float): Maximum distance from a point to an epipolar line in pixels
    confidence (float): Desirable level of confidence that estimated matrix is correct
    max_iters (int): Number of iterations

    Returns
    -------
    fundamental_matrix (numpy.ndarray of shape (3, 3)): Array of fundamental matrix
    inliers (numpy.ndarray of shape (n_keypoints)): Inlier mask
    """

    fundamental_matrix, inliers = cv2.findFundamentalMat(
        points1=keypoints1,
        points2=keypoints2,
        method=method,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=confidence,
        maxIters=max_iters
    )
    inliers = inliers.reshape(-1).astype(bool)

    return fundamental_matrix, inliers
