import cv2


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
