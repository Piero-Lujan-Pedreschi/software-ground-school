import cv2
import numpy as np

MIN_INLIERS = 30

def find_transform_and_inliers(greyscale_1, greyscale_2, debug=False):
    # Creates a SIFT (Scale-Invariant Feature Transform) object that will then be used to detect images to find stable landmark points and compute a descriptor for that point
    sift = cv2.SIFT_create()
    kepSIFT_1, desSIFT_1 = sift.detectAndCompute(greyscale_1, None)
    kepSIFT_2, desSIFT_2 = sift.detectAndCompute(greyscale_2, None)

    if desSIFT_1 is None or desSIFT_2 is None:
        return None, 0
    
    if debug:
        print(f'Num SIFT Key Points in Frame 1: {len(kepSIFT_1)}')
        print(f'Num SIFT Key Points in Frame 2: {len(kepSIFT_2)}')

    # Creates a Brute-Force Matcher which for every feature descriptor in first image, it calculates the distance to every single feature descriptor in the second image to find the closest ones.
    bf_SIFT = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=False)
    # Instead of just returning the absolute closest match for each feature, this method looks for the k nearest neighbors. Returns top k closest matches
    matches_SIFT = bf_SIFT.knnMatch(desSIFT_1, desSIFT_2, k=2)

    good_SIFT_matches = []
    # Loops through each specific feature from frame1 and compares the top 1 and 2 matches, if match_1 < 75% of match2 (Lowe's ratio), it appends to list of matches we will use
    for pair in matches_SIFT:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_SIFT_matches.append(m)

    # Visualizes the feature matching by stacking images side by side and drawing lines connecting their matching key points
    if debug:
        outImgSIFT = cv2.drawMatches(greyscale_1, kepSIFT_1,
                                     greyscale_2, kepSIFT_2,
                                     good_SIFT_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Good SIFT matches', outImgSIFT)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if len(good_SIFT_matches) < MIN_INLIERS:
        return None, 0

    pts1, pts2 = [], []
    for m in good_SIFT_matches:
        pts1.append(kepSIFT_1[m.queryIdx].pt)
        pts2.append(kepSIFT_2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # maps frame 2 (newer) into frame 1's (older) coordinates
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, 
                                             method=cv2.RANSAC, 
                                             ransacReprojThreshold=3.0)

    if M is None:
        return None, 0

    count = int(inliers.sum())
    if count < MIN_INLIERS:
        return None, 0

    return M, count