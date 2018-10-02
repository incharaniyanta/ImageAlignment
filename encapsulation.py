from __future__ import print_function
import cv2
import numpy as np

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img

def convert_to_grayscale(im):
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return imGray

def detectORB(MAX_MATCHES):
    orb = cv2.ORB_create(MAX_MATCHES)
    return orb

def compute_descriptors(orb, imGray):
    keypoints, descriptors = orb.detectAndCompute(imGray, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    return matches

def sort_matches_by_score(matches):
    matches.sort(key=lambda x: x.distance, reverse=False)
    return matches

def remove_bad_matches(matches,GOOD_MATCH_PERCENT):
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    return matches

def draw_top_matches(im1, keypoints1, im2, keypoints2, matches):
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches1.jpg", imMatches)
    print("done drawing top matches")


def extract_good_matches_loc(matches, keypoints1, keypoints2):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2

def find_homography(points1, points2):
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return h, mask

def use_homography(im1, im2):
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg

def save_aligned_img(img, outFilepath_name):
    print("Saving aligned image : ", outFilepath_name);
    cv2.imwrite(outFilepath_name, img)


if __name__ == '__main__':
    # Read reference image
    refFilename = "C:/Users/incha/PycharmProjects/ImageAlignment/form1.jpg"
    print("Reading reference image : ", refFilename)
    imReference = read_image(refFilename)

    # Read image to be aligned
    imFilename = "C:/Users/incha/PycharmProjects/ImageAlignment/scanned-form1.jpg"
    print("Reading image to align : ", imFilename);
    im = read_image(imFilename)

    print("Aligning images begins ...")
    # Registered image will be resotred in imReg towards the end.
    # The estimated homography will be stored in h at the end of the code.

    # Convert images to grayscale
    im1Gray = convert_to_grayscale(im)
    im2Gray = convert_to_grayscale(imReference)

    # Detect ORB features and compute descriptors.
    orb = detectORB(MAX_MATCHES)

    keypoints1, descriptors1 = compute_descriptors(orb, im1Gray)
    keypoints2, descriptors2 = compute_descriptors(orb, im2Gray)

    # Match features.
    matches = match_features(descriptors1, descriptors2)

    # Sort matches by score
    matches_sorted = sort_matches_by_score(matches)

    # Remove not so good matches
    matches_good = remove_bad_matches(matches_sorted, GOOD_MATCH_PERCENT)

    # Draw top matches
    draw_top_matches(im, keypoints1, imReference, keypoints2, matches_good)

    # Extract location of good matches
    points1, points2 = extract_good_matches_loc(matches_good, keypoints1, keypoints2)

    # Find homography
    h, mask = find_homography(points1, points2)

    # Use homography
    im1Reg = use_homography(im, imReference)

    # Write aligned image to disk.
    outFilepath_name = "C:/Users/incha/PycharmProjects/ImageAlignment/aligned1.jpg"
    save_aligned_img(im1Reg, outFilepath_name)

    # Print estimated homography
    print("Estimated homography : \n", h)



