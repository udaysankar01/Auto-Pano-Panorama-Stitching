#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import os
from skimage.feature import peak_local_max
import argparse

# Add any python libraries here

def readImages(images_path):
    """
		Takes in the path of folder containing the input images,
  		and returns the images inside the folder in the form of a list.
    """
    print(f"Reading input images from '{images_path}'")
    image_list = []
    image_file_names = sorted(os.listdir(images_path))
    print(f"The files found: {image_file_names}")
    for file_name in image_file_names:
        image_path = images_path + "/" + file_name
        image = cv2.imread(image_path)
        if image is None:
            raise TypeError(f"Error loading {image_path}")
        else:
            image_list.append(image)
        
    return image_list


def displayImages(images, save_path):
    """
        Takes the image list as input, resizes the images into same size and 
        then saves and displays them.
    """
    ############################################## add saving feature ##############################################
    for i, image in enumerate(images):
        cv2.imshow(f'{i}', image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        
def getCorners(images, CornerDetectionType):
    """
        Takes a list of images as input and returns a list of images whose
        corners are highlighted in red color.
    """
    print("Corner Detection Initiated...\n")
    corner_images = []
    corner_maps = []
    corner_coords = []
    
    for i, image in enumerate(images):
        img = image.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corner_map = np.zeros(np.shape(gray_img))
        
        if CornerDetectionType == 1:    

            # Harris Corner Detection Method
            print(f"\nUsing Harris Corner Detection Method for image {i+1}.")
            threshold = 0.0001
            dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
            dst[dst < 0.001 * dst.max()] = 0
            corner_maps.append(dst)

            corner_coordinate = np.where(dst > threshold * dst.max())
            corner_coords.append(corner_coordinate)

            img[dst > threshold * dst.max()] = [0, 0, 255]
            corner_images.append(img)

        else:             

            # Shi-Tomasi Corner Detection Method
            print(f"\nUsing Shi-Tomasi Corner Detection Method for image {i+1}.")
            dst = cv2.goodFeaturesToTrack(gray_img, 100000, 0.001, 10)
            dst = np.int0(dst)
            corner_coords.append(dst)
            
            for i in dst:
                x,y = i.ravel()
                cv2.circle(img,(x,y),2,(0,0,255),-1)
                cv2.circle(corner_map,(x,y),2,255,-1)

            corner_images.append(img)
            corner_maps.append(corner_map)
            
    return corner_images, corner_maps, corner_coords


def ANMS(images, corner_maps, N_best=500):
    """ 
        Takes in images and corresponding corner maps as inputs and performs
        Adaptive Non-Maximum Suppression on the corner maps and returns the 
        Suppressed images as output.  
    """
    imgs = images.copy()
    anms_images = []
    corner_coords = []

    for image_idx in range(len(imgs)):

        print(f"\nAdaptive Non-Maximum Suppression initiated for image {image_idx+1}.")
        img = imgs[image_idx].copy()
        corner_map = corner_maps[image_idx]
        local_maxima_list = peak_local_max(corner_map, 15)
        N_strong = len(local_maxima_list)

        r = [np.Inf for i in range(N_strong)]
        x, y = np.zeros((N_strong, 1)), np.zeros((N_strong, 1))
        x_best, y_best = np.zeros((N_best, 1)), np.zeros((N_best, 1))
        ED = 0

        for i in range(N_strong):
            for j in range(N_strong):
                x_j, y_j = local_maxima_list[j]
                x_i, y_i = local_maxima_list[i]

                if (corner_map[x_j, y_j] > corner_map[x_i,y_i]):
                    ED = (x_j - x_i)**2 + (y_j - y_i)**2

                if ED < r[i]:
                    r[i] = ED
                    x[i] = x_j
                    y[i] = y_j

        if len(x) < N_best:
            N_best = len(x)
        
        best_index = np.argsort(r)[: : -1][: N_best]

        for i in range(N_best):

            x_best[i], y_best[i] = y[best_index[i]], x[best_index[i]]
            cv2.circle(img, (int(x_best[i]), int(y_best[i])), 2, (0, 0, 255), -1)
        
        anms_corner = np.concatenate((x_best, y_best), axis = 1).astype(int)
        corner_coords.append(anms_corner)
        anms_images.append(img)

    return anms_images, corner_coords


def featureDescriptor(gray_image, x, y, patch_size=41):
    """
        Takes a grayscale image and a cartesian co-ordinate of a feature that needs to described as input
        and returns a feature vector that encodes the information regarding the feature point.
    """
    gray_img = gray_image.copy()
    patch = gray_img[int(x - patch_size/2): int(x + patch_size/2), int(y - patch_size/2): int(y + patch_size/2)]
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    patch_subsampled = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
    feature_vector = patch_subsampled.reshape(64)
    feature_vector_std = (feature_vector - feature_vector.mean()) / (feature_vector.std() + 1e-10)

    return feature_vector_std


def matchFeatures(image1, image2, corner_coords1, corner_coords2, patch_size=41, comparison_ratio=1):
    """
        Takes two images and their corresponding corner co-ordinates as input and matches the feature points
        that will later be stitched together for the panorama view.
    """
    print("\nMatching of features initiated.")
    img1 = image1.copy()
    img2 = image2.copy()
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    width1, height1 = gray_img1.shape
    width2, height2 = gray_img2.shape
    features1, features2 = [], []
    corners1, corners2 = [], []
    matched_coords = []

    for corner in corner_coords1:
        x, y = corner.ravel()

        if ((int(x - patch_size/2) > 0) and (int(x + patch_size/2) < height1)) and ((int(y - patch_size/2) > 0) and (int(y + patch_size/2) < width1)):
           
            feature_description = featureDescriptor(gray_img1, y, x, 41)
            features1.append(feature_description)
            corners1.append([x, y])

    for corner in corner_coords2:
        x, y = corner.ravel()

        if ((int(x - patch_size/2) > 0) and (int(x + patch_size/2) < height2)) and ((int(y - patch_size/2) > 0) and (int(y + patch_size/2) < width2)):
           
            feature_description = featureDescriptor(gray_img2, y, x, 41)
            features2.append(feature_description)
            corners2.append([x, y])
            
    
    for i in range(len(features1)):
        SSD = []
        for j in range(len(features2)):
            ssd_val = np.linalg.norm((features1[i]-features2[j]))**2
            SSD.append(ssd_val)

        sorted_SSD_index = np.argsort(SSD)
        if (SSD[sorted_SSD_index[0]] / SSD[sorted_SSD_index[1]]) < comparison_ratio:
            matched_coords.append((corners1[i], corners2[sorted_SSD_index[0]]))

    print(f"Number of matched co-ordinates: {len(matched_coords)}")

    if len(matched_coords) < 30:
        print('\nNot enough matches!\n')
        quit()

    return np.array(matched_coords)



def visualizeMatchedFeatures(image1, image2, matched_coords):
    """
        Takes two images and list of co-ordiates of matched features in the images
        and gives a visulaization of the mapping of these features across the images.
    """
    img1 = image1.copy()
    img2 = image2.copy()
    print("\nVisualizing the mapping of matched features of the two images.")

    # images has to be resized into same size (height needs to be same for concatenation)
    image_sizes = [img1.shape, img2.shape]
    target_size = np.max(image_sizes, axis=0)
    if img1.shape != list(target_size):
        print("\nResizing image 1 for proper visualization.")
        img1 = cv2.resize(img1, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    if img2.shape != list(target_size):
        print("\nResizing image 2 for proper visualization.")
        img2 = cv2.resize(img2, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    
    concatenated_image = np.concatenate((img1, img2), axis=1)
    corners1, corners2 = matched_coords[:, 0].astype(int).copy(), matched_coords[:, 1].astype(int).copy()
    corners2[:,0] += img1.shape[1]
    
    for coord1, coord2 in zip(corners1, corners2):
        cv2.line(concatenated_image, (coord1[0], coord1[1]), (coord2[0], coord2[1]), (0, 255, 255), 1)
        cv2.circle(concatenated_image, (coord1[0], coord1[1]), 3, (0,0,255), 1)
        cv2.circle(concatenated_image, (coord2[0], coord2[1]), 3, (0,255,0), 1)
    
    cv2.imwrite('matching_sample.png', concatenated_image)

def RANSAC(matched_coords, accuracy=0.9, threshold=5):
    """
        Takes in the matched co-ordinated as input and remove the wrong matchings
        by computing Homography and removing outliers
    """
    corners1 = matched_coords[:,0].copy()
    num_pairs = corners1.shape[0]
    corners2 = matched_coords[:,1].copy()
    N_best = 0
    H_best = np.zeros((3, 3))
    iterations = 3000

    for i in range(iterations):

        # select four random points
        random_index = np.random.choice(num_pairs, size=4)
        random_corners1 = np.array(corners1[random_index])
        random_corners2 = np.array(corners2[random_index])

        #compute homography
        H = cv2.getPerspectiveTransform(np.float32(random_corners1), np.float32(random_corners2)) ##########################################################################

        corners1_transformed = np.vstack((corners1[:,0], corners1[:,1], np.ones([1, num_pairs])))
        corners1_transformed = np.dot(H, corners1_transformed)
        corners1_transformed = np.array([corners1_transformed[0]/(corners1_transformed[2]+1e-10), corners1_transformed[1]/(corners1_transformed[2]+1e-10)]).T

        SSD = np.sum((corners2 - corners1_transformed)**2, axis=1)
        SSD[SSD <= threshold] = 1
        SSD[SSD > threshold] = 0
        N = np.sum(SSD)

        if N > N_best:
            N_best = N
            H_best = H
            matched_coords_filtered_indices = np.where(SSD == 1)
        
    corners_filtered1 = corners1[matched_coords_filtered_indices]
    corners_filtered2 = corners2[matched_coords_filtered_indices]
    print("Number of matched features after RANSAC: ", len(corners_filtered1))

    matched_coords_filtered = np.zeros([corners_filtered1.shape[0], corners_filtered1.shape[1], 2])
    matched_coords_filtered[:, 0, :], matched_coords_filtered[:, 1, :] = corners_filtered1, corners_filtered2

    return matched_coords_filtered.astype(int), H_best


def warpAndBlendImages(image1, image2, H, i, saveLocation):
    """
        Takes two images and homography matrix as input and lends the two images into a single seamless image.
    """
    img1 = image1.copy()
    img2 = image2.copy()
    h1, w1, _ = img1.shape
    h2,w2, _ = img2.shape

    img1_points = np.array([[0.0, 0.0], [0.0, h1], [w1, h1], [w1, 0.0]]).reshape(-1, 1, 2)
    img1_points_transformed = cv2.perspectiveTransform(img1_points, H)

    img2_points = np.array([[0.0, 0.0], [0.0, h2], [w2, h2], [w2, 0.0]]).reshape(-1, 1, 2)
    blend_img_points = np.concatenate((img1_points_transformed, img2_points), axis=0)
    blend_img_points_xy = []
    for idx in range(len(blend_img_points)):
        blend_img_points_xy.append(blend_img_points[idx].ravel())

    blend_img_points_xy = np.array(blend_img_points_xy)

    x_min, y_min = np.min(blend_img_points_xy, axis=0).astype(int)
    x_max, y_max = np.max(blend_img_points_xy, axis=0).astype(int)
    
    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    img1_transformed_and_stitched = cv2.warpPerspective(img1, np.dot(H_translate, H), (x_max - x_min, y_max - y_min))

    # overlap_area = cv2.polylines(image1,[np.int32(blend_img_points_xy)],True,255,3, cv2.LINE_AA) 
    # cv2.imwrite('overlap_stitch.png', image1)
    
    stitched_image = img1_transformed_and_stitched.copy()
    stitched_image[-y_min:-y_min+h2, -x_min: -x_min+w2] = img2

    indices = np.where(img2 == [0,0,0])
    y = indices[0] + -y_min 
    x = indices[1] + -x_min 

    stitched_image[y,x] = img1_transformed_and_stitched[y,x]
    cv2.imwrite(saveLocation + f'sample_stitch{i}.png', stitched_image)

    return stitched_image   

def stitchImages(images, cornerDetectionType, saveFolder):
    """
        Takes two images as input and performs panorama stitching on both the images.    
    """
    imgs = images.copy()
    img1 = images[0].copy()

    for idx in range(1, len(images)):

        img2 = images[idx].copy()
        img_pair = [img1, img2]
        """
        Corner Detection
        Save Corner detection output as corners.png
        """ 
        corner_images, corner_maps, corner_coords = getCorners(img_pair, cornerDetectionType)
        # displayImages(corner_images, saveFolder)


        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        if cornerDetectionType == 2:
            print("\nNon-maximum suppression is already performed in goodFeaturesToTrack(). So, skipping ANMS.")
        else:
            print("\nPerforming Adaptive Non-Maximum Suppression on the Harris corner detection outputs.")
            anms_images, corner_coords = ANMS(img_pair, corner_maps, N_best = 700)
            # displayImages(anms_images, saveFolder)

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        corner_coords1, corner_coords2 = corner_coords
        matched_coords = matchFeatures(img1, img2, corner_coords1, corner_coords2, patch_size=41, comparison_ratio=0.9)

        # to visualize the mapping of each matched features
        visualizeMatchedFeatures(img1, img2, matched_coords)

        """
        Refine: RANSAC, Estimate Homography
        """
        matched_coords, H = RANSAC(matched_coords, accuracy=0.9, threshold=5)

        # to visualize the mapping of each matched features
        visualizeMatchedFeatures(img1, img2, matched_coords)

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        stitched_image = warpAndBlendImages(img1, img2, H, idx, saveFolder)

        # width = img2.shape[1]
        # height = img2.shape[0]
        # dim = (width, height)
        # stitched_image = cv2.resize(stitched_image, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('mypano', stitched_image)
        # cv2.waitKey()
    return stitched_image


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagesFolder', default='Data/Train/Set1', help='The directory containing the input images. default: Data/Train/Set1')

    Args = Parser.parse_args()
    BasePath = './'
    ImagesFolder = Args.ImagesFolder
    CornerDetectionType = 2

    if 'Test' in ImagesFolder:
        SaveLocation = BasePath + 'Code/Results/TestSet' + ImagesFolder[-1] + '/'

    elif 'Train' in ImagesFolder:
        SaveLocation = BasePath + 'Code/Results/TrainSet' + ImagesFolder[-1] + '/'

    CornerDetectionType = 2

    if not os.path.exists(SaveLocation):
        os.mkdir(SaveLocation)

    """
    Read a set of images for Panorama stitching
    """
    print('\n')
    images_path = BasePath + ImagesFolder
    images = readImages(images_path)
    # displayImages(images, SaveFolder)
    N_images = len(images)
    print(f"\n{N_images} images found to be stitched.")

    # Starting stitching of input images
    input_array = images.copy()
    img1 = input_array[0]

    for img2 in input_array[1:]:
        image_array = [img1, img2]
        img1 = stitchImages(image_array, CornerDetectionType, SaveLocation)
    
    # while len(input_array) > 1:
    #     N_images = len(input_array)
    #     if N_images % 2 == 0:
    #         stitched_array = []
    #         for i in range(0, N_images, 2):
    #             image_array = [input_array[i], input_array[i+1]]
    #             stitched_array.append(stitchImages(image_array, CornerDetectionType, SaveLocation))
    #     else:
    #         stitched_array = []
    #         for i in range(0, N_images-1, 2):
    #             image_array = [input_array[i], input_array[i+1]]
    #             stitched_array.append(stitchImages(image_array, CornerDetectionType, SaveLocation))
    #         stitched_array.append(input_array[-1])
    #     input_array = stitched_array.copy()

    final_stitch = img1
    cv2.imshow('final_stitch', final_stitch)
    cv2.waitKey()

if __name__ == "__main__":
    main()
