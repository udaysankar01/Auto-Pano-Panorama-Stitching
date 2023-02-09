import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio

from PIL import Image



def generate_data(i,j,img,homography_list,corner_pointA):
#print(np.shape(img))
# Get image shape
    h, w, c = img.shape

    # Select random patch size
    patch_size = 128

    # Select random top-left corner coordinates
    x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)







    x1=(x + patch_size)
    y1=(y + patch_size)





    z = 32

    corner_points = np.array([[x,y], [x, y1] , [x1, y], [x1,y1]]) # coordinates of patch P_a
    corner_pointA.append(corner_points)

    perturbation = np.random.randint(-z, z+1, (4, 2))

    # Add perturbation to corner points
    perturbed_corner_points = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(corner_points, perturbation)]


    # find H inverse of usin patch coordinates of P_a, P_b
    H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(corner_points), np.float32(perturbed_corner_points))) 
            
    img1 = cv2.warpPerspective(img, H, (w,h))

    patch1 = img[y:y1, x:x1]
    patch2= img1[y:y1, x:x1]
    H4pt = (perturbed_corner_points - corner_points).astype(np.float32) 
    homography_list.append(H4pt)




    #data_img = np.concatenate((patch1, patch2), axis=-1)
    
    #np.savetxt('../Data/modified_train/{}_{}.jpg'.format(i,j+1), data_img, delimiter=',', fmt='%d')
    cv2.imwrite('../Data/modified_train/patchA/{}_{}.jpg'.format(i,j+1), patch1)
    cv2.imwrite('../Data/modified_train/patchB/{}_{}.jpg'.format(i,j+1), patch2)
    return homography_list,corner_pointA



def main():
    homography_list = []
    corner_pointA=[]
    for i in range(1,5001):
        img=cv2.imread(r"../Data/Train/{}.jpg".format(i))
        for j in range(3):
            homography_list,corner_pointA = generate_data(i,j,img,homography_list,corner_pointA)



    homography = np.array(homography_list)

# save the numpy array to a .npy file
    np.save('./TxtFiles/modified_train_labels.npy', homography)

    corner_pointA=np.array(corner_pointA)
    np.save('./TxtFiles/train_cornerpoints.npy', corner_pointA)
    #np.savetxt('./TxtFiles/modified_train_labels.', homography_list, delimiter='\n')

main()






