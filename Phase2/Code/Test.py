import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
import argparse
import random
from tqdm.notebook import tqdm
# os.chdir("D:\Computer Vision\sparashar_p1\Phase2\Code")
#from Model_supervised import HNet
#from Model_unsupervised import HNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load the trained model checkpoint
model = HNet()
Model_path='./'

#checkpoint = torch.load('../CheckPoints/48_model.ckpt')
#checkpoint = torch.load('../CheckPoints/11_model.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_image(images):
    images1 = cv2.resize(images[0], (128, 128), interpolation = cv2.INTER_AREA)
    images1=images1.astype(np.float32)
    images1 = torch.from_numpy(cv2.cvtColor(images1, cv2.COLOR_BGR2GRAY))
    

    images2 = cv2.resize(images[1], (128, 128), interpolation = cv2.INTER_AREA)
    images2=images2.astype(np.float32)
    images2 = torch.from_numpy(cv2.cvtColor(images2, cv2.COLOR_BGR2GRAY))

    #images2=cv2.resize(images[1],(64,64))
    img = torch.stack((images1, images2), dim=0)

    return img



def stitch_images(images, homography):
  
    height, width, channels = images[0].shape
    img2_warped = cv2.warpPerspective(images[1], homography, (width, height))

    return img2_warped

def predict_homographies(images):
    homographies = []

    img = preprocess_image(images)
    #print(np.shape(img))
    image = (img).float().unsqueeze(0)
        
       
    with torch.no_grad():
            output = model(image)
            writer=SummaryWriter(Model_path)
            writer.add_graph(model,image.to(device))
            writer.close
        
    # The output should be the predicted homography matrix
    homography = output.squeeze().numpy()
    #homographies.append(homography)
    return homography





image = cv2.imread(f'../Data/Train/1.jpg')


h, w, c = image.shape

    # Select random patch size
patch_size = 128

z=32

center_x = w // 2
center_y = h // 2

# Cut the 128x128 ROI from the center of the image
patch1 = image[center_y-64:center_y+64, center_x-64:center_x+64]
corner_points1=[[center_x-64,center_y-64], [center_x+64,center_y-64] , [center_x-64,center_y+64], [center_x+64,center_y+64]]



perturbation = np.random.randint(-z, z+1, (4, 2))

# Add perturbation to corner points
perturbed_corner_points = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(corner_points1, perturbation)]

H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(corner_points1), np.float32(perturbed_corner_points))) 
            
img1 = cv2.warpPerspective(image, H, (w,h))
cv2.imwrite('warped_ground.jpg', img1)



patch2= img1[center_y-64:center_y+64, center_x-64:center_x+64]

patch_images=[]
patch_images.append(patch1)
patch_images.append(patch2)
cv2.imwrite('patch1.jpg', patch1)
cv2.imwrite('patch2.jpg', patch2)
homography = predict_homographies(patch_images)

#print(homography)




cb=np.array([[homography[0]+center_x-64,homography[1]+center_y-64], [homography[2]+center_x+64, homography[3]+center_y-64] , [homography[4]+center_x-64, homography[5]+center_y+64], [homography[6]+center_x+64,homography[7]+center_y+64]]) 
ca=np.array([[center_x-64,center_y-64], [center_x+64,center_y-64] , [center_x-64,center_y+64], [center_x+64,center_y+64]])  

H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(ca),np.float32(cb)))
print(H)            


height, width, channels = image.shape
img2_warped = cv2.warpPerspective(image, H, (width, height))
cv2.imwrite('warped_supervised.jpg', img2_warped)





