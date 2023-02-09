import torch
import cv2
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
import numpy as np
from kornia.geometry.transform import HomographyWarper as HomographyWarper

# os.chdir("D:\Computer Vision\sparashar_p1\Phase2\Code")
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as T


def lossFn(warped_a, b):
    
    # print(warped_a.shape, b.shape)
    loss = (b - warped_a).float().mean()
    loss = torch.tensor(loss, requires_grad=True)
    # print(loss)
    return loss

def getLoss(pred, labels):
    criterion = nn.MSELoss()
    loss = criterion(pred, labels.view(-1, 8))
    return loss

def warpImage(I_A, H):
    # print('warpImage')
    warper = HomographyWarper(I_A.shape[2], I_A.shape[3])
    # I_A_warped = kornia.geometry.transform.warp_perspective(I_A, H, (128, 128))
    w = warper(torch.ones(1, 1, 420, 526), torch.eye(3).unsqueeze(0))
    # print(w.shape)
    
    return w
    # return I_A


def generateLossBatches(batch, H_batch):
    # print('generateLossBatches')
    p_ab, _, corners_a, imgA_paths = batch
    # print(p_ab.shape)
    batch_size = p_ab.shape[0]
    pa_warped_list = []
    
    for path, h, corner in zip(imgA_paths, H_batch, corners_a):
        
        y = int(corner[0][1])
        y1 = int(corner[1][1])

        x = int(corner[0][0])
        x1 = int(corner[2][0])

        ia = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ia = torch.from_numpy(ia).to(torch.float)
        ia = ia.unsqueeze(0).unsqueeze(0)
        # print(ia.shape)
    
        h=torch.FloatTensor(h).unsqueeze(0).to(device)

        
        ia_warped = warpImage(ia, h)
        # print(ia_warped.shape)
        pa_warped = ia_warped[:, :, y:y1, x:x1]
        # print(pa_warped.shape)
        pa_warped = pa_warped.to(device)
        pa_warped = (pa_warped-127.5)/127.5
        # print(pa_warped.shape)
        pa_warped = pa_warped.squeeze(0).squeeze(0)
        pa_warped_list.append(pa_warped)
    
    p_b = p_ab[:, 1, :, :]
    
    return torch.stack(pa_warped_list), p_b

def tensorDLT(C_A, h4pt):
    # print('tensorDLT')
    batch_size = C_A.shape[0]
    C_B = C_A + h4pt.view(batch_size, 4, 2) * 32.
    H_batch = []
    
    H = torch.tensor((batch_size, 3, 3)).to(device)
    for img in range(batch_size):
        c_a = C_A[img, :, :]
        c_b = C_B[img, :, :]
        A = []
        b = []
        for i in range(4):
            a = [ [0, 0, 0, -c_a[i, 0], -c_a[i, 1], -1, c_b[i, 1]*c_a[i, 0], c_b[i, 1]*c_a[i, 1]], 
                    [c_a[i, 0], c_a[i, 1], 1, 0, 0, 0, -c_b[i, 0]*c_a[i, 0], -c_b[i, 0]*c_a[i, 1]] ]
            rhs = [[-c_b[i, 1]], [c_b[i, 0]]]

            A.append(a)
            b.append(rhs)
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False).to(device).reshape(8, 8)
        b = torch.tensor(b, dtype=torch.float32, requires_grad=False).to(device).reshape(8, 1)
        x = torch.linalg.solve(A, b)
        # x = torch.tensor(x, requires_grad=False)
        h = torch.cat((x, torch.ones(1,1).to(device)), 0).to(device)
        H = h.view(3,3)
        H_batch.append(H)
    
    return torch.stack(H_batch)



class ModelBase(nn.Module):

    def training_step(self, batch):
        # print('model.training_step')
        p_ab, delta, c_a, imgA_paths = batch
        h4pt = self(p_ab)
        H_batch = tensorDLT(c_a, h4pt)
        p_a_warped, p_b = generateLossBatches(batch, H_batch)
        # print(p_a_warped.shape, p_b.shape)
        loss = lossFn(p_a_warped, p_b)
        return loss
    
    def validation_step(self, batch):
        p_ab, delta, c_a, imgA_paths = batch
        h4pt = self(p_ab)
        H_batch = tensorDLT(c_a, h4pt)
        p_a_warped, p_b = generateLossBatches(batch, H_batch)
        loss = lossFn(p_a_warped, p_b)
        return {'val_loss': loss.detach()}
    
    def validation_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))

class HNet(ModelBase):
    def __init__(self):
        super(HNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())        
        self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.fc1 = nn.Linear(128*16*16,1024)
        self.fc2 = nn.Linear(1024,8)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(-1,128* 16* 16)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = HNet().to(device)
summary(model, (2, 128, 128))