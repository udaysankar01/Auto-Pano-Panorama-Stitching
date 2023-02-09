import torch
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
from Model_supervised import HNet
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getImageNames(path):
    
    image_names = []
    for file in os.listdir(path):
        image_names.append(os.path.splitext(file)[0])
    
    return image_names

def generateBatch(path_A, path_B, coordinates_path, batch_size=64):
    
    img_batch = []
    labels_batch = []
    image_num = 0
    coordinates = np.load(coordinates_path)
    
    while image_num < batch_size:
        
        # Random Image Path
        image_names = getImageNames(path_A)               # Names are the same for both patch A and patch B
        RandIdx = random.randint(0, len(image_names)-1)
        image_pathA = path_A + '/' + image_names[RandIdx] + '.jpg'
        image_pathB = path_B + '/' + image_names[RandIdx] + '.jpg'
        
        image_num += 1

        # Read Data
        imgA = cv2.imread(image_pathA, cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imread(image_pathB, cv2.IMREAD_GRAYSCALE)
        
        label = coordinates[RandIdx]
        
        # Normalize Data and convert to torch tensors
        imgA = torch.from_numpy((imgA.astype(np.float32) - 127.5) / 127.5)
        imgB = torch.from_numpy((imgB.astype(np.float32) - 127.5) / 127.5)
        
        label = torch.from_numpy(label.astype(np.float32)/32.0)
        
        # Stack grayscale images
        img = torch.stack((imgA, imgB), dim=0)
        
        # Add to batch
        img_batch.append(img.to(device))
        labels_batch.append(label.to(device))
           
    return torch.stack(img_batch), torch.stack(labels_batch)

def prettyPrint(NumEpochs, MiniBatchSize):
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Mini Batch Size " + str(MiniBatchSize))

def train(path_ATrain, path_BTrain,
        path_AVal, path_BVal,
        coordinates_path,
        batch_size, 
        num_epochs, 
        CheckPointPath):
    
    torch.cuda.empty_cache()
    history = []

    #Model 
    model = HNet().to(device)
    
    #Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    num_samples_train = len(getImageNames(path_ATrain))
    num_samples_val = len(getImageNames(path_BTrain))
    
    num_iter_per_epoch = num_samples_train//batch_size
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(num_epochs)):
        
        for iter_counter in tqdm(range(num_iter_per_epoch)):
            
            train_batch = generateBatch(path_ATrain, path_BTrain, coordinates_path, batch_size)
            
            # Train
            model.train()
            optimizer.zero_grad()
            batch_loss_train = model.training_step(train_batch)
            train_losses.append(batch_loss_train)
            batch_loss_train.backward()
            optimizer.step()
        
        # Save model every epoch
        SaveName = CheckPointPath + str(epoch) + "_model.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": batch_loss_train,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


        # Evaluate
        num_iter_per_epoch_val = num_samples_val//batch_size
        model.eval()
        with torch.no_grad():
            for iter_count_val in tqdm(range(num_iter_per_epoch_val)):
                val_batch = generateBatch(path_AVal, path_BVal, coordinates_path, batch_size)
                val_losses.append(model.validation_step(val_batch))
                
        result = model.validation_end(val_losses)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history
    
def plotLosses(history):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.figure()
    plt.plot(train_losses, '-b')
    plt.plot(val_losses, '-r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. Epoch')
    plt.savefig('LossCurve.png')
    
    

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:../Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:64",
    )

    Args = Parser.parse_args()
    num_epochs = Args.NumEpochs
    BasePath = Args.BasePath
    batch_size = Args.MiniBatchSize
    CheckPointPath = Args.CheckPointPath

    path_ATrain = BasePath + "/modified_train/patchA"
    path_BTrain = BasePath + "/modified_train/patchB"
    path_AVal = BasePath + "/modified_val/patchA"
    path_BVal = BasePath + "/modified_val/patchB"
    coordinates_path = BasePath + '/modified_train_labels.npy' 

    prettyPrint(num_epochs, batch_size)

    history = []

    history += train(path_ATrain, path_BTrain,
            path_AVal, path_BVal,
            coordinates_path,
            batch_size, 
            num_epochs,
            CheckPointPath)

    np.save('history.npy', np.array(history))
    plotLosses(history)
    

    



if __name__ == "__main__":
    main()