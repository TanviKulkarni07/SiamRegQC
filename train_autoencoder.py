import glob
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.common_utils import read_volseg, visualise_AE_ipop, intensity_normalize
from utils.dataloader import AEDataset_Elastic
from create_dataset import create_multimodal_data

import torchvision.utils as vutils
from models.model import Autoencoder
GT_files = list(glob.glob('../reg_n4/*Guys*T1*'))
train_files = random.sample(GT_files,10)
test_files = random.sample(GT_files,5)
test_files = [x for x in test_files if x not in train_files]
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.CenterCrop(256),
                            ])
print('-----------Create Dataset------------')
data, train_loader =create_multimodal_data(train_files, transform)
dataloader = DataLoader(data, batch_size=5, shuffle=True)
input1, input2, label = next(iter(dataloader))
visualise_AE_ipop(input2, 5)
device = 'cuda:0'
autoencoder = Autoencoder()
autoencoder = autoencoder.to(device)
# List that will store the training loss 
train_loss = [] 
dataloader = DataLoader(data, batch_size=64, shuffle=True)
# Dictionary that will store the 
# different images and outputs for 
# various epochs 
outputs = {} 

# batch_size = len(dataloader) 
criterion = torch.nn.MSELoss() 
num_epochs = 200
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001) 
n = 90

# Training loop starts 
for epoch in tqdm(range(num_epochs)):
    
    running_loss = 0

    for _,x,_ in dataloader:
        
        img = x.type(torch.FloatTensor).to(device)
        # Generating output 
        out = autoencoder(img) 

        # Calculating loss 
        loss = criterion(out, img) 

        # Updating weights according 
        # to the calculated loss 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        # Incrementing loss 
        running_loss += loss.item() 

    # Averaging out loss over entire batch 
    running_loss /= len(dataloader)
    train_loss.append(running_loss) 

    # Storing useful images and 
    # reconstructed outputs for the last batch 
    outputs[epoch+1] = {'img': img, 'out': out} 
    
    if epoch%5 == 0:
        plt.figure()
        plt.subplot(121).imshow(np.transpose(vutils.make_grid(x.to(device)[:5], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.subplot(122).imshow(np.transpose(vutils.make_grid(out[:5], padding=2, normalize=True).cpu(),(1,2,0)))

        if best_loss>loss.item():
            best_loss = loss.item()
            print('-----Saving Model---------')
            print('Best Loss Achieved!:', best_loss)
            torch.save(autoencoder.state_dict(), '../model_weights/MRI_T2_AE.pt') 


# Plotting the training loss 
plt.plot(range(1,num_epochs+1),train_loss) 
plt.xlabel("Number of epochs") 
plt.ylabel("Training Loss") 
plt.show()