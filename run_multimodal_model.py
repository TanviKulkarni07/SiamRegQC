import glob
import random
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

from utils.common_utils import read_volseg, visualise_AE_ipop, intensity_normalize
from create_dataset import create_multimodal_data
from models.model import Autoencoder, MultimodalSiameseNetwork_AE, Autoencoder_Simon
from models.train_cv import train_cv
from config.config import *
from infer import infer_test


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])

GT_files = list(glob.glob('../reg_n4/*Guys*T1*'))
train_files = random.sample(GT_files,10)
test_files = random.sample(GT_files,5)
test_files = [x for x in test_files if x not in train_files]
print('-----------Create Dataset------------')
data, train_loader =create_multimodal_data(train_files, transform)

print('Num Train Sections:', len(data))
print('Num Test Vols:', len(test_files))
    
print('-------------Instantiate Model---------------')
device = 'cuda:0'

# Instantiate the model
# print('------------Loading Encoder T1-----------------')
autoencoder_T1 = Autoencoder()
autoencoder_T1 = autoencoder_T1.to(device)
autoencoder_T1.load_state_dict(torch.load('../model_weights/MRI_AE.pt'))

# print('------------Loading Encoder T2-----------------')
autoencoder_T2 = Autoencoder()
autoencoder_T2 = autoencoder_T1
autoencoder_T2.load_state_dict(torch.load('../model_weights/MRI_T2_AE.pt'))

ae_siam = MultimodalSiameseNetwork_AE(autoencoder_T1, autoencoder_T2).to(device)
# ae_siam.load_state_dict(torch.load('../model_weights/MRI_Guys_multimodal__20240312-210249_BS32_EP10.pt'))

# ae_siam = Autoencoder_Simon().to(device)

optimizer = torch.optim.Adam(ae_siam.parameters(), lr=learning_rate)

print('------------Training Begins!------------')
train_cv(ae_siam, data, optimizer, device, add_cc_loss, num_folds, num_epochs, batch_size, model_filename)

print('------------Inference Begins!----------------')
infer_test(test_files, ae_siam, device)

