from utils.common_utils import read_volseg, visualise_AE_ipop, intensity_normalize
from utils.dataloader import AEDataset_Elastic
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model import Autoencoder, SiameseNetwork_AE
from models.train_cv import test_epoch, test_epoch_multimodal
from config.config import *
import torch
import glob
import random

def infer_test(test_files, model, device):
    vol,_ = read_volseg(test_files[0])
    vol = intensity_normalize(vol)

    axis_to_extract = 1
    start_slice = 40
    end_slice = -40

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop(256),
                                ])

    true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
                            axis=axis_to_extract,is_true=True)
    distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
                                axis=axis_to_extract,is_true=False)

    test_data = true_dataset + distorted_dataset

    for file in test_files:
        vol, _ = read_volseg(file)
        vol = intensity_normalize(vol)    

        true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
                                axis=axis_to_extract, is_true=True)
        distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
                                    axis=axis_to_extract,is_true=False)
        
        test_data += true_dataset + distorted_dataset

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    if is_multimodal:
        loss, acc = test_epoch_multimodal(model, test_dataloader, device, add_cc_loss)
    else:
        loss, acc = test_epoch(model, test_dataloader, device, add_cc_loss)
    print('Loss:', loss, 'Accuracy:', acc)

def main():

    GT_files = list(glob.glob('../reg_n4/*Guys*T1*'))
    model_file = '../model_weights/MRI_GuysT1_NL_20240309-223041_BS32_EP10.pt'
    test_files = random.sample(GT_files,5)

    print('-------------Load Model---------------')
    print(model_file)
    device = 'cuda:0'

    # Instantiate the model
    autoencoder = Autoencoder()
    autoencoder = autoencoder.to(device)

    ae_siam = SiameseNetwork_AE(autoencoder.encoder).to(device)
    ae_siam.load_state_dict(torch.load(model_file))

    infer_test(test_files, ae_siam, device)

if __name__ == '__main__':
    main()