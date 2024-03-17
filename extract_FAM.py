import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from torchvision import transforms
from models.model import Autoencoder, SiameseNetwork_AE
from utils.fam_utils import *

def main():
    
    # model_file = '../model_weights/MRI_GuysT1__20240309-215304_BS32_EP10.pt'
    model_file = '../model_weights/MRI_T2_AE.pt'
    is_siamese = False
    device = 'cuda:0'
    # Instantiate the model
    
    print('-------------Load Model---------------')
    print(model_file)

    ref_file = '../ground_truth_sections_T2/IXI030-Guys-0708-T2086.npy'

    pref1 = os.path.basename(model_file)[:-3]
    pref2 = os.path.basename(ref_file)[:-4]
    fam_filename = '../FAMS/' + pref1 + pref2 + 'FAM.png'

    ref = np.load(ref_file)
    img = np.load(ref_file.replace('ground_truth_sections', 'distorted_sections'))

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                            ])
    if is_siamese:
        autoencoder = Autoencoder()
        autoencoder = autoencoder.to(device)
        model = SiameseNetwork_AE(autoencoder.encoder).to(device)
        model.load_state_dict(torch.load(model_file))

        ref_tensor = transform(ref).unsqueeze(0)
        img_tensor = transform(img).unsqueeze(0)
        
        print('-----------Running Model--------------')
        feature_maps1_dis, feature_maps2_dis = get_activations(model, ref_tensor, img_tensor, device)
        plt.figure(figsize=(15, 10))
        for i in range(32):
            plt.subplot(8, 4, i + 1)
            plt.imshow(np.abs(feature_maps2_dis[i]-feature_maps1_dis[i]), cmap='rainbow')
            plt.axis('off')
            plt.title(f'{i + 1}')
        plt.tight_layout()
        plt.savefig(fam_filename)

        plt.figure(figsize=(15, 10))
        plt.imshow(np.sqrt((ref-img)**2), cmap='rainbow')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fam_filename.replace('FAM.png', 'MSE.png'))
    
    else:
        autoencoder = Autoencoder()
        autoencoder = autoencoder.to(device)
        autoencoder.load_state_dict(torch.load(model_file))

        autoencoder_T1 = Autoencoder()
        autoencoder_T1 = autoencoder_T1.to(device)
        autoencoder_T1.load_state_dict(torch.load('../model_weights/MRI_AE.pt'))

        ref_tensor = transform(ref).unsqueeze(0)
        
        print('-----------Running Model--------------')
        feature_maps1_dis = get_activations_encoder(autoencoder.encoder, ref_tensor, device)
        plt.figure(figsize=(15, 10))
        for i in range(32):
            plt.subplot(8, 4, i + 1)
            plt.imshow(feature_maps1_dis[i], cmap='rainbow')
            plt.axis('off')
            plt.title(f'{i + 1}')
        plt.tight_layout()
        plt.savefig(fam_filename)

        plt.figure(figsize=(15, 10))
        plt.imshow(ref, cmap='rainbow')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fam_filename.replace('FAM.png', 'MSE.png'))
    

if __name__ == '__main__':
    main()