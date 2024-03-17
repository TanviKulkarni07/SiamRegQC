import numpy as np
import os
from torchvision import transforms
import torch
import torch.nn.functional as nnf
from utils.get_metric import _mse, _ssim, _siam, plot_loss_curve
from utils.common_utils import read_volseg, intensity_normalize
from utils.distortions import random_elastic_transform
from models.model import Autoencoder, SiameseNetwork_AE

def get_losses(model, ref, img, device, jitters_alpha, jitters_sigma):

    sims_in = []
    sims_out = []
    true_mse = []
    true_ssim = []
    print('---------Running Jitter Experiment---------------')
    for alpha in jitters_alpha:
        for sigma in jitters_sigma:

            grid = random_elastic_transform(img.shape[1:], 3, sigma, alpha)    
            deformed_img = nnf.grid_sample(img.unsqueeze(0).type(torch.FloatTensor), 
                        grid.unsqueeze(0).type(torch.FloatTensor), 
                        align_corners=True, mode='bilinear')
            out = _siam(model, ref.unsqueeze(0), deformed_img, device)

            sims_in.append(out[2])
            sims_out.append(out[1])
            true_mse.append(_mse(deformed_img.numpy()[0, 0], ref.numpy()[0]))
            true_ssim.append(_ssim(deformed_img.numpy()[0, 0], ref.numpy()[0]))
    return true_mse, true_ssim, sims_out

def main():
    run_main = True
    model_file = '../model_weights/MRI_GuysT1__20240309-215304_BS32_EP10.pt'
    filename = os.path.basename(model_file)

    jitters_alpha = np.arange(0.6, 1.6, 0.05)
    jitters_sigma = np.arange(20, 40, 0.5)
    X, Y = np.meshgrid(jitters_sigma, jitters_alpha)

    if run_main:
        print('----------Saving Loss Curves-------------')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.CenterCrop(256),
                                    ])

        vol,_ = read_volseg('../reg_n4/IXI090-Guys-0800-T1.nii.gz')
        vol = intensity_normalize(vol)
        idx = vol.shape[1]//2
        ref = transform(vol[:,idx,:])
        img = transform(vol[:,idx,:])

        device = 'cuda:0'
        # Instantiate the model
        autoencoder = Autoencoder()
        autoencoder = autoencoder.to(device)
        model = SiameseNetwork_AE(autoencoder.encoder).to(device)
        model.load_state_dict(torch.load(model_file))

        true_mse, true_ssim, sims_out = get_losses(model, ref, img, 
                            device, jitters_alpha, jitters_sigma)
        
        Z = np.array(sims_out).reshape(len(jitters_alpha), len(jitters_sigma))
        W = np.array(true_mse).reshape(len(jitters_alpha), len(jitters_sigma))
        U = np.array(true_ssim).reshape(len(jitters_alpha), len(jitters_sigma))

        np.save('metrics_outputs/' + filename.replace('.pt', '_SIAM.npy'), Z)
        np.save('metrics_outputs/' + filename.replace('.pt', '_SSIM.npy'), U)
        np.save('metrics_outputs/' + filename.replace('.pt', '_MSE.npy'), W)

    else:
        print('----------Reading Existing Loss Curves----------------')
        Z = np.load('metrics_outputs/' + filename.replace('.pt', '_SIAM.npy'))
        U = np.load('metrics_outputs/' + filename.replace('.pt', '_SSIM.npy'))
        W = np.load('metrics_outputs/' + filename.replace('.pt', '_MSE.npy'))

    plot_loss_curve(X, Y, Z, filename= 'loss_curves/' + filename.replace('.pt', '_SIAM.png'))
    plot_loss_curve(X, Y, U, filename= 'loss_curves/' + filename.replace('.pt', '_SSIM.png'))
    plot_loss_curve(X, Y, W, filename= 'loss_curves/' + filename.replace('.pt', '_MSE.png'))

if __name__ == '__main__':
    main()