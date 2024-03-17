import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from utils.get_metric import _mse, _ssim, _siam, _mind, _ncc
from models.model import Autoencoder, SiameseNetwork_AE, Autoencoder_Simon

def evaluate_sections(model, GT_sections, device, csv_filename):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])
    
    data_dict = {'Section': []}
    img_key_list = ['GT', 'Deformed', 'ANTS', 'VXM_MSE1', 'VXM_NCC', 'VXM_Simonovosky', 'VXM_DeepSIM', 'VXM_SIAM', 'VXM_SIAM_PRO']
    metric_list = ['MSE', 'SSIM', 'SIAM']
    for img_key in img_key_list:
        for metric in metric_list:
            data_dict[f'{img_key}_{metric}'] = []
    # print(data_dict)

    for file in tqdm(GT_sections):    
        ref = np.load(file)
        deformed_img = np.load(file.replace('ground_truth_sections', 'distorted_sections'))
        ants_img = np.load(file.replace('ground_truth_sections', 'registered_sections'))
        vxmmse_img = np.load(file.replace('ground_truth_sections', 'VXM_MSE1_ants_registered_sections'))
        vxmncc_img = np.load(file.replace('ground_truth_sections', 'VXM_NCC_ants_registered_sections'))
        vxmsimonovosky_img = np.load(file.replace('ground_truth_sections', 'VXM_Simonovsky_ants_registered_sections'))
        vxmdeepsim_img = np.load(file.replace('ground_truth_sections', 'VXM_DeepSim_ants_registered_sections'))
        vxmsiam_img = np.load(file.replace('ground_truth_sections', 'VXM_SiamRegQC_ants_registered_sections'))
        vxmsiampro_img = np.load(file.replace('ground_truth_sections', 'VXM_SiamRegQC_pro_registered_sections'))

        data_dict['Section'] = os.path.basename(file)[:-4]
        for img_key, img in zip(img_key_list, 
                            [ref, deformed_img, ants_img, vxmmse_img, vxmncc_img, vxmsimonovosky_img, vxmdeepsim_img, vxmsiam_img, vxmsiampro_img]):

            for metric in metric_list:
                if metric =='MSE':
                    data_dict[f'{img_key}_{metric}'] += [_mse(img, ref)]
                elif metric == 'SSIM':
                    data_dict[f'{img_key}_{metric}'] += [_ssim(img, ref)]
                else:
                    ref_tensor = transform(ref)
                    deformed_tensor = transform(img)
                    out = _siam(model, ref_tensor.unsqueeze(0), deformed_tensor.unsqueeze(0), device)
                    data_dict[f'{img_key}_{metric}'] += [out[1]]
    
    data_df = pd.DataFrame(data_dict)
    print(f'Saving to {csv_filename}')
    data_df.to_csv(csv_filename, index=False)

def evaluate_sections_simon(model, GT_sections, device, csv_filename):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])

    data_dict = {'Section': []}
    img_key_list = ['GT', 'Deformed', 'ANTS', 'VXM_MSE1', 'VXM_NCC', 'VXM_Simonovosky', 'VXM_DeepSIM', 'VXM_SIAM', 'VXM_SIAM_PRO']
    metric_list = ['MSE', 'SSIM', 'SIAM']
    for img_key in img_key_list:
        for metric in metric_list:
            data_dict[f'{img_key}_{metric}'] = []
    # print(data_dict)
    
    for file in tqdm(GT_sections):    
        ref = np.load(file)
        deformed_img = np.load(file.replace('ground_truth_sections', 'distorted_sections'))
        ants_img = np.load(file.replace('ground_truth_sections', 'registered_sections'))
        vxmmse_img = np.load(file.replace('ground_truth_sections', 'VXM_MSE1_ants_registered_sections'))
        vxmncc_img = np.load(file.replace('ground_truth_sections', 'VXM_NCC_ants_registered_sections'))
        vxmsimonovosky_img = np.load(file.replace('ground_truth_sections', 'VXM_Simonovsky_ants_registered_sections'))
        vxmdeepsim_img = np.load(file.replace('ground_truth_sections', 'VXM_DeepSim_ants_registered_sections'))
        vxmsiam_img = np.load(file.replace('ground_truth_sections', 'VXM_SiamRegQC_ants_registered_sections'))
        vxmsiampro_img = np.load(file.replace('ground_truth_sections', 'VXM_SiamRegQC_pro_registered_sections'))

        data_dict['Section'] = os.path.basename(file)[:-4]
        for img_key, img in zip(img_key_list, 
                            [ref, deformed_img, ants_img, vxmmse_img, vxmncc_img, vxmsimonovosky_img, vxmdeepsim_img, vxmsiam_img, vxmsiampro_img]):
            for metric in metric_list:
                if metric =='MSE':
                    data_dict[f'{img_key}_{metric}'] += [_mse(img, ref)]
                elif metric == 'SSIM':
                    data_dict[f'{img_key}_{metric}'] += [_ssim(img, ref)]
                else:
                    ref_tensor = transform(ref).unsqueeze(0).type(torch.FloatTensor).to(device)
                    deformed_tensor = transform(img).unsqueeze(0).type(torch.FloatTensor).to(device)
                    out = model(deformed_tensor, ref_tensor)
                    data_dict[f'{img_key}_{metric}'] += [out[0,0].item()]
    
    data_df = pd.DataFrame(data_dict)
    print(f'Saving to {csv_filename}')
    data_df.to_csv(csv_filename, index=False)

def evaluate_sections_multimodal(GT_sections, csv_filename):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])
    
    data_dict = {'Section': []}
    # img_key_list = ['GT', 'Deformed', 'ANTS', 'VXM_MSE1', 'VXM_NCC', 
    #     'VXM_Simonovosky', 'VXM_DeepSIM', 'VXM_SIAM', 'VXM_SIAM_PRO']
    img_key_list = ['GT', 'Deformed', 'VXM_NCC', 'VXM_Simonovosky', 'VXM_DeepSim', 'VXM_SiamRegQC', 'ANTS']
    metric_list = ['MSE', 'SSIM', 'NCC']
    for img_key in img_key_list:
        for metric in metric_list:
            data_dict[f'{img_key}_{metric}'] = []
    # print(data_dict)

    for file in tqdm(GT_sections):    
        # ref = np.load(file.replace('ground_truth_sections', 'ground_truth_sections_T2').replace('T1', 'T2'))
        ref = np.load(file.replace('ground_truth_sections', 'ground_truth_sections_T2').replace('T1', 'T2'))
        deformed_img = np.load(file.replace('ground_truth_sections', 'distorted_sections'))
        ants_img = np.load(file.replace('ground_truth_sections', 'multimodal_registered_sections_T1'))
        # vxmmse_img = np.load(file.replace('ground_truth_sections', 'VXM_MSE1_ants_registered_sections'))
        vxmncc_img = np.load(file.replace('ground_truth_sections', 'VXM_multimodal_NCC_registered_sections_T1'))
        vxmsimonovosky_img = np.load(file.replace('ground_truth_sections', 'VXM_multimodal_Simonovosky_registered_sections_T1'))
        vxmdeepsim_img = np.load(file.replace('ground_truth_sections', 'VXM_multimodal_DeepSim_registered_sections_T1'))
        # vxmsiam_img = np.load(file.replace('ground_truth_sections', 'VXM_SiamRegQC_ants_registered_sections'))
        vxmsiampro_img = np.load(file.replace('ground_truth_sections', 'VXM_multimodal_SiamRegQC_registered_sections_T1'))

        # image_list = [ref, deformed_img, ants_img, vxmmse_img, vxmncc_img, 
        #     vxmsimonovosky_img, vxmdeepsim_img, vxmsiam_img, vxmsiampro_img]
        image_list = [ref, deformed_img, vxmncc_img, vxmsimonovosky_img, vxmdeepsim_img, vxmsiampro_img, ants_img]
        data_dict['Section'] += [os.path.basename(file)[:-4]]
        # print(os.path.basename(file)[:-4])
        for img_key, img in zip(img_key_list, image_list):

            for metric in metric_list:
                if metric =='MSE':
                    data_dict[f'{img_key}_{metric}'] += [_mse(img, ref)]
                elif metric == 'SSIM':
                    data_dict[f'{img_key}_{metric}'] += [_ssim(img, ref)]
                # elif metric == 'MIND':
                #     data_dict[f'{img_key}_{metric}'] += [_mind(img, ref)]
                else:
                    # data_dict[f'{img_key}_{metric}'] += [_mind(img, ref)]
                    data_dict[f'{img_key}_{metric}'] += [_ncc(img, ref)]
    
    data_df = pd.DataFrame(data_dict)
    print(len(data_df['Section'].unique()))
    print(f'Saving to {csv_filename}')
    data_df.to_csv(csv_filename, index=False)

def main():

    model_file = '../model_weights/MRI_GuysT1__20240309-215304_BS32_EP10.pt'
    # model_file = '../model_weights/MRI_GuysT1_NL_20240309-223041_BS32_EP10.pt'
    # model_file = '../model_weights/MRI_GuysT1_Simon_NPNL_20240310-132842_BS32_EP10.pt'
    GT_sections = glob.glob('../ground_truth_sections/*')
    which_model = 'no'
    pref1 = os.path.basename(model_file)[:-3]
    pref2 = os.path.basename(GT_sections[0])[:-4]
    csv_filename = '../metric_csv_files/' + pref1 + pref2 + 'pro.csv'

    device = 'cuda:0'

    # Instantiate the model
    if which_model == 'simon':
        print('-------------Load Model---------------')
        print(model_file)
        ae_simonovosky = Autoencoder_Simon().to(device)
        ae_simonovosky.load_state_dict(torch.load(model_file))
        evaluate_sections_simon(ae_simonovosky, GT_sections, device, csv_filename)
    elif which_model == 'SiamRegQC':
        print('-------------Load Model---------------')
        print(model_file)
        autoencoder = Autoencoder()
        autoencoder = autoencoder.to(device)
        ae_siam = SiameseNetwork_AE(autoencoder.encoder).to(device)
        ae_siam.load_state_dict(torch.load(model_file))
    else:
        print('----------Evaluating Multimodal-----------')
        csv_filename = '../metric_csv_files/' + pref2 + '_multimodal_metrics.csv'
        evaluate_sections_multimodal(GT_sections, csv_filename)

    
   
if __name__ == '__main__':
    main()

            
