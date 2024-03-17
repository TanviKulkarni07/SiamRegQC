import numpy as np
import random
import torch
from torch.utils.data import Dataset
from utils.distortions import distort_image_2class, distort_elastic, distort_affine
from utils.common_utils import read_volseg, intensity_normalize

class AEDataset(Dataset):
    def __init__(self, volume_data, transform, axis=0, is_true=True, is_con=True):
        self.volume_data = volume_data
        self.axis = axis
        self.is_true = is_true
        self.transforms = transform
        self.is_con = is_con
        
    def __len__(self):
        return self.volume_data.shape[self.axis]

    def __getitem__(self, idx):
        n = self.__len__()//2
        if idx < n:
            ref_idx = idx+1
        else:
            ref_idx = idx-1
            
        if self.axis == 0:
            input1 = self.volume_data[idx, :, :]
            input2 = self.volume_data[ref_idx, :, :]
            
        elif self.axis == 1:
            input1 = self.volume_data[:, idx, :]
            input2 = self.volume_data[:, ref_idx, :]
            
        elif self.axis == 2:
            input1 = self.volume_data[:, :, idx]
            input2 = self.volume_data[:, :, ref_idx]
        else:
            raise ValueError("Invalid axis. Please use 0, 1, or 2.")
        
        input1 = self.transforms(input1).numpy()[0]
        input2 = self.transforms(input2).numpy()[0]
        
        if self.is_true:
            label = 1
        else:
            input1, label = distort_image_2class(input1)

        return input1[np.newaxis,...], input2[np.newaxis,...], label, idx, ref_idx
    
class AEDataset_Elastic(Dataset):
    def __init__(self, volume_data, transform, axis=0, is_true=True):
        self.volume_data = volume_data
        self.axis = axis
        self.is_true = is_true
        self.transforms = transform
        
    def __len__(self):
        return self.volume_data.shape[self.axis]

    def __getitem__(self, idx):
        n = self.__len__()//2
        if idx < n:
            ref_idx = idx+1
        else:
            ref_idx = idx-1
            
        if self.axis == 0:
            input1 = self.volume_data[idx, :, :]
            input2 = self.volume_data[ref_idx, :, :]
            
        elif self.axis == 1:
            input1 = self.volume_data[:, idx, :]
            input2 = self.volume_data[:, ref_idx, :]
            
        elif self.axis == 2:
            input1 = self.volume_data[:, :, idx]
            input2 = self.volume_data[:, :, ref_idx]
        else:
            raise ValueError("Invalid axis. Please use 0, 1, or 2.")
        
        input1 = self.transforms(input1)
        input2 = self.transforms(input2)
        
        if self.is_true:
            label = 1
        else:
            input1 = distort_elastic(input1)
            label = 0
        return input1, input2, label
    

class Multimodal_AEDataset(Dataset):
    def __init__(self, vol_file, transform, axis=0, is_true=True):
        
        self.vol_file = vol_file
        self.volume_data_T1, _ = read_volseg(self.vol_file)
        self.volume_data_T1 = intensity_normalize(self.volume_data_T1)
        self.volume_data_T2, _ = read_volseg(self.vol_file.replace('T1', 'T2'))
        self.volume_data_T2 = intensity_normalize(self.volume_data_T2)
        self.axis = axis
        self.volume_data_T1 = self.volume_data_T1[:,40:-40,:]
        self.volume_data_T2 = self.volume_data_T2[:,40:-40,:]

        self.is_true = is_true
        self.transforms = transform
        
    def __len__(self):
        return self.volume_data_T1.shape[self.axis]

    def __getitem__(self, idx):

        if self.axis == 0:
            input1 = self.volume_data_T1[idx, :, :]
            input2 = self.volume_data_T2[idx, :, :]
            
        elif self.axis == 1:
            input1 = self.volume_data_T1[:, idx, :]
            input2 = self.volume_data_T2[:, idx, :]
            
        elif self.axis == 2:
            input1 = self.volume_data_T1[:, :, idx]
            input2 = self.volume_data_T2[:, :, idx]
        else:
            raise ValueError("Invalid axis. Please use 0, 1, or 2.")
        
        input1 = self.transforms(input1)
        input2 = self.transforms(input2)
        
        if self.is_true:
            label = 1
        else:
            p_type = random.random()
            p_tf = random.random()
            if p_type<0.5:
                def_image = input1
            else:
                def_image = input2
            def_image = distort_elastic(def_image)
            # if p_tf<0.33:
            #     print('Only elastic')
            #     def_image = distort_elastic(def_image)
            # elif p_tf>0.33 and p_tf<0.66:
            #     def_image = distort_affine(def_image)
            # else:
            #     def_image = distort_elastic(def_image)
            #     def_image = distort_affine(def_image)
            
            label = 0
            if p_type<0.5:
                return def_image, input2, label
            else:
                return input1, def_image, label
        return input1, input2, label