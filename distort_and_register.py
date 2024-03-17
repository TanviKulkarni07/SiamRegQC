import sys
sys.path.insert(0,'/home/gayathri/Tanvi_Temp/for_vxm/lib/python3.8/site-packages/')

import ants
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from utils.distortions import distort_elastic, random_elastic_transform
from utils.common_utils import read_volseg, save_section, intensity_normalize, visualize_reg

def save_Gt_Def_Reg_sections(fullname):

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])
    
    filename = os.path.basename(fullname)
    vol, _ = read_volseg(fullname)
    vol = intensity_normalize(vol)
    
    for secnum in tqdm(range(45, vol.shape[1]-45)):
        section = vol[:,secnum,:]
        print(f'------------Saving Section {secnum:03d}-------------')
        section_tensor = transform(section)
        deformed_tensor = distort_elastic(section_tensor)

        deformed_section = deformed_tensor.numpy()[0]
        section_mod = section_tensor.numpy()[0]

        save_section(section_mod, 
                '../ground_truth_sections_T2/' + filename.replace('.nii.gz', f'{secnum:03d}.npy'))

        save_section(deformed_section, 
                '../distorted_sections_T2/' + filename.replace('.nii.gz', f'{secnum:03d}.npy'))
        
        fi_ants = ants.from_numpy(section_mod)
        mi_ants = ants.from_numpy(deformed_section)
        print('Sanity Check:', section.max(), section.min(), 
              deformed_section.max(), deformed_section.min())
        
        mytx = ants.registration(fixed=fi_ants, moving=mi_ants, type_of_transform = 'SyN' )
        moved_section = mytx['warpedmovout'].numpy()
        save_section(moved_section, 
                '../registered_sections_T2/' + filename.replace('.nii.gz', f'{secnum:03d}.npy'))
        
        visualize_reg(section_mod, deformed_section, moved_section, 
                '../visualize_sections_T2/' + filename.replace('.nii.gz', f'{secnum:03d}.png') )

def main():
    fullname = '../reg_n4/IXI030-Guys-0708-T2.nii.gz'
    save_Gt_Def_Reg_sections(fullname)

if __name__ == '__main__':
    main()

