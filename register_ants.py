import numpy as np
import time
from tqdm import tqdm
import glob
import sys
sys.path.insert(0,'/home/gayathri/Tanvi_Temp/for_vxm/lib/python3.8/site-packages/')
import os
import ants
from utils.common_utils import  visualize_reg

GT_files = glob.glob('../ground_truth_sections/*')

if not os.path.exists('../multimodal_registered_sections_T1'):
    os.makedirs('../multimodal_registered_sections_T1')

if not os.path.exists(f'../visualize_multimodal_registered_sections_T1'):
    os.makedirs(f'../visualize_multimodal_registered_sections_T1')

ants_list = []
for gt_file in tqdm(GT_files):
    print(gt_file)
    ref = np.load(gt_file.replace('ground_truth_sections', 'ground_truth_sections_T2').replace('T1', 'T2'))
    img = np.load(gt_file.replace('ground_truth_sections', 'distorted_sections'))

    fi_ants = ants.from_numpy(ref)
    mi_ants = ants.from_numpy(img)
    step_start_time = time.time()
    mytx = ants.registration(fixed=fi_ants, moving=mi_ants, type_of_transform = 'SyN',
                        aff_metric='CC', syn_metric='CC')
    moved_section = mytx['warpedmovout'].numpy()
    ants_list += [time.time() - step_start_time]
    out_file = gt_file.replace('ground_truth_sections', 'multimodal_registered_sections_T1')
    print(f'------------Saving to {out_file}----------------')
    np.save(out_file, moved_section)
    visualize_reg(ref, img, moved_section, 
                gt_file.replace('ground_truth_sections', 'visualize_multimodal_registered_sections_T1').replace('.npy', f'.png') )
    
print(f'Average ANTS Inference Time!: {np.mean(ants_list):.3f}')

