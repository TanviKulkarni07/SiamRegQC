import numpy as np
import time
from tqdm import tqdm
import glob
import sys
sys.path.insert(0,'/home/gayathri/Tanvi_Temp/for_vxm/lib/python3.8/site-packages/')
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import ants
import torch
from utils.common_utils import read_volseg, save_section, intensity_normalize, visualize_reg, pt_transform

GT_files = glob.glob('../ground_truth_sections/*')
device = 'cuda:0'

ndim = 1
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# vol_shape = torch_fxd.shape[1:-1]
vol_shape = (256,256)
batch_size=32
# device = 'cuda:0'
vxm_file = '../model_weights/VXM_multimodal_DeepSim.pt'
pref = os.path.basename(vxm_file)[:-3]
out_folder = f'../{pref}_registered_sections_T1'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

if not os.path.exists(f'../visualize_{pref}_sections_T1'):
    os.makedirs(f'../visualize_{pref}_sections_T1')

vxm_model = vxm.networks.VxmDense(vol_shape,nb_features, int_steps=0).to(device)
# vxm_model = vxm.networks.VxmDense.load(vxm_file, device)
print('---------------Loaded Model------------------')
print(vxm_file)
vxm_model.load_state_dict(torch.load(vxm_file))
ts_list = []
ants_list = []
for gt_file in tqdm(GT_files):
    print(gt_file)
    ref = np.load(gt_file.replace('ground_truth_sections', 'ground_truth_sections_T2').replace('T1', 'T2'))
    img = np.load(gt_file.replace('ground_truth_sections', 'affine_registered_sections'))

    # fi_ants = ants.from_numpy(ref)
    # mi_ants = ants.from_numpy(img)
    # step_start_time = time.time()
    # mytx = ants.registration(fixed=fi_ants, moving=mi_ants, type_of_transform = 'Affine',
    #                     aff_metric='CC')
    # moved_section = mytx['warpedmovout'].numpy()
    # ants_list += [time.time() - step_start_time]
    # np.save(gt_file.replace('ground_truth_sections', f'multimodal_affine_registered_sections'), moved_section)
    
    # inputs1 = pt_transform(moved_section).unsqueeze(0)
    inputs1 = pt_transform(img).unsqueeze(0)
    inputs2 = pt_transform(ref).unsqueeze(0)
    inputs1 = inputs1.type(torch.FloatTensor).to(device)
    inputs2 = inputs2.type(torch.FloatTensor).to(device)
    step_start_time = time.time()
    out = vxm_model(inputs1, inputs2)
    ts_list += [time.time() - step_start_time]

    out_np = out[0][0,0].detach().cpu().numpy()

    out_file = gt_file.replace('ground_truth_sections', f'{pref}_registered_sections_T1')
    np.save(out_file, out_np)
    visualize_reg(ref, img, out_np, 
                gt_file.replace('ground_truth_sections', f'visualize_{pref}_sections_T1').replace('.npy', f'.png') )
    print(f'------------Saving to {out_file}----------------')
print(f'Average VXM Inference Time!: {np.mean(ts_list):.3f}')
# print(f'Average ANTS Inference Time!: {np.mean(ants_list):.3f}')

