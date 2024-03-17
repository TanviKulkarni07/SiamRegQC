import random
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torchvision import transforms
import torch.nn.functional as nnf

def distort_image_2class(img):
    
    p = random.uniform(0,1)
    r = np.random.uniform(4,20 )
        
    if p<0.33:
        angle = r
        t_x = 0
        t_y = 0
    elif p< 0.66:
        angle = 0
        t_x = r
        t_y = 0
    else:
        angle = 0
        t_x = 0
        t_y = r
        
    mov = transforms.functional.affine(transforms.ToTensor()(img), angle, 
                    [t_x,t_y],1,1,
                    interpolation= transforms.InterpolationMode.BILINEAR).numpy()[0]
        
    return mov, 0

def random_elastic_transform(shape, max_displacement, sigma, alpha):
    deformation = np.random.uniform(-max_displacement, max_displacement, (*shape, 2))
    smoothed_deformation = np.zeros_like(deformation)
    for i in range(2):
        smoothed_deformation[..., i] = gaussian_filter(deformation[..., i], sigma=sigma)
        
    smoothed_deformation_tensor = torch.from_numpy(smoothed_deformation).type(torch.FloatTensor)
    d = torch.linspace(-1, 1, shape[0])
    meshx, meshy = torch.meshgrid((d, d))
    meshx_new = alpha*meshx + smoothed_deformation_tensor[...,0]
    meshy_new = alpha*meshy + smoothed_deformation_tensor[...,1]
    
    grid = torch.stack((meshy_new, meshx_new), 2)
    return grid

def distort_elastic(tensor):
    max_displacement = np.random.uniform(2,5)
    sigma = np.random.uniform(20,30)
    # alpha = np.random.uniform(0.6, 1.5)
    alpha = np.random.uniform(0.9, 1.1)

    grid = random_elastic_transform(tensor.shape[1:], max_displacement, sigma, alpha)
    
    out = nnf.grid_sample(tensor.unsqueeze(0).type(torch.FloatTensor), grid.unsqueeze(0).type(torch.FloatTensor), align_corners=True, mode='bilinear')
    return out[0]

def distort_affine(img):
    
    p = random.uniform(0,1)
    r = np.random.uniform(4,20 )
        
    if p<0.33:
        angle = r
        t_x = 0
        t_y = 0
    elif p< 0.66:
        angle = 0
        t_x = r
        t_y = 0
    else:
        angle = 0
        t_x = 0
        t_y = r
        
    mov = transforms.functional.affine(img, angle, 
                    [t_x,t_y],1,1,
                    interpolation= transforms.InterpolationMode.BILINEAR)
        
    return mov, 0