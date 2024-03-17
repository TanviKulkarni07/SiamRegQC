import nibabel as nib
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torchvision.utils as vutils
from torchvision import transforms

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def intensity_normalize(img):
    return (img-img.min())/(img.max()-img.min())

def read_volseg(volfile):
    
    if volfile.endswith('pkl'):
        vol,seg = pkload(volfile)
        seg = (seg!=0).astype(np.uint8)
    else:
        vol = nib.load(volfile).get_fdata()
        try:
            seg = nib.load(volfile.replace('volumes','labels')).get_fdata()
        except:
            seg = None
    return vol, seg

def visualise_AE_ipop(tensor, num_images):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(tensor[:num_images], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()
    print(tensor.min(), tensor.max())

def visualize_reg(gt, deformed, reg, filename):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plotting on the first subplot
    axes[0].imshow(gt, cmap='gray')
    axes[0].set_title('GT')

    # Plotting on the second subplot
    axes[1].imshow(deformed, cmap='gray')
    axes[1].set_title('Elastic Def')

    # Plotting on the third subplot
    axes[2].imshow(reg, cmap='gray')
    axes[2].set_title('Ants Reg')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(filename)

def save_section(img, filename):
    np.save(filename, img)

def pt_transform(ref, transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256)])):
    ref_tensor = transform(ref)
    return ref_tensor