import glob
import random
import os
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataloader import Multimodal_AEDataset
from utils.common_utils import visualise_AE_ipop


def create_multimodal_data(GT_files, transform):
    multi_data = Multimodal_AEDataset(GT_files[0], transform, 1, is_true=True)
    multi_data += Multimodal_AEDataset(GT_files[0], transform, 1, is_true=False)

    for i in range(1,len(GT_files)):
        if not os.path.exists(GT_files[i]) or not os.path.exists(GT_files[i].replace('T1', 'T2')):
            continue
        true_data = Multimodal_AEDataset(GT_files[i], transform, 1, is_true=True)
        false_data = Multimodal_AEDataset(GT_files[i], transform, 1, is_true=False)

        multi_data += true_data + false_data
    multi_dataloader = DataLoader(multi_data, batch_size=32, shuffle=True)
    input1, input2, label = next(iter(multi_dataloader))
    visualise_AE_ipop(input2, 5)
    plt.savefig('ainvayi.png')
    visualise_AE_ipop(input1, 5)
    plt.savefig('ainvayi1.png')
    return multi_data, multi_dataloader


# GT_files = random.sample(glob.glob('../reg_n4/*Guys*T1*'), 10)
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.CenterCrop(256),
#                                ])
# print(len(multi_data))
# input1, input2, label = next(iter(multi_dataloader))
# visualise_AE_ipop(input2, 5)
# plt.savefig('ainvayi.png')
# visualise_AE_ipop(input1, 5)
# plt.savefig('ainvayi1.png')
