import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.signal import correlate2d
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F

def _mse(ref, img):
    return np.mean(np.abs(ref-img)**2)

def _ssim(ref, img):
    return ssim(ref, img, data_range = img.max()-img.min())

def _mind(ref, img, window_size=5):
    mind_score = 0.0
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            # Extract local neighborhoods
            neighborhood1 = ref[max(0, i - window_size):min(ref.shape[0], i + window_size + 1),
                                   max(0, j - window_size):min(ref.shape[1], j + window_size + 1)]
            neighborhood2 = img[max(0, i - window_size):min(img.shape[0], i + window_size + 1),
                                   max(0, j - window_size):min(img.shape[1], j + window_size + 1)]
            # Compute histograms
            hist1, _ = np.histogram(neighborhood1, bins=256, range=(0, 1))
            hist2, _ = np.histogram(neighborhood2, bins=256, range=(0, 1))
            # Compute mutual information
            mind_score += mutual_info_score(hist1, hist2)
    return mind_score

def _ncc(image1, image2):
    # Compute means
    mean_image1 = np.mean(image1)
    mean_image2 = np.mean(image2)
    
    # Subtract means
    image1_centered = image1 - mean_image1
    image2_centered = image2 - mean_image2
    
    # Compute normalized cross-correlation
    cross_corr = correlate2d(image1_centered, image2_centered, mode='valid')
    ncc_value = cross_corr / (np.sqrt(np.sum(image1_centered**2)) * np.sqrt(np.sum(image2_centered**2)))
    
    return ncc_value[0,0]

def _siam(model, ref, img, device):

    input1 = img.type(torch.FloatTensor).to(device)
    input2 = ref.type(torch.FloatTensor).to(device)
    x = model(input1, input2)
    output1 = model.forward_one(input1)
    output2 = model.forward_one(input2)
    return torch.argmax(x,1).cpu().detach().numpy()[0], 1-F.cosine_similarity(output1.reshape(1,-1), output2.reshape(1,-1)).cpu().detach().numpy()[0], 1-F.cosine_similarity(input1.reshape(1,-1), input2.reshape(1,-1)).cpu().detach().numpy()[0]


def plot_loss_curve(X,Y,Z, title='SiamRegQC', z_max = np.arange(0,0.1, 0.05), filename='nil.png'):
    # Plot 
    plt.close()
    a = 1
    b = 1
    # a_s = 5
    # b_s = 2
    a_s = 1
    b_s = 1
    fig = plt.figure(figsize=(8,8))
    # plt.subplots_adjust(top=1, bottom=0)
    ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X[a:-a:a_s, b:-b:b_s]*50/1000, Y[a:-a:a_s, b:-b:b_s], Z[a:-a:a_s, b:-b:b_s],
    #                        cmap='jet', alpha=0.9)
    surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.9)

    # contour_levels = np.linspace(0, 0.0014, 15)
    # # Plot full contours
    # ax.contour(X[a:-a:a_s, b:-b:b_s]*50/1000, Y[a:-a:a_s, b:-b:b_s], MSE_Z[a:-a:a_s, b:-b:b_s],
    #            contour_levels, zdir='z', offset=0, cmap='jet', alpha=1)

    # Label axes
    ax.set_xlabel('Translation Error (mm)', fontsize=17, labelpad=12)
    ax.set_ylabel('Rotation Error (deg)', fontsize=17, labelpad=12)
    ax.view_init(elev=30, azim=45)

    # ax.set_xticks(np.arange(-0.3,0.4,0.15))
    # ax.set_yticks(np.arange(-4,6,2))
    # ax.set_zticks(z_max)
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,1))

#     yfmt = ticker.ScalarFormatter(useMathText=True)
#     yfmt.set_powerlimits((-2, -1))
#     ax.zaxis.set_major_formatter(yfmt)
#     ax.get_zaxis().get_offset_text().set_visible(True)
#     ax_max = max(ax.get_zticks())
#     exponent_axis = np.floor(np.log10(ax_max)).astype(int)
#     ax.annotate(r'$\times$10$^{%i}$'%(exponent_axis),
#                  xy=(0,0), xycoords='axes fraction')
    plt.tick_params(labelsize=17, grid_alpha=0.1)
    plt.savefig(filename)
    
    # # H = hessian_matrix(Z)
    # jac_num = np.sum(np.linalg.eigvals(H) >= 0) 
    
    neighborhood_size = 30

    # Calculate the local variance
    local_variance = np.zeros_like(Z)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # Define the indices of the local neighborhood
            i_min = max(0, i - neighborhood_size // 2)
            i_max = min(Z.shape[0], i + neighborhood_size // 2 + 1)
            j_min = max(0, j - neighborhood_size // 2)
            j_max = min(Z.shape[1], j + neighborhood_size // 2 + 1)

            # Extract the local neighborhood
            local_patch = Z[i_min:i_max, j_min:j_max]

            # Calculate the variance of the local neighborhood
            local_variance[i, j] = np.var(local_patch)

    print('Sensitivity:', np.round((np.max(Z)-np.min(Z)),3))
    # print('Jacobian:', jac_num)
    print('LV:', local_variance.max())
    

def hessian_matrix(Z):
    loss_function = Z
    h_xx = np.gradient(np.gradient(loss_function, axis=0), axis=0)
    h_yy = np.gradient(np.gradient(loss_function, axis=1), axis=1)
    h_xy = np.gradient(np.gradient(loss_function, axis=0), axis=1)
    H = np.array([[h_xx, h_xy], [h_xy, h_yy]])
    print(H.shape)
    # Check if the Hessian matrix is positive semidefinite (PSD) for all points
    is_positive_semidefinite = np.sum(np.linalg.eigvals(H) < 0)/(H.shape[0]*H.shape[1])
    return is_positive_semidefinite, np.var(H)