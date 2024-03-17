import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Autoencoder, SiameseNetwork_AE, Autoencoder_Simon, MultimodalSiameseNetwork_AE

class CosineContrastiveLoss(nn.Module):
    def __init__(self,margin=0.5):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = F.cosine_similarity(output1, output2)

        # Contrastive loss formula based on cosine similarity
        loss_contrastive = torch.mean((label) * torch.pow(1 - cosine_similarity, 2) +
                                      (1-label) * torch.pow(torch.clamp(cosine_similarity - self.margin, min=0.0), 2))

        return loss_contrastive

class SiamRegQC_loss(nn.Module):
    def __init__(self, device, model_file = '../model_weights/MRI_GuysT1__20240309-215304_BS32_EP10.pt', margin=0.5):
        super(SiamRegQC_loss, self).__init__()
        self.margin = margin
        self.device = device
        autoencoder = Autoencoder()
        autoencoder = autoencoder.to(device)
        self.model = SiameseNetwork_AE(autoencoder.encoder).to(device)
        self.model.load_state_dict(torch.load(model_file))
        
    def forward(self, ref_tensor, img_tensor):
        self.model.eval()
        output1 = self.model.forward_one(img_tensor)
        output2 = self.model.forward_one(ref_tensor)
        return 1-F.cosine_similarity(output1, output2).mean()

class SiamRegQC_multimodal_loss(nn.Module):
    def __init__(self, device, model_file = '../MRI_Guys_multimodal__20240312-210249_BS32_EP10.pt', margin=0.5):
        super(SiamRegQC_multimodal_loss, self).__init__()
        self.margin = margin
        self.device = device
        # Instantiate the model
        print('------------Loading Encoder T1-----------------')
        autoencoder_T1 = Autoencoder()
        autoencoder_T1 = autoencoder_T1.to(device)
        autoencoder_T1.load_state_dict(torch.load('../model_weights/MRI_AE.pt'))

        print('------------Loading Encoder T2-----------------')
        autoencoder_T2 = Autoencoder()
        autoencoder_T2 = autoencoder_T1
        autoencoder_T2.load_state_dict(torch.load('../model_weights/MRI_T2_AE.pt'))

        self.model = MultimodalSiameseNetwork_AE(autoencoder_T1, autoencoder_T2).to(device)
        self.model.load_state_dict(torch.load('../model_weights/MRI_Guys_multimodal__20240312-210249_BS32_EP10.pt'))
        self.model.eval()
        
    def forward(self, ref_tensor, img_tensor):
        self.model.eval()
        output1 = self.model.forward_one(self.model.encoder_T1, img_tensor)
        output2 = self.model.forward_one(self.model.encoder_T2, ref_tensor)
        return 1-F.cosine_similarity(output1, output2).mean()
    
class Simonvosky_loss(nn.Module):
    def __init__(self, device, model_file = '../model_weights/MRI_GuysT1_Simon_NPNL_20240310-132842_BS32_EP10.pt', margin=0.5):
        super(Simonvosky_loss, self).__init__()
        self.margin = margin
        self.device = device
        self.model = Autoencoder_Simon().to(device)
        self.model.load_state_dict(torch.load(model_file))
        
    def forward(self, ref_tensor, img_tensor):
        self.model.eval()
        out = self.model(img_tensor, ref_tensor)
        return out[..., 0].mean()