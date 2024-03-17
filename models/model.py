import torch
import torch.nn as nn

class SiameseNetwork_AE(nn.Module):
    def __init__(self, input_ae):
        super(SiameseNetwork_AE, self).__init__()

        # Define the shared subnetwork (twin)
        self.shared_layer1 = input_ae
        
#         for name, param in self.shared_layer1.named_parameters():
#             param.requires_grad = False
        
        # Fully connected layers for the final embedding
        self.fc = nn.Sequential(
            nn.Linear(16*16*16*2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
#             nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward_one(self, x):
        # Forward pass for one input
        x = self.shared_layer1(x)
        x = x.view(x.size()[0], -1)
#         x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        concatenated_latent = torch.cat((output1, output2), dim=1)
        x = self.fc(concatenated_latent)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Autoencoder_Simon(nn.Module):
    def __init__(self):
        super(Autoencoder_Simon, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*16*16*1, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
#             nn.Sigmoid()
            nn.Softmax(dim=1)
        )
    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

import copy

class MultimodalSiameseNetwork_AE(nn.Module):
    def __init__(self, encoder_T1, encoder_T2):
        super(MultimodalSiameseNetwork_AE, self).__init__()
        
        # self.encoder_T1 = copy.deepcopy(encoder_T1.encoder)
        # self.encoder_T2 = copy.deepcopy(encoder_T2.encoder)
        self.encoder_T1 = encoder_T1.encoder
        self.encoder_T2 = encoder_T2.encoder

        self.common_cnn = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            )

        self.fc = nn.Sequential(
            nn.Linear(16*16*16*1, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
#             nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def intermediate_outputs(self, model, input_tensor):
        # output_tensor = copy.deepcopy(input_tensor)
        output_tensor = input_tensor.clone()
        for name, layer in model.named_children():
            if name == '8':
                break
            output_tensor = layer(output_tensor)
        return output_tensor
    
    def forward_one(self, encoder, input_T1):
        out_T1 = self.intermediate_outputs(encoder, input_T1)
        return out_T1.view(out_T1.size()[0], -1)

    def forward(self, input_T1, input_T2):
        out_T1 = self.intermediate_outputs(self.encoder_T1, input_T1)
        out_T2 = self.intermediate_outputs(self.encoder_T2, input_T2)
        out_cat = torch.cat([out_T1, out_T2], dim=1)
        out_cat = self.common_cnn(out_cat)
        out_cat = out_cat.view(out_cat.size()[0], -1)
        out_cat = self.fc(out_cat)
        return out_cat
    

class MultimodalSiameseNetwork_VGG(nn.Module):
    def __init__(self, vgg):
        super(MultimodalSiameseNetwork_VGG, self).__init__()
        
        self.vgg_encoder = torch.nn.Sequential(*list(vgg.features.children()))
        self.vgg_encoder[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Copy the weights and biases from the original model's first layer
        with torch.no_grad():
            self.vgg_encoder[0].weight[:, 0, :, :] = torch.mean(vgg.features[0].weight, dim=1)
            self.vgg_encoder[0].bias = vgg.features[0].bias

        self.common_cnn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            )

        self.fc = nn.Sequential(
            # nn.Linear(64*8*8, 1024),
            nn.Linear(512*32*32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
#             nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def intermediate_outputs(self, model, input_tensor):
        output_tensor = copy.deepcopy(input_tensor)
        for name, layer in model.named_children():
            if name == '17':
                break
            # print(name)
            output_tensor = layer(output_tensor)
        return output_tensor
    
    def forward_one(self, encoder, input_T1):
        out_T1 = self.intermediate_outputs(encoder, input_T1)
        return out_T1.view(out_T1.size()[0], -1)

    def forward(self, input_T1, input_T2):
        out_T1 = self.intermediate_outputs(self.vgg_encoder, input_T1)
        out_T2 = self.intermediate_outputs(self.vgg_encoder, input_T2)
        # print(out_T1.shape, out_T2.shape)
        out_cat = torch.cat([out_T1, out_T2], dim=1)
        # out_cat = self.common_cnn(out_cat)
        out_cat = out_cat.view(out_cat.size()[0], -1)
        out_cat = self.fc(out_cat)
        return out_cat