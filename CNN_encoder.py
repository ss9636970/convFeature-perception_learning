import torch
import torch.nn as nn
import numpy as np
import os

# CNN encoder
class CNN_model_encoder(nn.Module):
    def __init__(self):       #words為字集合，m為embedding長度
        super(CNN_model_encoder, self).__init__()
        # for input 3 * 1024 * 1024 tensor
        
        self.conv1 = nn.Sequential(        # 計算 loss  15層
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, stride=2),
            nn.LayerNorm([3, 256, 256], elementwise_affine=True),
            nn.ReLU()
        )        # h1, 3 * 256 * 256
        self.conv2 = nn.Sequential(        # 計算 loss  7層
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, stride=2),
            nn.LayerNorm([3, 128, 128], elementwise_affine=True),
            nn.ReLU()
        )        #h2, 3 * 128 * 128
        self.conv3 = nn.Sequential(        # 計算 loss  6層
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, stride=2),
            nn.LayerNorm([3, 64, 64], elementwise_affine=True),
            nn.ReLU()
        )        #h3, 3 * 64 * 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1, stride=2),
            nn.LayerNorm([1, 32, 32], elementwise_affine=True),
            nn.ReLU()
        )        #h4, 1 * 32 * 32
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(1, 3, kernel_size=(3, 3), padding=1, stride=1),
            nn.LayerNorm([3, 32, 32], elementwise_affine=True),
            nn.ReLU()
        )        #h5, 3 * 32 * 32
        self.conv6 = nn.Sequential(        # 計算 loss  3層
            nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), padding=1, output_padding=1, stride=2),
            nn.LayerNorm([3, 64, 64], elementwise_affine=True),
            nn.ReLU()
        )        #h6, 3 * 64 * 64
        self.conv7 = nn.Sequential(        # 計算 loss  2層
            nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), padding=1, output_padding=1, stride=2),
            nn.LayerNorm([3, 128, 128], elementwise_affine=True),
            nn.ReLU()
        )        #h7, 3 * 128 * 128
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), padding=1, output_padding=1, stride=2),
            nn.LayerNorm([3, 256, 256], elementwise_affine=True),
            nn.ReLU()
        )        #h8, 3 * 256 * 256
        self.conv9 = nn.Sequential(        # 計算 loss  14層
            nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), padding=1, output_padding=1, stride=2),
            nn.LayerNorm([3, 512, 512], elementwise_affine=True),
            nn.ReLU()
        )        #h9, 3 * 128 * 128
        

    def forward(self, x):
        h0 = x
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        h6 = self.conv6(h5)
        h7 = self.conv7(h6)
        h8 = self.conv8(h7)
        h9 = self.conv9(h8)
        y = [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9]

        return y

    def feature_extract(self, x):     # h4, h11      32 * 32
        with torch.no_grad():
            n = x.shape[0]
            h1 = self.conv1(x)
            h2 = self.conv2(h1)
            h3 = self.conv3(h2)
            h4 = self.conv4(h3)
            feature = h4.view(n, -1)
            
        return feature

# # CNN encoder
# class CNN_model_encoder(nn.Module):
#     def __init__(self):       #words為字集合，m為embedding長度
#         super(CNN_model_encoder, self).__init__()
#         # for input 3 * 1024 * 1024 tensor
        
#         self.conv1 = nn.Sequential(        # 計算 loss  15層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((256, 256)),
#             nn.LayerNorm([256, 256], elementwise_affine=False)
#         )        # h1, 3 * 256 * 256
#         self.conv2 = nn.Sequential(        # 計算 loss  7層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((128, 128)),
#             nn.LayerNorm([128, 128], elementwise_affine=False)
#         )        #h2, 3 * 128 * 128
#         self.conv3 = nn.Sequential(        # 計算 loss  6層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((64, 64)),
#             nn.LayerNorm([64, 64], elementwise_affine=False)
#         )        #h3, 3 * 64 * 64
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((32, 32)),
#             nn.LayerNorm([32, 32], elementwise_affine=False)
#         )        #h4, 1 * 32 * 32
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((32, 32)),
#             nn.LayerNorm([32, 32], elementwise_affine=False)
#         )        #h5, 3 * 32 * 32
#         self.conv6 = nn.Sequential(        # 計算 loss  3層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((64, 64)),
#             nn.LayerNorm([64, 64], elementwise_affine=False)
#         )        #h6, 3 * 64 * 64
#         self.conv7 = nn.Sequential(        # 計算 loss  2層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((128, 128)),
#             nn.LayerNorm([128, 128], elementwise_affine=False)
#         )        #h7, 3 * 128 * 128
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((256, 256)),
#             nn.LayerNorm([256, 256], elementwise_affine=False)
#         )        #h8, 3 * 256 * 256
#         self.conv9 = nn.Sequential(        # 計算 loss  14層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((128, 128)),
#             nn.LayerNorm([128, 128], elementwise_affine=False)
#         )        #h9, 3 * 128 * 128
#         self.conv10 = nn.Sequential(        # 計算 loss  13層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((64, 64)),
#             nn.LayerNorm([64, 64], elementwise_affine=False)
#         )        #h10, 3 * 64 * 64
#         self.conv11 = nn.Sequential(
#             nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((32, 32)),
#             nn.LayerNorm([32, 32], elementwise_affine=False)
#         )        #h11, 1 * 32 * 32
#         self.conv12 = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((32, 32)),
#             nn.LayerNorm([32, 32], elementwise_affine=False)
#         )        #h12, 3 * 32 * 32
#         self.conv13 = nn.Sequential(             # 計算 loss  10層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((64, 64)),
#             nn.LayerNorm([64, 64], elementwise_affine=False)
#         )        #h13, 3 * 64 * 64
#         self.conv14 = nn.Sequential(              # 計算 loss  9層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((128, 128)),
#             nn.LayerNorm([128, 128], elementwise_affine=False)
#         )        #h14, 3 * 128 * 128
#         self.conv15 = nn.Sequential(              # 計算 loss  1層
#             nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
#             nn.BatchNorm2d(3, affine=False),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d((256, 256)),
#             nn.LayerNorm([256, 256], elementwise_affine=False)
#         )        #h15, 3 * 256 * 256

        

#     def forward(self, x):
#         h1 = self.conv1(x)
#         h2 = self.conv2(h1)
#         h3 = self.conv3(h2)
#         h4 = self.conv4(h3)
#         h5 = self.conv5(h4)
#         h6 = self.conv6(h5)
#         h7 = self.conv7(h6)
#         h8 = self.conv8(h7)
#         h9 = self.conv9(h8)
#         h10 = self.conv10(h9)
#         h11 = self.conv11(h10)
#         h12 = self.conv12(h11)
#         h13 = self.conv13(h12)
#         h14 = self.conv14(h13)
#         h15 = self.conv15(h14)
#         y = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15]

#         return y

#     def feature_extract(self, x):     # h4, h11      32 * 32
#         with torch.no_grad():
#             n = x.shape[0]
#             h1 = self.conv1(x)
#             h2 = self.conv2(h1)
#             h3 = self.conv3(h2)
#             h4 = self.conv4(h3)
#             h5 = self.conv5(h4)
#             h6 = self.conv6(h5)
#             h7 = self.conv7(h6)
#             h8 = self.conv8(h7)
#             h9 = self.conv9(h8)
#             h10 = self.conv10(h9)
#             h11 = self.conv11(h10)
#             f1 = h4.view(n, -1)
#             f2 = h11.view(n, -1)
#             feature = torch.cat([f1, f2], dim=1)

#         return feature





