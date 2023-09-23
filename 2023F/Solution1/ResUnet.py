import sys
sys.path.append("2023F/Dataloader")
from data_loader import get_loader

import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T


class SelfAttention(nn.Module):
    """Self Attention Layer"""
    def __init__(self, in_channels, activation='relu', k=8):
        super(SelfAttention, self).__init__()
        self.in_channels =  in_channels
        self.activation = activation
        
        self.W_query = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_key = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):
        B, C, W, H = X.size()
        
        queries = self.W_query(X).view(B, -1, W*H).permute(0, 2, 1) 
        keys = self.W_key(X).view(B, -1, W*H)
        values = self.W_value(X).view(B, -1 ,W*H)
        qk = torch.bmm(queries, keys)
        attention = self.softmax(qk)
        output = torch.bmm(values, attention.permute(0, 2, 1))
        output = output.view(B, C, W, H)
        output = self.gamma * output + X
        
        return output, attention
    
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
    
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
    
class ResUnet(nn.Module):    
    def __init__(self, input_channels=10, filters=None):
        super(ResUnet, self).__init__()
        
        if filters is None:
            filters = [64, 128, 256, 512, 512, 512]
            
 
        
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        )
        
        self.residual_conv_1 = ResidualConv(64, 128, 2, 1)
        self.residual_conv_2 = ResidualConv(128, 256, 2, 1)
        self.residual_conv_3 = ResidualConv(256, 512, 2, 1)
        self.residual_conv_4 = ResidualConv(512, 512, 2, 1)
        self.residual_conv_5 = ResidualConv(512, 512, 2, 1)
        self.residual_conv_6 = ResidualConv(512, 512, 2, 1)
        
        self.bridge = ResidualConv(512, 512, 2, 1)

        # self.b7 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )
        
    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.residual_conv_4(x4)
        x6 = self.residual_conv_5(x5)
        x7 = self.residual_conv_6(x6)
        x8 = self.bridge(x7)

        return [x1, x2, x3, x4, x5, x6, x7, x8]
    
class Decoder(nn.Module):
    def __init__(self, out_channels, add_i):
        super(Decoder, self).__init__()
        self.upsample_1 = Upsample(512*add_i, 512*add_i, 2, 2)
        self.up_residual_conv1 = ResidualConv(512*(1+add_i), 512, 1, 1)
        
        self.upsample_2 = Upsample(512*add_i, 512*add_i, 2, 2)
        self.up_residual_conv2 = ResidualConv(512*(1+add_i), 512, 1, 1)
        
        self.upsample_3 = Upsample(512*add_i, 512*add_i, 2, 2)
        self.up_residual_conv3 = ResidualConv(512*(1+add_i), 512, 1, 1)
        
        self.upsample_4 = Upsample(512*add_i, 512*add_i, 2, 2)
        self.up_residual_conv4 = ResidualConv(512*(1+add_i), 256, 1, 1)
        
        self.upsample_5 = Upsample(256*add_i, 256*add_i, 2, 2)
        self.up_residual_conv5 = ResidualConv(256*(1+add_i), 128, 1, 1)
        
        self.upsample_6 = Upsample(128*add_i, 128*add_i, 2, 2)
        self.up_residual_conv6 = ResidualConv(128*(1+add_i), 64, 1, 1)
        
        self.upsample_7 = Upsample(64*add_i, 64*add_i, 2, 2)
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(64*(1+add_i), out_channels, 1, 1)
        )
    
    def forward(self, encoder_output):
        x1, x2, x3, x4, x5, x6, x7, x8 = encoder_output
        u1 = self.upsample_1(x8)
        c1 = torch.cat([u1, x7], dim=1)
        
        u2 = self.up_residual_conv1(c1)
        u2 = self.upsample_2(u2)
        c2 = torch.cat([u2, x6], dim=1)
        
        u3 = self.up_residual_conv2(c2)
        u3 = self.upsample_3(u3)
        c3 = torch.cat([u3, x5], dim=1)
        
        u4 = self.up_residual_conv3(c3)
        u4 = self.upsample_4(u4)
        c4 = torch.cat([u4, x4], dim=1)
        
        u5 = self.up_residual_conv4(c4)
        u5 = self.upsample_5(u5)
        c5 = torch.cat([u5, x3], dim=1)
        
        u6 = self.up_residual_conv5(c5)
        u6 = self.upsample_6(u6)
        c6 = torch.cat([u6, x2], dim=1)
        
        u7 = self.up_residual_conv6(c6)
        u7 = self.upsample_7(u7)
        c7 = torch.cat([u7, x1], dim=1)
        
        output = self.output_layer(c7)
        return [c1, c2, c3, c4, c5, c6, c7, output]
        
    
def trainer():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    dataloader = get_loader(mode="train", question="1", batch_size=6)
    
    FE_1km_Zh = Encoder(10).cuda()
    FE_1km_Zh = torch.nn.DataParallel(FE_1km_Zh)
    optim_FE_1km_Zh = torch.optim.AdamW(FE_1km_Zh.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FE_1km_Zdr = Encoder(10).cuda()
    FE_1km_Zdr = torch.nn.DataParallel(FE_1km_Zdr)
    optim_FE_1km_Zdr = torch.optim.AdamW(FE_1km_Zdr.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FE_1km_Kdp = Encoder(10).cuda()
    FE_1km_Kdp = torch.nn.DataParallel(FE_1km_Kdp)
    optim_FE_1km_Kdp = torch.optim.AdamW(FE_1km_Kdp.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    

if __name__ == "__main__":
    E = Encoder(10)
    x = torch.randn((4, 10, 256, 256))
    y = E(x)
    D = Decoder(10, 1)
    y = D(y)
    for i in y:
        print(i.shape)
    
    