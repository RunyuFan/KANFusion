import torchvision.models as models
import torch
from torch import nn, Tensor
from models.MMF_KANMixer import *


class MMFKANMixerUV(nn.Module):
    def __init__(self, n_class):
        super(MMFKANMixerUV,self).__init__()
        self.n_class=n_class

        # resnet50 = models.resnet50(pretrained=False)
        # print('resnet50 parameters:', sum(p.numel() for p in resnet50.parameters() if p.requires_grad))
        self.resnext50_32x4d = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnext50_32x4d = nn.Sequential(*self.resnext50_32x4d)
        # print('resnet50 parameters:', sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad))

        # self.resnet50_sv = list(models.resnet50(pretrained=True).children())[:-5]
        # # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        # self.resnet50_sv = nn.Sequential(*self.resnet50_sv)

        # resnet50_sv0 = models.resnet50(pretrained=False)
        # print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv0.parameters() if p.requires_grad))
        self.resnext50_32x4d_sv0 = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnext50_32x4d_sv0 = nn.Sequential(*self.resnext50_32x4d_sv0)

        # resnet50_sv1 = models.resnet50(pretrained=False)
        # print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv1.parameters() if p.requires_grad))
        self.resnext50_32x4d_sv1 = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnext50_32x4d_sv1 = nn.Sequential(*self.resnext50_32x4d_sv1)

        # resnet50_sv2 = models.resnet50(pretrained=False)
        # print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv2.parameters() if p.requires_grad))
        self.resnext50_32x4d_sv2 = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnext50_32x4d_sv2 = nn.Sequential(*self.resnext50_32x4d_sv2)

        # resnet50_sv3 = models.resnet50(pretrained=False)
        # print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv3.parameters() if p.requires_grad))
        self.resnext50_32x4d_sv3 = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnext50_32x4d_sv3 = nn.Sequential(*self.resnext50_32x4d_sv3)

        self.conv_block = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(256*4, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

        self.mixer = MMF_KANMixer(num_classes=2, image_size=64, patch_size=14, channels = 256)
        # self.mixer_img = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.mixer_sv = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.fc = nn.Linear(128, self.n_class)

    def forward(self, img, sv0, sv1, sv2, sv3):
        # print(self.resnet50)
        # print(img.shape, sv0.shape, sv1.shape, sv2.shape, sv3.shape)
        img = self.resnext50_32x4d(img)
        # print(img.shape)
        sv0 = self.resnext50_32x4d_sv0(sv0)
        sv1 = self.resnext50_32x4d_sv1(sv1)
        sv2 = self.resnext50_32x4d_sv2(sv2)
        sv3 = self.resnext50_32x4d_sv3(sv3)
        # print(img.shape, sv0.shape, sv1.shape, sv2.shape, sv3.shape)
        sv = self.conv_block(torch.cat([sv0, sv1, sv2, sv3], 1))
        # sv = nn.AdaptiveAvgPool2d((64, 64))
        # print(img.shape, sv.shape)
        img = self.mixer(img, sv)
        # img = self.mixer_img(img)
        # sv = self.mixer_sv(sv)
        #
        # fuse_cat = torch.cat([img, sv], 1)
        # out = self.fc(fuse_cat)
        return img, img



class MMFKANMixerUV_one_perspective(nn.Module):
    def __init__(self, n_class):
        super(MMFKANMixerUV_one_perspective,self).__init__()
        self.n_class=n_class

        self.resnext50_32x4d = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        self.resnext50_32x4d = nn.Sequential(*self.resnext50_32x4d)

        self.resnext50_32x4d_sv = list(models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2').children())[:-5]
        self.resnext50_32x4d_sv = nn.Sequential(*self.resnext50_32x4d_sv)


        self.mixer = MMF_KANMixer(num_classes=2, image_size=64, patch_size=14, channels = 256)
        # self.mixer_img = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.mixer_sv = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.fc = nn.Linear(128, self.n_class)

        self.conv_block = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, img, sv):
        # print(self.resnet50)
        # print(img.shape, sv0.shape, sv1.shape, sv2.shape, sv3.shape)
        img = self.resnext50_32x4d(img)
        # print(img.shape)
        sv = self.resnext50_32x4d_sv(sv)
        img = self.mixer(img, sv)
        return img, img




