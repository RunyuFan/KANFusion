import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .kan import KANLinear
class PatchEmbeddings(nn.Module):

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        channels: int
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalAveragePooling(nn.Module):

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = KANLinear(input_dim, num_classes)
        # nn.init.zeros_(self.model.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class KANBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            KANLinear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.GELU(),
            KANLinear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class KANMixerBlock(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            KANBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            KANBlock(num_channels, channels_hidden_dim)
        )
        self.token_mixing_sv = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            KANBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_sv = nn.Sequential(
            nn.LayerNorm(num_channels),
            KANBlock(num_channels, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        x_token_img = img + self.token_mixing_img(img)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)

        x_token_sv = sv + self.token_mixing_sv(sv)
        x_sv = x_token_sv + self.channel_mixing_sv(x_token_sv)
        return x_img, x_sv




class KANMixerBlock_fuse(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*2),
            Rearrange("b p c -> b c p"),
            KANBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*2),
            KANBlock(num_channels*2, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # print(img.shape)
        # print(self.token_mixing_img(img).shape)
        x_token_img = img + self.token_mixing_img(img)
        # print('x_token_img.shape, self.token_mixing_img(img).shape, self.channel_mixing_img(x_token_img).shape', x_token_img.shape, self.token_mixing_img(img).shape, self.channel_mixing_img(x_token_img).shape)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)
        # print(x_img.shape)
        return x_img



class MMF_KANMixer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        patch_size: int,
        channels: int = 256,
        hidden_dim: int = 256,
        tokens_hidden_dim: int = 128,
        channels_hidden_dim: int = 1024
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.embed_img = PatchEmbeddings(patch_size, hidden_dim, channels)
        self.embed_sv = PatchEmbeddings(patch_size, hidden_dim, channels)
        # self.embed_fuse = PatchEmbeddings(patch_size, hidden_dim, channels*2)
        self.Mixerlayer1 = KANMixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixerlayer2 = KANMixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixerlayer3 = KANMixerBlock(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )
        self.Mixerlayer4 = KANMixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )

        self.Mixerlayer_fuse_early = KANMixerBlock_fuse(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )
        # self.Mixerlayer_fuse_mid = MixerBlock_fuse(
        #         num_patches=num_patches,
        #         num_channels=hidden_dim,
        #         tokens_hidden_dim=tokens_hidden_dim,
        #         channels_hidden_dim=channels_hidden_dim
        #     )
        self.Mixerlayer_fuse_late = KANMixerBlock_fuse(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )

        self.norm = nn.LayerNorm(hidden_dim*6)

        self.pool = GlobalAveragePooling(dim=1)
        self.classifier = Classifier(hidden_dim*6, num_classes)


    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        # x = torch.cat([img, sv], 1)
        # print(x_fuse.shape)
        # x_fuse = self.embed_fuse(x_fuse)
        # x_img = self.ChannelAttentionModule(x_img)
        x_img = self.embed_img(img)           # [b, p, c]
        x_sv = self.embed_sv(sv)           # [b, p, c]
        # print(x_img.shape, x_sv.shape)
        # x = torch.cat([x_img, x_sv], 1)
        # print(x.shape)
        x_img_1, x_sv_1 = self.Mixerlayer1(x_img, x_sv)          # [b, p, c]
        # print(x_img_1.shape, x_sv_1.shape)

        # x_fuse_early = self.MLP_early(x_fuse_early)
        x_fuse_early = torch.cat([x_img_1, x_sv_1], 2)
        # print(x_img_1.shape, x_sv_1.shape, x_fuse_early.shape)
        # print('x_fuse_early.shape:', x_fuse_early.shape)
        x_fuse_early = self.Mixerlayer_fuse_early(x_fuse_early)
        # print('x_fuse_early.shape:', x_fuse_early.shape)

        x_sv_1 = x_img_1 + x_sv_1
        x_img_2, x_sv_2 = self.Mixerlayer2(x_img_1, x_sv_1)
        x_img = x_img + x_img_2
        x_sv = x_sv + x_sv_2 + x_img_2

        x_img_3, x_sv_3 = self.Mixerlayer3(x_img, x_sv)

        # x_fuse_mid = torch.cat([x_img_3, x_sv_3], 2)
        # # print(x_img_1.shape, x_sv_1.shape, x_fuse_early.shape)
        # x_fuse_mid = self.Mixerlayer_fuse_mid(x_fuse_mid)

        x_sv_3 = x_img_3 + x_sv_3
        x_img_4, x_sv_4 = self.Mixerlayer4(x_img_3, x_sv_3)


        x_fuse_late = torch.cat([x_img_4, x_sv_4], 2)

        # print('x_fuse_late.shape:', x_fuse_late.shape)
        x_fuse_late = self.Mixerlayer_fuse_late(x_fuse_late)
        # print('x_fuse_late.shape:', x_fuse_late.shape)


        x_img = x_img + x_img_4
        x_sv = x_sv + x_sv_4

        x = torch.cat([x_img, x_sv, x_fuse_early, x_fuse_late], 2)
        # x = torch.cat([x_img, x_sv], 2)
        # print(x.shape)
        x = self.norm(x)
        x = self.pool(x)
        # print(x.shape)# [b, c]

        x = self.classifier(x) # [b, num_classes]
        # print(x.shape)
        return x



