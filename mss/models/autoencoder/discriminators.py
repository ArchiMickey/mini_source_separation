import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Resample

from .stable_audio_tools.models.discriminators import get_hinge_losses

from typing import List, Tuple
import typing


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        discriminator_channel_mult,
        period: List[int],
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        self.d_mult = discriminator_channel_mult
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        in_channels,
                        int(32 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(32 * self.d_mult),
                        int(128 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(128 * self.d_mult),
                        int(512 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(512 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(1024 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        1,
                        padding=(2, 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, mpd_reshapes, discriminator_channel_mult, use_spectral_norm):
        super().__init__()
        self.mpd_reshapes = mpd_reshapes
        print(f"mpd_reshapes: {self.mpd_reshapes}")
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(in_channels, discriminator_channel_mult, rs, use_spectral_norm=use_spectral_norm)
                for rs in self.mpd_reshapes
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, logits_fake, feature_true, feature_fake = self(y, x)
        
        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)
        
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss, adv_loss, feature_matching_distance

class DiscriminatorCQT(nn.Module):
    def __init__(self, 
                 filters: int, 
                 max_filters: int, 
                 filters_scale: int, 
                 dilations: List[int], 
                 in_channels: int, 
                 out_channels: int, 
                 sampling_rate: int, 
                 hop_length: int, 
                 n_octaves: int, 
                 bins_per_octave: int, 
                 cqtd_normalize_volume: bool = False):
        super().__init__()

        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sampling_rate
        self.cqtd_normalize_volume = cqtd_normalize_volume
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = cqtd_normalize_volume
        if self.cqtd_normalize_volume:
            print(
                f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def get_2d_padding(
        self,
        kernel_size: typing.Tuple[int, int],
        dilation: typing.Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)
        
        if x.shape[1] == 2:
            lz = self.cqt_transform(x[:, :1])
            rz = self.cqt_transform(x[:, 1:2])
            z_amplitude = torch.stack([lz[:, :, :, 0], rz[:, :, :, 0]], dim=1)
            z_phase = torch.stack([lz[:, :, :, 1], rz[:, :, :, 1]], dim=1)
        else:
            z = self.cqt_transform(x)
            z_amplitude = z[:, :, :, 0].unsqueeze(1)
            z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, 
                 filters: int = 32, 
                 max_filters: int = 1024, 
                 filters_scale: int = 1, 
                 dilations: List[int] = [1, 2, 4], 
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 hop_lengths: List[int] = [512, 256, 256], 
                 n_octaves: List[int] = [9, 9, 9], 
                 bins_per_octaves: List[int] = [24, 36, 48]):
        super().__init__()

        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.dilations = dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hop_lengths = hop_lengths
        self.n_octaves = n_octaves
        self.bins_per_octaves = bins_per_octaves

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    filters=self.filters,
                    max_filters=self.max_filters,
                    filters_scale=self.filters_scale,
                    dilations=self.dilations,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    sampling_rate=44100,  # Assuming a default sampling rate
                    hop_length=self.hop_lengths[i],
                    n_octaves=self.n_octaves[i],
                    bins_per_octave=self.bins_per_octaves[i],
                )
                for i in range(len(self.hop_lengths))
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, logits_fake, feature_true, feature_fake = self(y, x)
        
        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)
        
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss, adv_loss, feature_matching_distance

class CombinedDiscriminator(nn.Module):
    def __init__(self, discriminator_configs: List[dict]):
        super().__init__()
        self.discriminators = nn.ModuleList([])
        for cfg in discriminator_configs:
            if cfg["type"] == "mscqt":
                self.discriminators.append(MultiScaleSubbandCQTDiscriminator(**cfg["config"]))
            elif cfg["type"] == "mpd":
                self.discriminators.append(MultiPeriodDiscriminator(**cfg["config"]))
            else:
                raise NotImplementedError(f"Discriminator type {cfg['type']} not implemented!")
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for d in self.discriminators:
            y_d_r, y_d_g, fmap_r, fmap_g = d(y, y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def loss(self, x, y):
        dis_losses = []
        adv_losses = []
        feature_matching_distances = []
        
        for d in self.discriminators:
            dis_loss, adv_loss, feature_matching_distance = d.loss(x, y)
            dis_losses.append(dis_loss)
            adv_losses.append(adv_loss)
            feature_matching_distances.append(feature_matching_distance)

        return sum(dis_losses), sum(adv_losses), sum(feature_matching_distances)