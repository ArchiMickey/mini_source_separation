import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from einops import rearrange

from mss.third_party.stable_audio_tools.stable_audio_tools.models.autoencoders import create_autoencoder_from_config
from mss.third_party.stable_audio_tools.stable_audio_tools.models.transformer import ContinuousTransformer


def create_pretrained_autoencoder(path: str):
    config_path = os.path.join(path, "config.json")
    ckpt_path = os.path.join(path, "vae.ckpt")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    autoencoder = create_autoencoder_from_config(config)
    
    autoencoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return autoencoder

class LatentTransformer(nn.Module):
    def __init__(self, autoencoder_path: str, model: nn.Module):
        super().__init__()
        self.autoencoder = create_autoencoder_from_config(autoencoder_path)
        self.model = model
        
        self.autoencoder.requires_grad_(False)
    
    def forward(self, x):
        sample_size = x.shape[-1]
        if x.shape[-1] % self.autoencoder.downsampling_ratio != 0:
            x = F.pad(x, (0, self.autoencoder.downsampling_ratio - x.shape[-1] % self.autoencoder.downsampling_ratio))
        latent = self.autoencoder.encode(x)
        latent = rearrange(latent, "b d n -> b n d")
        x = self.model(latent)
        x = self.autoencoder.decode(rearrange(x, "b n d -> b d n"))[..., :sample_size]
        return x
        

if __name__ == "__main__":
    autoencoder = create_pretrained_autoencoder("/home/archimickey/Projects/mini_source_separation/pretrained/stable-audio-open-1.0")
    
    transformer = ContinuousTransformer(
        dim=128,
        depth=1,
        dim_in=64,
        dim_out=64
    )
    model = LatentTransformer(transformer, autoencoder).to("cuda:1")
    print(model)
    y = torch.randn(1, 2, 88200).to("cuda:1")
    x = torch.rand(1, 2, 88200).to("cuda:1")
    loss = F.mse_loss(model(x), y)
    loss.backward()