from hydra.utils import instantiate


def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'
    
    if model_type in ['diffusion_cond', 'diffusion_cond_inpaint, diffusion_prior', 'diffusion_mss']:
        from mss.models.autoencoder.stable_audio_tools.models.diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)

    elif model_type == 'autoencoder':
        from mss.models.autoencoder.stable_audio_tools.models.autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)

    elif model_type == 'autoencoder_v2':
        from mss.models.autoencoder.autoencoders_v2 import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)
    
    else:
        return instantiate(model_config["model"])