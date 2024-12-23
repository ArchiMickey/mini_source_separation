import torch
from torch.nn import Parameter        
from ..models.factory import create_model_from_config

def create_training_wrapper_from_config(model_config, model):
    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'
    model_type = model_config.get('model_type', None)
    if model_type in ["diffusion_cond", "diffusion_mss"]:
        from .diffusion_mss import DiffusionMSSTrainingWrapper
        return DiffusionMSSTrainingWrapper(
            model,
            target_instrument=training_config["target_instrument"],
            lr=training_config.get("lr", None),
            use_ema=training_config.get("use_ema", True),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler=training_config.get("timestep_sampler", "logit_normal"),
        )
    elif model_type in ['autoencoder', 'autoencoder_v2']:
        from .autoencoders import AutoencoderTrainingWrapper
        
        ema_copy = None

        if training_config.get("use_ema", False):
            ema_copy = create_model_from_config(model_config)
            ema_copy = create_model_from_config(model_config) # I don't know why this needs to be called twice but it broke when I called it once
            # Copy each weight to the ema copy
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        use_ema = training_config.get("use_ema", False)

        latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)

        teacher_model = training_config.get("teacher_model", None)
        if teacher_model is not None:
            teacher_model = create_model_from_config(teacher_model)
            teacher_model = teacher_model.eval().requires_grad_(False)

            teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
            if teacher_model_ckpt is not None:
                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
            else:
                raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")
        
        return AutoencoderTrainingWrapper(
            model,
            lr=training_config["learning_rate"],
            clip_grad_norm=training_config.get("clip_grad_norm", 0.0),
            warmup_steps=training_config.get("warmup_steps", 0), 
            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
            sample_rate=model_config["sample_rate"],
            loss_config=training_config.get("loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=use_ema,
            ema_copy=ema_copy if use_ema else None,
            force_input_mono=training_config.get("force_input_mono", False),
            latent_mask_ratio=latent_mask_ratio,
            teacher_model=teacher_model
        )
    
    elif model_type in ["bsroformer", "latentmss_transformer"]:
        from .mss import MSSTrainingWrapper

        return MSSTrainingWrapper(
            model,
            lr=training_config.get("lr", None),
            sample_rate=model_config["sample_rate"],
            mono=model_config["mono"],
            use_ema=training_config.get("use_ema", True),
            loss_configs=training_config.get("loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
        )
    
    elif model_type == "bsroformer_t":
        from .res_mss import ShiftMSSTrainingWrapper

        return ShiftMSSTrainingWrapper(
            model,
            lr=training_config.get("lr", None),
            sample_rate=model_config["sample_rate"],
            mono=model_config["mono"],
            use_ema=training_config.get("use_ema", True),
            optimizer_configs=training_config.get("optimizer_configs", None),
        )
    
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

def create_demo_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'
    
    demo_config = model_config.get('demo', None)
    assert demo_config is not None, 'demo config must be specified in model config'
    if model_type in ["diffusion_cond", "diffusion_mss"]:
        from .diffusion_mss import DiffusionMSSDemoCallback
        return DiffusionMSSDemoCallback(
            **demo_config, **kwargs
        )
    elif model_type in ["autoencoder", "autoencoder_v2"]:
        from .autoencoders import AutoencoderDemoCallback
        return AutoencoderDemoCallback(
            **demo_config, **kwargs
        )
    elif model_type in ["bsroformer", "latentmss_transformer"]:
        from .mss import MSSDemoCallback
        
        return MSSDemoCallback(
            **demo_config, **kwargs
        )
    elif model_type == "bsroformer_t":
        from .res_mss import MSSDemoCallback
        
        return MSSDemoCallback(
            **demo_config, **kwargs
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

def create_validation_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'
    
    val_config = model_config.get('validation', None)
    assert val_config is not None, 'validation config must be specified in model config'
    
    if model_type in ["bsroformer", "latentmss_transformer"]:
        from .mss import MSSValidateCallback
        
        return MSSValidateCallback(
            **val_config, **kwargs
        )
    elif model_type == "bsroformer_t":
        from .res_mss import MSSValidateCallback
        
        return MSSValidateCallback(
            **val_config, **kwargs
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")