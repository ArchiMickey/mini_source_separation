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
    elif model_type == "bsroformer":
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
    elif model_type == "bsroformer":
        from .mss import MSSDemoCallback
        
        return MSSDemoCallback(
            **demo_config, **kwargs
        )
    elif model_type == "bsroformer_t":
        from .res_mss import MSSDemoCallback
        
        return MSSDemoCallback(
            **demo_config, **kwargs
        )

def create_validation_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'
    
    val_config = model_config.get('validation', None)
    assert val_config is not None, 'validation config must be specified in model config'
    
    if model_type == "bsroformer":
        from .mss import MSSValidateCallback
        
        return MSSValidateCallback(
            **val_config, **kwargs
        )
    elif model_type == "bsroformer_t":
        from .res_mss import MSSValidateCallback
        
        return MSSValidateCallback(
            **val_config, **kwargs
        )