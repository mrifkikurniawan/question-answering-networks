from easydict import EasyDict as edict
from importlib import import_module

__all__ = ["initialize_dataset", "initialize_optimizer", "initialize_model"]


def initialize_model(module: object, model: str, **args):    
    print(f"initializing model {model}")

    args = edict(args)
    model_ = getattr(module, model)
    model_ = model_(**args)
    return model_

def initialize_dataset(module: object, dataset: str, **args):    
    print(f"initializing dataset {dataset}")

    args = edict(args)
    dataset_ = getattr(module, dataset)
    dataset_ = dataset_(**args)
    return dataset_

def initialize_optimizer(module: object, method: str):
    print(f"Initializing optimizer {method}")

    optimizer = getattr(module, method)
    return optimizer

def create_instance(module_cfg: edict, **kwargs):
    module = module_cfg.module
    method = module_cfg.method
    module_args = module_cfg.args
    module_args.update(kwargs)
    
    print(f"Initializing {module}.{method}")
    module = import_module(module)
    module = getattr(module, method)
    
    try:
        instance = module(**module_args)
    except:
        instance = [module(module_args)]
        
    return instance

def initialize_pretrained(module_cfg: edict, **kwargs):
    module = module_cfg.module
    method = module_cfg.method
    module_args = module_cfg.args
    print(f"Initializing {module}.{method}")
    
    module = import_module(module)
    method_obj = getattr(module, method)
    return method_obj.from_pretrained(**module_args)