from easydict import EasyDict as edict

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