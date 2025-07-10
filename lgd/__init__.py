# Import all config modules to register them with GraphGym
from .config import *

# Import all other modules to make them available
from . import (
    act,
    asset,
    ddpm,
    encoder,
    head,
    layer,
    loader,
    loss,
    model,
    optimizer,
    train,
    transform,
)

# Make sure all config modules are imported and registered
from .config import (
    custom_gnn_config,
    data_preprocess_config,
    dataset_config,
    defaults_config,
    gt_config,
    mlflow_config,
    optimizers_config,
    posenc_config,
    pretrained_config,
    split_config,
    wandb_config,
)

# Import utils
from . import utils