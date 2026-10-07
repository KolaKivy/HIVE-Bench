"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module registers its constructor with FRAMEWORK_REGISTRY.
"""

import pkgutil
import importlib
from ._framework_names import normalize_framework_config
from hivebench.model.tools import FRAMEWORK_REGISTRY

from hivebench.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    try:
        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            importlib.import_module(f"{__name__}.{module_name}")
    except Exception as e:
        logger.warning(f"Warning: Failed to auto-import framework submodules: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: registered framework identifier
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"):
        raise ValueError("Configuration must define framework.name")

    normalize_framework_config(cfg)
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented. Available frameworks are registered from Policy/hivebench/model/framework.")
    
    MODEL_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODEL_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
