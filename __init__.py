import os
import sys
from .f_lite.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, NODE_DESCRIPTION_MAPPINGS


import importlib.util
import sys


# Ensuring that an import of f_lite.FLitePipeline will work, as this is required to 
# load the model using the diffusers library

custom_node_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
for module_name in ["model", "pipeline", "comfyui"]:
    # Get the current package name
    current_package = __name__.split('.')[0]
    
    # Construct the full module path
    spec = importlib.util.find_spec(f'.f_lite.{module_name}', package=current_package)

    # Create the module
    module = importlib.util.module_from_spec(spec)
    # Execute the module
    spec.loader.exec_module(module)
    # Inject the module into f_lite
    sys.modules[f'f_lite.{module_name}'] = module
    # Also make it accessible as f_lite.model
    if 'f_lite' in sys.modules:
        setattr(sys.modules['f_lite'], module_name, module)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "NODE_DESCRIPTION_MAPPINGS"]
