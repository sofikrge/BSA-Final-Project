import os
import glob

# Automatically find all Python files in the `functions/` folder
module_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

for module_file in module_files:
    module_name = os.path.basename(module_file)[:-3]  # Remove ".py" extension
    if module_name != "__init__":  # Avoid importing __init__.py itself
        exec(f"from .{module_name} import *")  # Dynamically import all functions
