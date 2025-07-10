from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

# Import all modules to ensure decorators are executed
for module_name in __all__:
    try:
        exec(f"from . import {module_name}")
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}")
