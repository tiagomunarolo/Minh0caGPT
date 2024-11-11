import os
from pathlib import Path


def wire_all_modules(current_module: str) -> list[str]:
    """
    Dynamically wires all modules in specified subdirectories by scanning
    the package and importing all Python files.
    """
    path_root = Path(current_module).parent
    dirs_to_link = {'tokenizer', 'data_preparation', 'models'}
    modules_to_wire = []

    # Scan the specified directories for modules
    for root, dirs, _ in os.walk(path_root):
        dir_name = os.path.basename(root)
        if dir_name in dirs_to_link:
            # Collect Python files in the current directory, excluding __init__.py and private files
            python_files = [
                f.replace('.py', '').replace(os.sep, '.')
                for f in os.listdir(root)
                if f.endswith('.py') and not f.startswith('__')
            ]
            # Append the module names
            modules_to_wire.extend(f'{dir_name}.{file}' for file in python_files)

    return modules_to_wire
