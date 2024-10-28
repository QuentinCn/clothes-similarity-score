import os
import pydoc

# Set the path to the folder containing your Python modules
folder_path = './'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.py'):  # Check if the file is a Python file
        module_name = filename[:-3]  # Remove the .py extension
        print(f'Generating documentation for {module_name}')
        pydoc.writedoc(module_name)
