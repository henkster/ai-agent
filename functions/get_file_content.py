import os
from config import MAX_CHARS

def get_file_content(working_directory, file_path):
    working_dir_abs = os.path.abspath(working_directory)
    target_path = os.path.normpath(os.path.join(working_dir_abs, file_path))
    if os.path.commonpath([working_dir_abs, target_path]) != working_dir_abs:
        return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'
    if not os.path.isfile(target_path):
        return f'Error: "{file_path}" is not a valid file.'
    
    content = ""
    try:    
        with open(target_path, "r") as f:
            content += f.read(MAX_CHARS)
            if f.read(1):
                content += f'[...File "{file_path}" truncated at {MAX_CHARS} characters]'
        return content
    except Exception as e:
        return f'Error: Problem reading "{file_path}": {e}'