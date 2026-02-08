import os

def get_files_info(working_directory, directory="."):
    working_dir_abs = os.path.abspath(working_directory)
    target_dir = os.path.normpath(os.path.join(working_dir_abs, directory))
    if os.path.commonpath([working_dir_abs, target_dir]) != working_dir_abs:
        return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'
    if not os.path.isdir(target_dir):
        return f'Error: "{directory}" is not a directory'

    result = f"Result for {'current' if directory == "." else f"'{directory}'"} directory:"
    try:
        for item in os.listdir(target_dir):
            result += f"\n- {item}: file_size: {os.path.getsize(os.path.normpath(os.path.join(target_dir, item)))} bytes, is_dir={os.path.isdir(os.path.normpath(os.path.join(target_dir, item)))}"
        return result
    except Exception as e:
        return f"Error: An error occurred reading the directory contents: {e}"