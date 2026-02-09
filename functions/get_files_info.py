import os

def get_files_info(working_directory, directory="."):
    try: # Moved this up to top because some of the initial OS function calls can raise exceptions/
        working_dir_abs = os.path.abspath(working_directory)
        target_dir = os.path.normpath(os.path.join(working_dir_abs, directory))
        if os.path.commonpath([working_dir_abs, target_dir]) != working_dir_abs:
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'
        if not os.path.isdir(target_dir):
            return f'Error: "{directory}" is not a directory'

        result = f"Result for {'current' if directory == "." else f"'{directory}'"} directory:"
        files_info = [] # preferable to concatenating a string in each loop since string is immutable and a new one is created each time.
        for filename in os.listdir(target_dir):
            files_info.append(f"\n- {filename}: file_size: {os.path.getsize(os.path.join(target_dir, filename))} bytes, is_dir={os.path.isdir(os.path.normpath(os.path.join(target_dir, filename)))}")
        return "".join(files_info)
    except Exception as e:
        return f"Error: An error occurred reading the directory contents: {e}"