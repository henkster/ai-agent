import os
import subprocess
from subprocess import PIPE

def run_python_file(working_directory, file_path, args=None):
    try:
        working_dir_abs = os.path.abspath(working_directory)
        target_path = os.path.normpath(os.path.join(working_dir_abs, file_path))
        if os.path.commonpath([working_dir_abs, target_path]) != working_dir_abs:
            return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'
        if not os.path.isfile(target_path):
            return f'Error: "{file_path}" does not exist or is not a regular file'
        if not target_path.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file'
        
        command = ["python", target_path]
        if not args is None:
            command.extend(args)
        completed_process = subprocess.run(command, capture_output=True, text=True, timeout=30)

        result = []

        if completed_process.returncode != 0:
            result.append(f"Process exited with code {completed_process.returncode}")
        if completed_process.stdout is None and completed_process.stderr is None:
            result.append("No output produced")
        if completed_process.stdout:
            result.append(f"STDOUT: {completed_process.stdout}")
        if completed_process.stderr:
            result.append(f"STDERR: {completed_process.stderr}")
        return "\n".join(result)

    except Exception as e:
        return f'Error: executing Python file: {e}'