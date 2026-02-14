import os
import subprocess
from google.genai import types

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

schema_run_python_file = types.FunctionDeclaration(
    name="run_python_file",
    description="Runs a Python file within the working directory",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the Python file to run, relative to the working directory",
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                description="Optional list of arguments to pass to the Python script",
                items=types.Schema( # Figured this part on my own, from Google search https://discuss.ai.google.dev/t/function-calling-issues-with-type-object-and-array/34581
                    type=types.Type.STRING
                )
            ),
        },
        required=["file_path"]
    ),
)