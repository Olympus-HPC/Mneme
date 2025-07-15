import re
from pathlib import Path
import subprocess as sb
import os
import argparse

def search_kernel_functions(directory_path):
    directory = Path(directory_path)
    file_exts = ['*.cu', '*.cpp', '*.h']

    func_pattern = re.compile(r'\b__global__\s+\w+\s+(\w+)\s*\(.*?\)\s*\{', re.DOTALL)
    function_names = {}

    for ext in file_exts:
        for file in directory.rglob(ext):
            with open(file, 'r', encoding='utf8', errors='ignore') as f:
                matches = func_pattern.findall(f.read())
                if matches:
                    function_names[Path(file).expanduser().resolve()] = set(matches)

    return function_names

def main():
    parser = argparse.ArgumentParser(description="Search for CUDA kernel functions and extract code.")
    parser.add_argument('source_dir', help='Directory to search for source files')
    parser.add_argument('temp_dir', help='Path to temp folder (for chdir and code-extract)')
    args = parser.parse_args()

    temp_dir = os.path.expanduser(args.temp_dir)
    source_dir = os.path.expanduser(args.source_dir)

    kernel_functions = search_kernel_functions(source_dir)

    os.chdir(temp_dir)

    for file, funcs in kernel_functions.items():
        print(f"File: {file}")
        print(f"Functions: {funcs}")
        cc_path = file.parent / 'compile_commands.json'
        if not cc_path.exists():
            cmd = ['bear', '--', 'make', '-f', 'Makefile.clang']
            os.chdir(file.parent)
            sb.run(['make', 'clean'])
            sb.run(cmd)
            os.chdir(temp_dir)
        for func in funcs:
            cmd = ['../code-extract', str(file.parent), func]
            sb.run(cmd)
        break

if __name__ == "__main__":
    main()
