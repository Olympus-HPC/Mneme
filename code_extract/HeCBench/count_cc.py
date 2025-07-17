import re
from pathlib import Path
import subprocess as sb
import os
import argparse

from tqdm import tqdm

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

    temp_dir = Path(args.temp_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()

    kernel_functions = search_kernel_functions(source_dir)

    os.chdir(temp_dir)

    linked = 0
    compiled = 0
    total= 0

    for file, funcs in tqdm(kernel_functions.items()):
        # print(f"Functions: {funcs}")
        # cc_path = file.parent / 'compile_commands.json'
        # # if not cc_path.exists():
        # cmd = ['bear', '--', 'make', '-f', 'Makefile.clang']
        # os.chdir(file.parent)
        # sb.run(['make', 'clean'], stdout=sb.DEVNULL, stderr=sb.DEVNULL)
        # sb.run(cmd, stdout=sb.DEVNULL, stderr=sb.DEVNULL)
        # assert os.path.isfile(cc_path), f"CC does not exist for: {file.parent}"
        # assert os.path.getsize(cc_path) > 2, f"CC is empty for: {file.parent}"
        # os.chdir(temp_dir)
        for func in funcs:
            cmd = ['../code-extract', '--emitAllDecls', str(file.parent), func]
            res = sb.run(cmd, stderr=sb.PIPE, stdout=sb.PIPE, text=True)

            if "Compilation successful!" in res.stdout:
                compiled += 1
            if "Linking successful!" in res.stdout:
                linked += 1

            total += 1
        #     if "Linking successful!" not in res.stdout:
        #         print(res.stderr)
                # exit()

    print("Total kernels: ", total)
    print("Compiled kernels: ", compiled)
    print("Linked kernels: ", linked)

if __name__ == "__main__":
    main()
