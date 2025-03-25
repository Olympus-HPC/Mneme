import re
from pathlib import Path
import subprocess as sb
import os

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
                    function_names[Path(file)] = set(matches)

    return function_names

directory = '/home/grimmy/Mneme/code_extract/tests/HeCBench_cuda'  

kernel_functions = search_kernel_functions(directory)

os.chdir('/home/grimmy/Mneme/code_extract/build/temp')

total_supported = 0
total_compiled = 0
total_missing_features = 0
total_bench = 0
compiled_bench = 0

for file, funcs in kernel_functions.items():
    cc_path = file.parent / 'compile_commands.json'
    if not cc_path.exists() or ('ge-spmm-cuda' in str(file)):
        continue
    tmp = total_compiled
    for func in funcs:
        cmd = ['../code-extract', file.parent, func, '-emitAllDecls']
        result = sb.run(cmd, stdout=sb.DEVNULL, stderr=sb.PIPE, text=True)
        if result.returncode != 0:
            # print(result.stderr)
            continue
        else:
            total_supported += 1
            if 'Compilation failed!' in result.stderr:
                if 'linker command failed with exit code' in result.stderr:
                    # print("Compiled")
                    total_compiled += 1
                # print(result.stderr)
            elif 'Compilation successful!'  in result.stderr:
                print("Linked")
                total_compiled += 1
            else:
                # print("NA")
                total_missing_features += 1
    total_bench += 1
    if total_compiled != tmp:
        compiled_bench += 1
    print(file.parent)
    print("Total: ", total_supported, " Compiled: ", total_compiled, " Total missing: ", total_missing_features)
print(" Total bench: ", total_bench, " Total compiled: ", compiled_bench)