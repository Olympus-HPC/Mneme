from pathlib import Path
from tqdm import tqdm
import subprocess, os

from mkedit import MakefileEditor


def update_cflags(inp):
    update_map = {
        "-arch=$(ARCH)" : "--offload-arch=$(ARCH)",
        "--use_fast_math" : "-ffast-math",
        "--x" : "-x",
        "cu" : "cuda",
    }
    unsupported = [
        "--extended-lambda",
        "-Xcompiler",
        "-stdpar=gpu",
        "--expt-relaxed-constexpr",
        "-maxrregcount=32",
        "--default-stream", "per-thread",
        "-Minfo",
        "-mp=gpu",
        "-gpu=cc7",
        "-cudalib=cublas",
    ]
    required = [
        "--cuda-path=$(CUDA_HOME)",
        "--offload-arch=$(ARCH)",
        "--save-temps",
        "-I$(CUDA_HOME)/nvidia/targets/ppc64le-linux/include",
    ]

    cflags = inp.split()
    for i in reversed(range(len(cflags))):
        if cflags[i] in unsupported:
            del cflags[i]
            continue
        if cflags[i] in update_map:
            cflags[i] = update_map[cflags[i]]
        else:
            assert "arch" not in cflags[i] or "--offload-arch" in cflags[i], f"{inp}"

        if cflags[i] in required:
            required.remove(cflags[i])

    for flag in required:
        cflags.append(flag)

    return " ".join(cflags)

def update_ldflags(inp):
    update_map = {
    }
    unsupported = [
    ]
    required = [
        "-Wl,-rpath,${CUDA_HOME}/nvidia/lib64",
        "-L${CUDA_HOME}/nvidia/lib64",
        "-lcuda",
        "-lcudadevrt",
        "-lcudart_static",
        "-lrt",
        "-lpthread",
        "-ldl",
    ]
    ldflags = inp.split()
    for i in reversed(range(len(ldflags))):
        if ldflags[i] in unsupported:
            del ldflags[i]
            continue
        if ldflags[i] in update_map:
            ldflags[i] = update_map[ldflags[i]]

        if ldflags[i] in required:
            required.remove(ldflags[i])
    
    for flag in required:
        ldflags.append(flag)

    return " ".join(ldflags)

dataset = "./src/"
ds_path =  Path(dataset)
datapoints = list(ds_path.glob("*-cuda"))

for data in datapoints:
    dir = data
    if dir.name == "daphne-cuda":
        dir = dir / "src" / "points2image"
    elif dir.name in ["local-ht-cuda", "mf-sgd-cuda", "miniFE-cuda", "slu-cuda"]:
        dir = dir / "src"
    elif dir.name == "logic-resim-cuda":
        dir = dir / "Simulation"
    elif dir.name == "snicit-cuda":
        dir = dir / "bin"
    elif dir.name in ["dwconv1d-cuda", "hpl-cuda", "si-cuda"]:
        continue # FIXME: these either don't have a Makefile or their Makefile or their Makefiles aren't proper format

    if dir.name == "lsqt-cuda":
        makefile = dir / "Makefile.gpu"
    else:
        makefile = dir / "Makefile"

    assert makefile.exists(), f"{makefile}"

    # FIXME: handle these later:
    # ans-cuda ['NVCC_FLAGS']
    # convolutionDeformable-cuda []
    # halo-finder-cuda:
    #           ['NVCC_FLAGS', 'HACC_CFLAGS', 'HACC_CXXFLAGS', 'HACC_LDFLAGS', 'HACC_MPI_CFLAGS', 'HACC_MPI_CXXFLAGS', 'HACC_MPI_LDFLAGS']
    #           include uses variables
    # kmeans-cuda ['CFLAGS']
    # lsqt-cuda ['CFLAGS']
    # minimap2-cuda ['COMPILE_FLAGS', 'NVCC_COMPILE_FLAGS']

    if dir.name in ["halo-finder-cuda", "minimap2-cuda", "kmeans-cuda", "lsqt-cuda", "kmeans-cuda", "ans-cuda", "convolutionDeformable-cuda"]:
        continue
    

    if data.name in [
        "addBiasQKV-cuda", 
        "addBiasResidualLayerNorm-cuda",
        "allreduce-cuda",                 # mpi
        "assert-cuda",                    # probably save-temps issue
        "axhelm-cuda",                    # "dfloat" type not found
        "bicgstab-cuda",                  # requires cusparseSpSVDescr_t
        "blas-fp8gemm-cuda",              # 'cuda_fp8.h' file not found
        "blas-gemmEx-cuda",               # 'cublasComputeType_t' type not found
        "blas-gemmEx2-cuda",              # 'cublasComputeType_t' type not found
        "blas-gemmStridedBatched-cuda",   # 'reference.h' not found
        "bmf-cuda",                       
        "bm3d-cuda",                      # clang frontend error
        "ccl-cuda",                       # 'nccl.h' file not found
        "conversion-cuda",                # tries static cast of float to char
        "convolution3D-cuda",             # 'cudnn.h' file not found
        "diamond-cuda",                   # undefined reference error
        "dp-cuda",                        # 'execution' file not found
        "dwconv-cuda",                    # 'tensorAccessor.h' file not found
        "f16atomic-cuda",                 # use of undeclared identifier '__ushort_as_bfloat16'
        "fresnel-cuda",                   # Unresolved extern function '_Z21Fresnel_Sine_Integrald'
        "gels-cuda",                      # use of undeclared identifier 'cublasGetStatusString'
        "gelu-cuda",                      # no matching function for call to '__hadd2'
        "gerbil-cuda",                    # too many errors
        "gpp-cuda",                       # too many errors
        "heat2d-cuda",                    # unknown type name 'vector' (altivec.h)
        "interval-cuda",                  # Unresolved extern function '_ZL3nanPKc'
        "intrinsics-simd-cuda",           # undeclared identifier '__viaddmax_s16x2'
        "kmc-cuda",                       # cuda bitcode file not generated
        "layernorm-cuda",                 # undeclared identifier '__stcs'
        "lda-cuda",                       # Cannot select: intrinsic %llvm.nvvm.shfl.down.f32 (Has CUDA .bc file)
        "logan-cuda",                     # linker error
        "lr-cuda",                        # 'hip/hip_runtime.h' file not found
        "ludb-cuda",                      # undeclared identifier 'cublasGetStatusString'
        "merkle-cuda",                    # linker error
        "mf-sgd-cuda",                    # Cannot select: intrinsic %llvm.nvvm.shfl.down.f32 (Has CUDA .bc file)
        "miniDGS-cuda",                   # 'mpi.h' not found
        "miniFE-cuda",                    # save-temps not working
        "miniWeather-cuda",               # 'mpi.h' not found
        "minmax-cuda",                    # consume_tile no matching member call
        "mmcsf-cuda",                     # 'boost/sort/sort.hpp' file not found
        "nlll-cuda",                      # no viable overloaded '+='
        "permute-cuda",                   # unknown type name 'cublasComputeType_t'
        "pingpong-cuda",                  # 'mpi.h' not found
        "qkv-cuda",                       # unknown type name 'cublasComputeType_t'
        "sad-cuda",                       # no rule to make target main.o
        "saxpy-ompt-cuda",                # undefined reference to `cublasCreate_v2'
        "seam-carving-cuda",              # no rule to make target main.o
        "shmembench-cuda",                # undefined reference to `__cudaPopCallConfiguration'
        "slit-cuda",                      # no rule to make target main.o
        "slu-cuda",                       # 'nicslu.h' file not found
        "sobol-cuda",                     # ptxas: File uses too much global constant data
        "sparkler-cuda",                  # 'mpi.h' not found
        "spsm-cuda",                      # error: unknown type name 'cusparseSpSMDescr_t'
        "ssim-cuda",                      # error: no member named 'trilinear_kernel' in namespace 'util'
        "stsg-cuda",                      # 'gdal/gdal_priv.h' file not found
        "testSNAP-cuda",                  # 'refdata_2J14_W.h' file not found
        "wmma-cuda",                      # '-' is ambiguous for '__half'
        "xlqc-cuda",                      # too many linker errors (Has CUDA .bc file)
        "xsbench-cuda",                   # __device__ redefined
        ]:
        continue

    mkedit = MakefileEditor(makefile)
    mkedit.set_variable("CXX",  "clang++", add_if_new=False)
    mkedit.set_variable("CC",   "clang++", add_if_new=False)
    mkedit.set_variable("NVCC", "clang++", add_if_new=False)

    if data.name in [
        "all-pairs-distance-cuda", "accuracy-cuda", "attention-cuda", "channelSum-cuda",                
        "determinant-cuda", "f16sp-cuda", "kurtosis-cuda", "laplace-cuda", "lzss-cuda",                      
        "logic-rewrite-cuda", "mtf-cuda",  "multinomial-cuda", "nonzero-cuda",                   
        "nosync-cuda", "radixsort2-cuda", "remap-cuda", "rle-cuda", "sa-cuda",                        
        "scan3-cuda", "scel-cuda", "segment-reduce-cuda", "sort-cuda", "sortKV-cuda",                    
        "tsne-cuda", "wordcount-cuda"]:
        mkedit.set_variable("ARCH", "sm_60", add_if_new=True)    # sm_70 unsupported
    else:
        mkedit.set_variable("ARCH", "sm_70", add_if_new=True)


    # NOTE: "kmeans-cuda" doesn't have LDFLAGS but it uses it during linking
    nvccflags = mkedit.get_variable("NVCCFLAGS")

    if nvccflags is None:
        cflags = mkedit.get_variable("CFLAGS")
        assert cflags is not None
        mkedit.set_variable("CFLAGS", update_cflags(cflags))
    else:
        mkedit.set_variable("NVCCFLAGS", update_cflags(nvccflags))

    cxxflags = mkedit.get_variable("CXXFLAGS")
    if cxxflags:
        mkedit.set_variable("CXXFLAGS", update_cflags(cxxflags))

    ldflags = mkedit.get_variable("LDFLAGS")
    assert ldflags is not None
    mkedit.set_variable("LDFLAGS", update_ldflags(ldflags))

    mkclang = makefile.with_suffix(".clang")
    
    if data.name == "b+tree-cuda":
        mkedit.get_targets()["b+tree.out"][-1].cmds[-1] += " $(LDFLAGS)"
    elif data.name == "bmf-cuda":
        mkedit.set_variable("GPU", "true")
    elif data.name == "heat2d-cuda":
        cflags = mkedit.get_variable("CFLAGS")
        mkedit.set_variable("CFLAGS", cflags + " -DNO_WARN_X86_INTRINSICS")
    elif data.name == "leukocyte-cuda":
        cmd1 = mkedit.get_targets()["$(MATRIX_DIR)/meschach.a"][-1].cmds[-1]
        mkedit.get_targets()["$(MATRIX_DIR)/meschach.a"][-1].cmds[-1] = cmd1.replace("make", "make -f makefile.clang")

        meschach_mkfile = makefile.parent / mkedit.get_variable("MATRIX_DIR") / "makefile"
        meschach_mkedit = MakefileEditor(meschach_mkfile)
        cflags = meschach_mkedit.get_variable("CFLAGS")
        meschach_mkedit.set_variable("CFLAGS", update_cflags(cflags))
        meschach_mkedit.set_variable("ARCH", "sm_70", add_if_new=True)
        meschach_mkedit.output_makefile(meschach_mkfile.with_suffix(".clang"))
    elif data.name == "saxpy-ompt-cuda":
        mkedit.set_variable("DEVICE", "cuda")


    mkedit.set_variable("OPTIMIZE", "false") # We don't want -O3 optimization
    cflags = mkedit.get_variable("CFLAGS")
    if data.name in [
        'aidw-cuda', 'bitcracker-cuda', 'dslash-cuda', 'frna-cuda',
        'grrt-cuda', 'haversine-cuda', 'henry-cuda', 'langevin-cuda',
        'minkowski-cuda', 'mriQ-cuda', 'prna-cuda', 'perplexity-cuda',
        'scel-cuda', 'stddev-cuda', 'svd3x3-cuda', 'tsp-cuda', 'cooling-cuda',
        'gabor-cuda', 'logic-rewrite-cuda', 'logprob-cuda', 'meanshift-cuda',
        'p4-cuda', 'softmax-cuda', 'softmax-fused-cuda', 'tsne-cuda'
        ]:
        cflags += " -ffast-math"

    mkedit.set_variable("CFLAGS", cflags)


    # elif data.name == "gerbil-cuda":
    #     cflags = mkedit.get_variable("CFLAGS")
    #     cflags = cflags.replace("-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/include", "-I$(CUDA_HOME)/nvidia/targets/ppc64le-linux/include")
    #     mkedit.set_variable("CFLAGS", cflags + " -DNO_WARN_X86_INTRINSICS")
    mkedit.output_makefile(mkclang)

    env = os.environ.copy()
    env["VERBOSE"] = "1"
    builddir = mkclang.parent
    cmd = ["make", "-f", f"{mkclang.resolve()}"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE, cwd=builddir, env=env)
    retcode = proc.returncode

    bc_files = list(builddir.glob("*-cuda-nvptx64-nvidia-cuda-sm_*.bc"))
    print("\t", data.name, len(bc_files))

    if retcode != 0:
        print(f"  Return Code: {retcode} \t\t Has cuda bc: {len(bc_files) > 0}\n\ncd {builddir.resolve()} && {' '.join(cmd)} \n\n-------------------------------- LOG -----------------------------\n\n\n{proc.stdout.decode()}\n\n--------------------\n\n{proc.stderr.decode()}")
    if len(bc_files) == 0:
        print( f"\n\ncd {builddir.resolve()} && {' '.join(cmd)}\n{list(builddir.glob('*'))}\n{proc.stderr.decode()}")

