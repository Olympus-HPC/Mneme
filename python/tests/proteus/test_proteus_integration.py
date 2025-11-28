import pytest

from mneme.proteus import jit
from mneme.llvm.module import ModuleRef, parse_assembly
from mneme.llvm.buffer import MemBufferRef
from ctypes import POINTER, c_bool, c_char, c_char_p, c_int, c_uint, c_uint64, c_void_p
import ctypes
from mneme.llvm import ffi
from mneme.mneme_types import dim3


def test_specialize_args():
    # 1. Construct minimal IR
    IR = r"""
; ModuleID = 'test'
target triple = "amdgcn-amd-amdhsa"


define amdgpu_kernel void @kernel_add(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  ; get thread id
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  ; compare tid > 512
  %cmp = icmp ugt i32 %tid, 512
  br i1 %cmp, label %early_return, label %continue

early_return:
  ret void

continue:
  ; original kernel body
  %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
  %old = load i32, i32 addrspace(1)* %elem
  %new = add i32 %old, %val
  store i32 %new, i32 addrspace(1)* %elem
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
"""

    mod = parse_assembly(IR)

    # 2. Initial prune
    jit.pruneIR(mod)

    # 3. Specialize argument %val = 7
    val = c_int(7)
    val_ptr = ctypes.byref(val)
    KernelArgs = (c_void_p * 2)()
    KernelArgs[0] = c_void_p()  # first arg not specialized
    KernelArgs[1] = c_void_p(ctypes.addressof(val))

    mod_hash = 0
    new_hash = jit.specialize_args(
        mod,
        mod_hash,
        "kernel_add",
        KernelArgs,
        num_args=2,
        specialize_indexes=[1],  # specialize %val
    )

    assert new_hash != mod_hash

    assert "%new = add i32 %old, 7" in str(mod)


def test_specialize_dims_assume():
    # 1. Construct minimal IR
    IR = r"""
; ModuleID = 'test'
target triple = "amdgcn-amd-amdhsa"


define amdgpu_kernel void @kernel_add(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  ; get thread id
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  ; compare tid > 512
  %cmp = icmp ugt i32 %tid, 512
  br i1 %cmp, label %early_return, label %continue

early_return:
  ret void

continue:
  ; original kernel body
  %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
  %old = load i32, i32 addrspace(1)* %elem
  %new = add i32 %old, %val
  store i32 %new, i32 addrspace(1)* %elem
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
"""

    mod = parse_assembly(IR)
    old_mod = str(mod)

    print("before", mod)

    block = dim3(4, 4, 4)
    grid = dim3(16, 16, 16)

    mod_hash = 0
    new_hash = jit.specialize_dims_assume(mod, mod_hash, "kernel_add", grid, block)

    assert new_hash != mod_hash
    print("After", mod)
    assert "%tid = call i32 @llvm.amdgcn.workitem.id.x(), !range !0" in str(mod)
    assert "!0 = !{i32 0, i32 4}" in str(mod)


def test_specialize_dims():
    # 1. Construct minimal IR
    IR = r"""
; ModuleID = 'test'
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @kernel_add(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  ; threadIdx.x
  %tid_x = call i32 @llvm.amdgcn.workitem.id.x()

  ; blockDim.x / y / z
  %bdx = call i32 @llvm.amdgcn.workgroup.size.x()
  %bdy = call i32 @llvm.amdgcn.workgroup.size.y()
  %bdz = call i32 @llvm.amdgcn.workgroup.size.z()

  ; ------ compute global linear thread id ------
  ; global_tid = tid_x + blockIdx.x * blockDim.x
  ; but since we don't have blockIdx here, we just fold dims into tid

  ; compute  threads_per_block = blockDim.x * blockDim.y * blockDim.z
  %tmp1 = mul i32 %bdx, %bdy
  %threads_per_block = mul i32 %tmp1, %bdz

  ; global_tid = tid_x + threads_per_block  (fake linearization to use all dims)
  %global_tid = add i32 %tid_x, %threads_per_block

  ; branch: if (global_tid > 512) early return
  %cmp = icmp ugt i32 %global_tid, 512
  br i1 %cmp, label %early_return, label %continue

early_return:
  ret void

continue:
  ; main kernel body: ptr[global_tid] += val
  %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %global_tid
  %old = load i32, i32 addrspace(1)* %elem
  %new = add i32 %old, %val
  store i32 %new, i32 addrspace(1)* %elem
  ret void
}

; intrinsics
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.size.x()
declare i32 @llvm.amdgcn.workgroup.size.y()
declare i32 @llvm.amdgcn.workgroup.size.z()
"""

    mod = parse_assembly(IR)
    old_mod = str(mod)

    block = dim3(4, 4, 4)
    grid = dim3(16, 16, 16)

    mod_hash = 0
    new_hash = jit.specialize_dims(mod, mod_hash, "kernel_add", grid, block)

    assert new_hash != mod_hash
    opt_hash = jit.optimize(mod, "gfx942", "default<O3>", 3)
    assert "%bdx = call i32 @llvm.amdgcn.workgroup.size.x()" not in str(mod)
    assert "%bdy = call i32 @llvm.amdgcn.workgroup.size.y()" not in str(mod)
    assert "%bdz = call i32 @llvm.amdgcn.workgroup.size.z()" not in str(mod)
    print(mod)


def test_launch_bounds():
    # 1. Construct minimal IR
    IR = r"""
; ModuleID = 'test'
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel_add(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  ; thread id
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  ; blockDim.x / y / z  (workgroup size)
  %bdx = call i32 @llvm.amdgcn.workgroup.size.x()
  %bdy = call i32 @llvm.amdgcn.workgroup.size.y()
  %bdz = call i32 @llvm.amdgcn.workgroup.size.z()

  ; branch: if tid > 512 return
  %cmp = icmp ugt i32 %tid, 512
  br i1 %cmp, label %early_return, label %continue

early_return:
  ret void

continue:
  ; original kernel body: ptr[tid] += val
  %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
  %old = load i32, i32 addrspace(1)* %elem
  %new = add i32 %old, %val
  store i32 %new, i32 addrspace(1)* %elem

  ; optionally use blockDim values to avoid dead code elimination
  ; (useful for testing prune)
  %sum_dims = add i32 %bdx, %bdy
  %sum_dims2 = add i32 %sum_dims, %bdz
  %_unused = add i32 %new, %sum_dims2

  ret void
}

; intrinsics
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.size.x()
declare i32 @llvm.amdgcn.workgroup.size.y()
declare i32 @llvm.amdgcn.workgroup.size.z()
"""

    mod = parse_assembly(IR)
    old_mod = str(mod)

    mod_hash = 0
    new_hash = jit.set_launch_bounds(mod, mod_hash, "kernel_add", 128, 2)

    assert new_hash != mod_hash
    opt_hash = jit.optimize(mod, "gfx942", "default<O3>", 3)
    assert (
        'attributes #0 = { "amdgpu-flat-work-group-size"="1,128" "amdgpu-waves-per-eu"="2,2" }'
        in str(mod)
    )


def test_link_llvm_modules(tmp_path):
    MODULE_A = r"""
    ; ModuleID = 'module_a'
    target triple = "amdgcn-amd-amdhsa"

    define amdgpu_kernel void @kernel_a(i32 addrspace(1)* %ptr) {
    entry:
      %tid = call i32 @llvm.amdgcn.workitem.id.x()
      %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
      %old = load i32, i32 addrspace(1)* %elem
      %new = add i32 %old, 1
      store i32 %new, i32 addrspace(1)* %elem
      ret void
    }

    declare i32 @llvm.amdgcn.workitem.id.x()
    """
    MODULE_B = r"""
    ; ModuleID = 'module_b'
    target triple = "amdgcn-amd-amdhsa"

    define amdgpu_kernel void @kernel_b(i32 addrspace(1)* %ptr, i32 %val) {
    entry:
      %tid = call i32 @llvm.amdgcn.workitem.id.x()
      %elem = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i32 %tid
      %old = load i32, i32 addrspace(1)* %elem
      %new = add i32 %old, %val
      store i32 %new, i32 addrspace(1)* %elem
      ret void
    }

    declare i32 @llvm.amdgcn.workitem.id.x()
    """
    # 1. Create paths inside pytest's temporary directory
    module_a_path = tmp_path / "module_a.ll"
    module_b_path = tmp_path / "module_b.ll"
    mod_A = parse_assembly(MODULE_A)
    mod_A.to_bitcode(str(module_a_path))

    mod_B = parse_assembly(MODULE_B)
    mod_B.to_bitcode(str(module_b_path))

    modules = [str(module_a_path), str(module_b_path)]

    mod = jit.link_llvm_modules(modules, "kernel_a", False, False)

    assert (
        "define amdgpu_kernel void @kernel_b(ptr addrspace(1) %ptr, i32 %val)"
        in str(mod)
    )
    assert "define amdgpu_kernel void @kernel_a(ptr addrspace(1) %ptr)" in str(mod)
    mod = jit.link_llvm_modules(modules, "kernel_a", False, True)
    assert (
        "define internal amdgpu_kernel void @kernel_b(ptr addrspace(1) %ptr, i32 %val)"
        in str(mod)
    )
