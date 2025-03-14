from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
                    c_uint, c_uint64, c_bool, c_void_p)
import enum
import warnings

from llvm4ml.binding import ffi
from llvm4ml.binding.common import _decode_string, _encode_string
from llvm4ml.binding.typeref import TypeRef

class AttributeRef(ffi.ObjectRef):

    @property
    def kind(self):
        return ffi.lib.LLVMPY_GetEnumAttributeKind(self)

    @property
    def is_type(self):
        return ffi.lib.LLVMPY_AttributeIsType(self)

    @property
    def is_enum(self):
        return ffi.lib.LLVMPY_AttributeIsEnum(self)

    @property
    def is_integer(self):
        return ffi.lib.LLVMPY_AttributeIsInt(self)

    @property
    def is_string(self):
        return ffi.lib.LLVMPY_AttributeIsString(self)

    def __str__(self):
        return ffi.ret_string(ffi.lib.LLVMPY_GetAttributeAsString(self))

    @staticmethod
    def attribute_iterator(value):
        itr = iter(())
        if value.is_function:
            it = ffi.lib.LLVMPY_FunctionAttributesIter(value)
            itr = _AttributeListIterator(it)
        elif value.is_instruction:
            if value.opcode == 'call':
                it = ffi.lib.LLVMPY_CallInstAttributesIter(value)
                itr = _AttributeListIterator(it)
            elif value.opcode == 'invoke':
                it = ffi.lib.LLVMPY_InvokeInstAttributesIter(value)
                itr = _AttributeListIterator(it)
        elif value.is_global:
            it = ffi.lib.LLVMPY_GlobalAttributesIter(value)
            itr = _AttributeSetIterator(it)
        elif value.is_argument:
            it = ffi.lib.LLVMPY_ArgumentAttributesIter(value)
            itr = _AttributeSetIterator(it)
        return itr

class _AttributeIterator(ffi.ObjectRef):

    def __next__(self):
        vp = self._next()
        if vp:
            return vp
        else:
            raise StopIteration

    next = __next__

    def __iter__(self):
        return self


class _AttributeListIterator(_AttributeIterator):

    def _dispose(self):
        self._capi.LLVMPY_DisposeAttributeListIter(self)

    def _next(self):
        return _AttributeSetIterator(ffi.lib.LLVMPY_AttributeListIterNext(self))


class _AttributeSetIterator(_AttributeIterator):

    def _dispose(self):
        self._capi.LLVMPY_DisposeAttributeSetIter(self)

    def _next(self):
        return AttributeRef(ffi.lib.LLVMPY_AttributeSetIterNext(self))

ffi.lib.LLVMPY_GetEnumAttributeKindForName.argtypes = [c_char_p, c_size_t]
ffi.lib.LLVMPY_GetEnumAttributeKindForName.restype = c_uint

ffi.lib.LLVMPY_GetEnumAttributeKind.argtypes = [ffi.LLVMAttributeRef]
ffi.lib.LLVMPY_GetEnumAttributeKind.restype = c_uint

ffi.lib.LLVMPY_FunctionAttributesIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_FunctionAttributesIter.restype = ffi.LLVMAttributeListIterator

ffi.lib.LLVMPY_CallInstAttributesIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_CallInstAttributesIter.restype = ffi.LLVMAttributeListIterator

ffi.lib.LLVMPY_InvokeInstAttributesIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_InvokeInstAttributesIter.restype = ffi.LLVMAttributeListIterator

ffi.lib.LLVMPY_GlobalAttributesIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GlobalAttributesIter.restype = ffi.LLVMAttributeSetIterator

ffi.lib.LLVMPY_ArgumentAttributesIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_ArgumentAttributesIter.restype = ffi.LLVMAttributeSetIterator

ffi.lib.LLVMPY_DisposeAttributeListIter.argtypes = [
    ffi.LLVMAttributeListIterator]

ffi.lib.LLVMPY_DisposeAttributeSetIter.argtypes = [ffi.LLVMAttributeSetIterator]

ffi.lib.LLVMPY_AttributeListIterNext.argtypes = [ffi.LLVMAttributeListIterator]
ffi.lib.LLVMPY_AttributeListIterNext.restype = ffi.LLVMAttributeSetIterator

ffi.lib.LLVMPY_AttributeSetIterNext.argtypes = [ffi.LLVMAttributeSetIterator]
ffi.lib.LLVMPY_AttributeSetIterNext.restype = ffi.LLVMAttributeRef

ffi.lib.LLVMPY_AddFunctionAttr.argtypes = [ffi.LLVMAttributeRef]

ffi.lib.LLVMPY_AttributeIsType.argtypes = [ffi.LLVMAttributeRef]
ffi.lib.LLVMPY_AttributeIsType.restype = c_bool

ffi.lib.LLVMPY_AttributeIsInt.argtypes = [ffi.LLVMAttributeRef]
ffi.lib.LLVMPY_AttributeIsInt.restype = c_bool

ffi.lib.LLVMPY_AttributeIsEnum.argtypes = [ffi.LLVMAttributeRef]
ffi.lib.LLVMPY_AttributeIsEnum.restype = c_bool

ffi.lib.LLVMPY_AttributeIsString.argtypes = [ffi.LLVMAttributeRef]
ffi.lib.LLVMPY_AttributeIsString.restype = c_bool

ffi.lib.LLVMPY_GetAttributeAsString.argtypes = [ffi.LLVMAttributeRef]
ffi.lib.LLVMPY_GetAttributeAsString.restype = c_void_p

