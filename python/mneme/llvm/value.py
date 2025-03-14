from ctypes import (
    POINTER,
    byref,
    cast,
    c_char_p,
    c_double,
    c_int,
    c_int64,
    c_size_t,
    c_uint,
    c_uint8,
    c_uint64,
    c_bool,
    c_void_p,
)
import enum

from llvm4ml.binding import ffi
from llvm4ml.binding.common import _decode_string, _encode_string
from llvm4ml.binding.typeref import TypeRef
from llvm4ml.binding.attribute import AttributeRef


class Linkage(enum.IntEnum):
    # The LLVMLinkage enum from llvm-c/Core.h

    external = 0
    available_externally = 1
    linkonce_any = 2
    linkonce_odr = 3
    linkonce_odr_autohide = 4
    weak_any = 5
    weak_odr = 6
    appending = 7
    internal = 8
    private = 9
    dllimport = 10
    dllexport = 11
    external_weak = 12
    ghost = 13
    common = 14
    linker_private = 15
    linker_private_weak = 16


class Visibility(enum.IntEnum):
    # The LLVMVisibility enum from llvm-c/Core.h

    default = 0
    hidden = 1
    protected = 2


class StorageClass(enum.IntEnum):
    # The LLVMDLLStorageClass enum from llvm-c/Core.h

    default = 0
    dllimport = 1
    dllexport = 2


class ValueKind(enum.IntEnum):
    # The LLVMValueKind enum from llvm-c/Core.h

    argument = 0
    basic_block = 1
    memory_use = 2
    memory_def = 3
    memory_phi = 4

    function = 5
    global_alias = 6
    global_ifunc = 7
    global_variable = 8
    block_address = 9
    constant_expr = 10
    constant_array = 11
    constant_struct = 12
    constant_vector = 13

    undef_value = 14
    constant_aggregate_zero = 15
    constant_data_array = 16
    constant_data_vector = 17
    constant_int = 18
    constant_fp = 19
    constant_pointer_null = 20
    constant_token_none = 21

    metadata_as_value = 22
    inline_asm = 23

    instruction = 24
    poison_value = 25


class ValueRef(ffi.ObjectRef):
    """A weak reference to a LLVM value."""

    def __init__(self, ptr, kind, parents):
        self._kind = kind
        self._parents = parents
        ffi.ObjectRef.__init__(self, ptr)

    def __str__(self):
        with ffi.OutputString() as outstr:
            ffi.lib.LLVMPY_PrintValueToString(self, outstr)
            return str(outstr)

    @property
    def module(self):
        """
        The module this function or global variable value was obtained from.
        """
        return self._parents.get("module")

    @property
    def function(self):
        """
        The function this argument or basic block value was obtained from.
        """
        return self._parents.get("function")

    @property
    def block(self):
        """
        The block this instruction value was obtained from.
        """
        return self._parents.get("block")

    @property
    def instruction(self):
        """
        The instruction this operand value was obtained from.
        """
        return self._parents.get("instruction")

    @property
    def is_global(self):
        return self._kind == "global"

    @property
    def is_function(self):
        return self._kind == "function"

    @property
    def is_block(self):
        return self._kind == "block"

    @property
    def is_argument(self):
        return self._kind == "argument"

    @property
    def is_instruction(self):
        return self._kind == "instruction"

    @property
    def alignment(self):
        """The alignment property."""
        return ffi.lib.LLVMPY_GetAlignment(self)

    @property
    def is_memory_instruction(self):
        if self._kind != "instruction":
            return False

        memory_instructions = ("alloca", "store", "load", "getelementptr")
        if self.opcode in memory_instructions:
            return True

        return False

    @property
    def is_operand(self):
        return self._kind == "operand"

    @property
    def is_constant(self):
        return bool(ffi.lib.LLVMPY_IsConstant(self))

    @property
    def value_kind(self):
        return ValueKind(ffi.lib.LLVMPY_GetValueKind(self))

    @property
    def name(self):
        return _decode_string(ffi.lib.LLVMPY_GetValueName(self))

    @name.setter
    def name(self, val):
        ffi.lib.LLVMPY_SetValueName(self, _encode_string(val))

    @property
    def linkage(self):
        if self.value_kind in (
            ValueKind.global_alias,
            ValueKind.global_ifunc,
            ValueKind.global_variable,
            ValueKind.function,
        ):
            return Linkage(ffi.lib.LLVMPY_GetLinkage(self))
        raise TypeError(f"expected global value, got {self}." f"ValueKind is {self.value_kind.name}")

    @linkage.setter
    def linkage(self, value):
        if not isinstance(value, Linkage):
            value = Linkage[value]
        ffi.lib.LLVMPY_SetLinkage(self, value)

    @property
    def visibility(self):
        return Visibility(ffi.lib.LLVMPY_GetVisibility(self))

    @visibility.setter
    def visibility(self, value):
        if not isinstance(value, Visibility):
            value = Visibility[value]
        ffi.lib.LLVMPY_SetVisibility(self, value)

    @property
    def storage_class(self):
        return StorageClass(ffi.lib.LLVMPY_GetDLLStorageClass(self))

    @storage_class.setter
    def storage_class(self, value):
        if not isinstance(value, StorageClass):
            value = StorageClass[value]
        ffi.lib.LLVMPY_SetDLLStorageClass(self, value)

    def add_function_attribute(self, attr):
        """Only works on function value

        Parameters
        -----------
        attr : str
            attribute name
        """
        if not self.is_function:
            raise ValueError("expected function value, got %s" % (self._kind,))
        attrname = str(attr)
        attrval = ffi.lib.LLVMPY_GetEnumAttributeKindForName(_encode_string(attrname), len(attrname))
        if attrval == 0:
            raise ValueError("no such attribute {!r}".format(attrname))
        ffi.lib.LLVMPY_AddFunctionAttr(self, attrval)

    def add_function_key_value_attribute(self, key, value):
        if not self.is_function:
            raise ValueError("expected function value, got %s" % (self._kind,))

        ffi.lib.LLVMPY_AddFunctionKeyValueAttr(self, _encode_string(key), len(key), _encode_string(value), len(value))

    @property
    def type(self):
        """
        This value's LLVM type.
        """
        # XXX what does this return?
        return TypeRef(ffi.lib.LLVMPY_TypeOf(self), self.module)

    @property
    def memory_type(self):
        """
        The memory type accessed by this instruction LLVM type.
        """
        if not self.is_memory_instruction:
            raise ValueError("Argument is not  amemory instruciton {!r}".format(str(self)))

        return TypeRef(ffi.lib.LLVMPY_TypeOfMemory(self), self.module)

    @property
    def has_initializer(self):
        """
        Returns True if a global variable has an initializer.
        """
        if self.value_kind != ValueKind.global_variable:
            raise ValueError("expected global value, got %s" % (self._kind))
        return ffi.lib.LLVMPY_HasInitializer(self)

    @property
    def initializer(self):
        """
        Returns the initializer of a global variable.
        """
        if self.value_kind != ValueKind.global_variable:
            raise ValueError("expected global value, got %s" % (self._kind))
        if not self.has_initializer:
            return None
        return ValueRef(ffi.lib.LLVMPY_GetInitializer(self), "initializer", self._parents)

    @property
    def is_declaration(self):
        """
        Whether this value (presumably global) is defined in the current
        module.
        """
        if not (self.is_global or self.is_function):
            raise ValueError("expected global or function value, got %s" % (self._kind,))
        return ffi.lib.LLVMPY_IsDeclaration(self)

    @property
    def attributes(self):
        """
        Return an iterator over this value's attributes.
        The iterator will yield a string for each attribute.
        """
        return AttributeRef.attribute_iterator(self)

    @property
    def blocks(self):
        """
        Return an iterator over this function's blocks.
        The iterator will yield a ValueRef for each block.
        """
        if not self.is_function:
            raise ValueError("expected function value, got %s" % (self._kind,))
        it = ffi.lib.LLVMPY_FunctionBlocksIter(self)
        parents = self._parents.copy()
        parents.update(function=self)
        return _BlocksIterator(it, parents)

    @property
    def arguments(self):
        """
        Return an iterator over this function's arguments.
        The iterator will yield a ValueRef for each argument.
        """
        if not self.is_function:
            raise ValueError("expected function value, got %s" % (self._kind,))
        it = ffi.lib.LLVMPY_FunctionArgumentsIter(self)
        parents = self._parents.copy()
        parents.update(function=self)
        return _ArgumentsIterator(it, parents)

    @property
    def instructions(self):
        """
        Return an iterator over this block's instructions.
        The iterator will yield a ValueRef for each instruction.
        """
        if not self.is_block:
            raise ValueError("expected block value, got %s" % (self._kind,))
        it = ffi.lib.LLVMPY_BlockInstructionsIter(self)
        parents = self._parents.copy()
        parents.update(block=self)
        return _InstructionsIterator(it, parents)

    @property
    def operands(self):
        """
        Return an iterator over this instruction's operands.
        The iterator will yield a ValueRef for each operand.
        """
        if self.value_kind not in (
            ValueKind.constant_array,
            ValueKind.constant_vector,
            ValueKind.constant_struct,
            ValueKind.constant_data_array,
            ValueKind.constant_data_vector,
            ValueKind.global_alias,
            ValueKind.constant_expr,
            ValueKind.instruction,
            ValueKind.global_variable,
            ValueKind.constant_aggregate_zero,
            ValueKind.undef_value,
            ValueKind.constant_int,
            ValueKind.constant_fp,
            ValueKind.constant_pointer_null,
        ):
            raise ValueError(
                "expected instruction value, constant aggregate, or global."
                " Got %s %s" % (self._kind, self.value_kind.name)
            )

        if self.value_kind in (
            ValueKind.constant_data_array,
            ValueKind.constant_data_vector,
        ):
            it = ffi.lib.LLVMPY_ConstantDataIter(self)
            parents = self._parents.copy()
            parents.update(instruction=self)
            return _ConstantDataIterator(it, parents)

        it = ffi.lib.LLVMPY_OperandsIter(self)
        parents = self._parents.copy()
        parents.update(instruction=self)
        return _OperandsIterator(it, parents)

    @property
    def opcode(self):
        if not self.is_instruction:
            raise ValueError("expected instruction value, got %s" % (self._kind,))
        return ffi.ret_string(ffi.lib.LLVMPY_GetOpcodeName(self))

    @property
    def incoming_blocks(self):
        if not self.is_instruction or self.opcode != "phi":
            raise ValueError("expected phi instruction value, got %s" % (self._kind,))
        it = ffi.lib.LLVMPY_PhiIncomingBlocksIter(self)
        parents = self._parents.copy()
        parents.update(instruction=self)
        return _IncomingBlocksIterator(it, parents)

    @property
    def indices(self):
        if not self.is_instruction or self.opcode not in (
            "insertvalue",
            "extractvalue",
        ):
            raise ValueError("expected insert/extractvalue value, got %s" % (self._kind,))
        it = ffi.lib.LLVMPY_IndicesIter(self)
        parents = self._parents.copy()
        parents.update(instruction=self)
        return _IndicesIterator(it, parents)

    def get_constant_value(self, signed_int=False, round_fp=False):
        """
        Return the constant value, either as a literal (when supported)
        or as a string.

        Parameters
        -----------
        signed_int : bool
            if True and the constant is an integer, returns a signed version
        round_fp : bool
            if True and the constant is a floating point value, rounds the
            result upon accuracy loss (e.g., when querying an fp128 value).
            By default, raises an exception on accuracy loss
        """
        if not self.is_constant:
            raise ValueError("expected constant value, got %s" % (self._kind,))

        if self.value_kind == ValueKind.constant_int:
            # Python integers are also arbitrary-precision
            little_endian = c_bool(False)
            numbytes = self.type.type_width // 8
            ptr = ffi.lib.LLVMPY_GetConstantIntRawValue(self, byref(little_endian))
            asbytes = bytes(cast(ptr, POINTER(c_uint8 * numbytes)).contents)
            return int.from_bytes(
                asbytes,
                ("little" if little_endian.value else "big"),
                signed=signed_int,
            )
        elif self.value_kind == ValueKind.constant_fp:
            # Convert floating-point values to double-precision (Python float)
            accuracy_loss = c_bool(False)
            value = ffi.lib.LLVMPY_GetConstantFPValue(self, byref(accuracy_loss))
            if accuracy_loss.value and not round_fp:
                raise ValueError("Accuracy loss encountered in conversion of constant " f"value {str(self)}")

            return value
        elif self.value_kind == ValueKind.constant_expr:
            # Convert constant expressions to their corresponding operands
            return [op.get_constant_value(signed_int, round_fp) for op in self.operands]
        elif self.value_kind == ValueKind.global_variable:
            # Obtain constant value from global initializer
            return self.initializer.get_constant_value(signed_int, round_fp)
        elif self.value_kind in (
            ValueKind.constant_array,
            ValueKind.constant_vector,
            ValueKind.constant_struct,
        ):
            # Convert constant aggregates to lists
            return [op.get_constant_value(signed_int, round_fp) for op in self.operands]
        elif self.value_kind in (
            ValueKind.constant_data_array,
            ValueKind.constant_data_vector,
        ):
            # Try to get the value as a constant data (sequential)
            value = ffi.lib.LLVMPY_GetConstantDataAsString(self)
            if value:
                return ffi.ret_string(value)
            # Try to get sequence elements via a slower but safer route
            num_elements = ffi.lib.LLVMPY_GetConstantSequenceNumElements(self)
            return [
                ValueRef(
                    ffi.lib.LLVMPY_GetConstantSequenceElement(self, i),
                    self._kind,
                    self._parents,
                ).get_constant_value(signed_int, round_fp)
                for i in range(num_elements)
            ]
        elif self.value_kind in (ValueKind.function, ValueKind.basic_block):
            return self

        # Otherwise, return the IR string
        return str(self)

    def as_instruction(self):
        """
        Returns a constant expression value as an instruction.
        """
        if self.value_kind != ValueKind.constant_expr:
            raise ValueError("expected constant expr, got %s" % (self.value_kind))
        return ValueRef(ffi.lib.LLVMPY_ConstantExprAsInstruction(self), "instruction", self._parents)


class _ValueIterator(ffi.ObjectRef):
    kind = None  # derived classes must specify the Value kind value

    # as class attribute

    def __init__(self, ptr, parents):
        ffi.ObjectRef.__init__(self, ptr)
        # Keep parent objects (module, function, etc) alive
        self._parents = parents
        if self.kind is None:
            raise NotImplementedError("%s must specify kind attribute" % (type(self).__name__,))

    def __next__(self):
        vp = self._next()
        if vp:
            return ValueRef(vp, self.kind, self._parents)
        else:
            raise StopIteration

    next = __next__

    def __iter__(self):
        return self


class _BlocksIterator(_ValueIterator):
    kind = "block"

    def _dispose(self):
        self._capi.LLVMPY_DisposeBlocksIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_BlocksIterNext(self)


class _ArgumentsIterator(_ValueIterator):
    kind = "argument"

    def _dispose(self):
        self._capi.LLVMPY_DisposeArgumentsIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_ArgumentsIterNext(self)


class _InstructionsIterator(_ValueIterator):
    kind = "instruction"

    def _dispose(self):
        self._capi.LLVMPY_DisposeInstructionsIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_InstructionsIterNext(self)


class _OperandsIterator(_ValueIterator):
    kind = "operand"

    def _dispose(self):
        self._capi.LLVMPY_DisposeOperandsIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_OperandsIterNext(self)


class _ConstantDataIterator(_ValueIterator):
    kind = "operand"

    def _dispose(self):
        self._capi.LLVMPY_DisposeConstantDataIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_ConstantDataIterNext(self)


class _IncomingBlocksIterator(_ValueIterator):
    kind = "block"

    def _dispose(self):
        self._capi.LLVMPY_DisposeIncomingBlocksIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_IncomingBlocksIterNext(self)


class _IndicesIterator(_ValueIterator):
    kind = "block"

    def _dispose(self):
        self._capi.LLVMPY_DisposeIndicesIter(self)

    def __next__(self):
        val = ffi.lib.LLVMPY_IndicesIterNext(self)
        if val >= 0:
            return val
        else:  # val < 0 means that the iterator has finished
            raise StopIteration


# FFI

ffi.lib.LLVMPY_PrintValueToString.argtypes = [ffi.LLVMValueRef, POINTER(c_char_p)]

ffi.lib.LLVMPY_GetGlobalParent.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetGlobalParent.restype = ffi.LLVMModuleRef

ffi.lib.LLVMPY_GetValueName.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetValueName.restype = c_char_p

ffi.lib.LLVMPY_SetValueName.argtypes = [ffi.LLVMValueRef, c_char_p]

ffi.lib.LLVMPY_TypeOf.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_TypeOf.restype = ffi.LLVMTypeRef

ffi.lib.LLVMPY_TypeOfMemory.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_TypeOfMemory.restype = ffi.LLVMTypeRef

ffi.lib.LLVMPY_GetTypeName.argtypes = [ffi.LLVMTypeRef]
ffi.lib.LLVMPY_GetTypeName.restype = c_void_p

ffi.lib.LLVMPY_GetLinkage.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetLinkage.restype = c_int

ffi.lib.LLVMPY_SetLinkage.argtypes = [ffi.LLVMValueRef, c_int]

ffi.lib.LLVMPY_GetVisibility.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetVisibility.restype = c_int

ffi.lib.LLVMPY_SetVisibility.argtypes = [ffi.LLVMValueRef, c_int]

ffi.lib.LLVMPY_GetDLLStorageClass.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetDLLStorageClass.restype = c_int

ffi.lib.LLVMPY_SetDLLStorageClass.argtypes = [ffi.LLVMValueRef, c_int]

ffi.lib.LLVMPY_AddFunctionAttr.argtypes = [ffi.LLVMValueRef, c_uint]

ffi.lib.LLVMPY_AddFunctionKeyValueAttr.argtypes = [
    ffi.LLVMValueRef,
    c_char_p,
    c_size_t,
    c_char_p,
    c_size_t,
]

ffi.lib.LLVMPY_IsDeclaration.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_IsDeclaration.restype = c_int

ffi.lib.LLVMPY_FunctionBlocksIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_FunctionBlocksIter.restype = ffi.LLVMBlocksIterator

ffi.lib.LLVMPY_FunctionArgumentsIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_FunctionArgumentsIter.restype = ffi.LLVMArgumentsIterator

ffi.lib.LLVMPY_BlockInstructionsIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_BlockInstructionsIter.restype = ffi.LLVMInstructionsIterator

ffi.lib.LLVMPY_OperandsIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_OperandsIter.restype = ffi.LLVMOperandsIterator

ffi.lib.LLVMPY_ConstantDataIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_ConstantDataIter.restype = ffi.LLVMConstantDataIterator

ffi.lib.LLVMPY_PhiIncomingBlocksIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_PhiIncomingBlocksIter.restype = ffi.LLVMIncomingBlocksIterator

ffi.lib.LLVMPY_IndicesIter.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_IndicesIter.restype = ffi.LLVMIndicesIterator

ffi.lib.LLVMPY_DisposeBlocksIter.argtypes = [ffi.LLVMBlocksIterator]

ffi.lib.LLVMPY_DisposeInstructionsIter.argtypes = [ffi.LLVMInstructionsIterator]

ffi.lib.LLVMPY_DisposeOperandsIter.argtypes = [ffi.LLVMOperandsIterator]

ffi.lib.LLVMPY_DisposeConstantDataIter.argtypes = [ffi.LLVMConstantDataIterator]

ffi.lib.LLVMPY_DisposeIncomingBlocksIter.argtypes = [ffi.LLVMIncomingBlocksIterator]

ffi.lib.LLVMPY_DisposeIndicesIter.argtypes = [ffi.LLVMIndicesIterator]

ffi.lib.LLVMPY_BlocksIterNext.argtypes = [ffi.LLVMBlocksIterator]
ffi.lib.LLVMPY_BlocksIterNext.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_ArgumentsIterNext.argtypes = [ffi.LLVMArgumentsIterator]
ffi.lib.LLVMPY_ArgumentsIterNext.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_InstructionsIterNext.argtypes = [ffi.LLVMInstructionsIterator]
ffi.lib.LLVMPY_InstructionsIterNext.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_OperandsIterNext.argtypes = [ffi.LLVMOperandsIterator]
ffi.lib.LLVMPY_OperandsIterNext.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_ConstantDataIterNext.argtypes = [ffi.LLVMConstantDataIterator]
ffi.lib.LLVMPY_ConstantDataIterNext.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_IncomingBlocksIterNext.argtypes = [ffi.LLVMIncomingBlocksIterator]
ffi.lib.LLVMPY_IncomingBlocksIterNext.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_IndicesIterNext.argtypes = [ffi.LLVMIndicesIterator]
ffi.lib.LLVMPY_IndicesIterNext.restype = c_int64

ffi.lib.LLVMPY_GetOpcodeName.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetOpcodeName.restype = c_void_p

ffi.lib.LLVMPY_IsConstant.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_IsConstant.restype = c_bool

ffi.lib.LLVMPY_GetValueKind.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetValueKind.restype = c_int

ffi.lib.LLVMPY_GetConstantIntRawValue.argtypes = [ffi.LLVMValueRef, POINTER(c_bool)]
ffi.lib.LLVMPY_GetConstantIntRawValue.restype = POINTER(c_uint64)

ffi.lib.LLVMPY_GetConstantIntNumWords.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetConstantIntNumWords.restype = c_uint

ffi.lib.LLVMPY_GetAlignment.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetAlignment.restype = c_uint64

ffi.lib.LLVMPY_GetConstantFPValue.argtypes = [ffi.LLVMValueRef, POINTER(c_bool)]
ffi.lib.LLVMPY_GetConstantFPValue.restype = c_double

ffi.lib.LLVMPY_ConstantExprAsInstruction.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_ConstantExprAsInstruction.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_HasInitializer.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_HasInitializer.restype = c_bool

ffi.lib.LLVMPY_GetInitializer.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetInitializer.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_GetConstantDataAsString.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetConstantDataAsString.restype = c_void_p

ffi.lib.LLVMPY_GetConstantSequenceElement.argtypes = [ffi.LLVMValueRef, c_uint]
ffi.lib.LLVMPY_GetConstantSequenceElement.restype = ffi.LLVMValueRef

ffi.lib.LLVMPY_GetConstantSequenceNumElements.argtypes = [ffi.LLVMValueRef]
ffi.lib.LLVMPY_GetConstantSequenceNumElements.restype = c_size_t

# ffi.lib.LLVMPY_ExtractBasicBlock.argtypes = [ffi.LLVMValueRef, ffi.LLVMValueRef]
# ffi.lib.LLVMPY_ExtractBasicBlock.restype = ffi.LLVMValueRef
