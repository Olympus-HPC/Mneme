#include "core.h"
#include "llvm-c/Core.h"
#include <string>

#include <iostream>

#include "llvm/Analysis/DOTGraphTraitsPass.h"

/* An iterator around a attribute list, including the stop condition */
struct AttributeListIterator {
    typedef llvm::AttributeList::iterator const_iterator;
    const_iterator cur;
    const_iterator end;

    AttributeListIterator(const_iterator cur, const_iterator end)
        : cur(cur), end(end) {}
};

struct OpaqueAttributeListIterator;
typedef OpaqueAttributeListIterator *LLVMAttributeListIteratorRef;

/* An iterator around a attribute set, including the stop condition */
struct AttributeSetIterator {
    typedef llvm::AttributeSet::iterator const_iterator;
    const_iterator cur;
    const_iterator end;

    AttributeSetIterator(const_iterator cur, const_iterator end)
        : cur(cur), end(end) {}
};

struct OpaqueAttributeSetIterator;
typedef OpaqueAttributeSetIterator *LLVMAttributeSetIteratorRef;

namespace llvm {

static LLVMAttributeListIteratorRef wrap(AttributeListIterator *GI) {
    return reinterpret_cast<LLVMAttributeListIteratorRef>(GI);
}

static AttributeListIterator *unwrap(LLVMAttributeListIteratorRef GI) {
    return reinterpret_cast<AttributeListIterator *>(GI);
}

static LLVMAttributeSetIteratorRef wrap(AttributeSetIterator *GI) {
    return reinterpret_cast<LLVMAttributeSetIteratorRef>(GI);
}

static AttributeSetIterator *unwrap(LLVMAttributeSetIteratorRef GI) {
    return reinterpret_cast<AttributeSetIterator *>(GI);
}

} // namespace llvm

extern "C" {

API_EXPORT(LLVMAttributeListIteratorRef)
LLVMPY_FunctionAttributesIter(LLVMValueRef F) {
    using namespace llvm;
    Function *func = unwrap<Function>(F);
    AttributeList attrs = func->getAttributes();
    return wrap(new AttributeListIterator(attrs.begin(), attrs.end()));
}

API_EXPORT(LLVMAttributeSetIteratorRef)
LLVMPY_ArgumentAttributesIter(LLVMValueRef A) {
    using namespace llvm;
    Argument *arg = unwrap<Argument>(A);
    unsigned argno = arg->getArgNo();
    const AttributeSet attrs = arg->getParent()->getAttributes().
#if LLVM_VERSION_MAJOR < 14
                               getParamAttributes(argno)
#else
                               getParamAttrs(argno)
#endif
        ;
    return wrap(new AttributeSetIterator(attrs.begin(), attrs.end()));
}

API_EXPORT(LLVMAttributeListIteratorRef)
LLVMPY_CallInstAttributesIter(LLVMValueRef C) {
    using namespace llvm;
    CallInst *inst = unwrap<CallInst>(C);
    AttributeList attrs = inst->getAttributes();
    return wrap(new AttributeListIterator(attrs.begin(), attrs.end()));
}

API_EXPORT(LLVMAttributeListIteratorRef)
LLVMPY_InvokeInstAttributesIter(LLVMValueRef C) {
    using namespace llvm;
    InvokeInst *inst = unwrap<InvokeInst>(C);
    AttributeList attrs = inst->getAttributes();
    return wrap(new AttributeListIterator(attrs.begin(), attrs.end()));
}

API_EXPORT(LLVMAttributeSetIteratorRef)
LLVMPY_GlobalAttributesIter(LLVMValueRef G) {
    using namespace llvm;
    GlobalVariable *g = unwrap<GlobalVariable>(G);
    AttributeSet attrs = g->getAttributes();
    return wrap(new AttributeSetIterator(attrs.begin(), attrs.end()));
}

API_EXPORT(LLVMAttributeSetIteratorRef)
LLVMPY_AttributeListIterNext(LLVMAttributeListIteratorRef GI) {
    using namespace llvm;
    AttributeListIterator *iter = unwrap(GI);
    if (iter->cur != iter->end) {
        const llvm::AttributeSet *attrs = iter->cur;
        iter->cur++;
        return wrap(new AttributeSetIterator(attrs->begin(), attrs->end()));
    } else {
        return NULL;
    }
}

API_EXPORT(LLVMAttributeRef)
LLVMPY_AttributeSetIterNext(LLVMAttributeSetIteratorRef GI) {
    using namespace llvm;
    AttributeSetIterator *iter = unwrap(GI);

    if (iter->cur != iter->end) {
        const Attribute *attr = iter->cur;
        iter->cur++;
        return wrap(static_cast<const Attribute>(*attr));
    } else {
        return NULL;
    }
}

API_EXPORT(void)
LLVMPY_DisposeAttributeListIter(LLVMAttributeListIteratorRef GI) {
    delete llvm::unwrap(GI);
}

API_EXPORT(void)
LLVMPY_DisposeAttributeSetIter(LLVMAttributeSetIteratorRef GI) {
    delete llvm::unwrap(GI);
}

API_EXPORT(unsigned int)
LLVMPY_GetEnumAttributeKindForName(const char *name, size_t len) {
    /* zero is returned if no match */
    return LLVMGetEnumAttributeKindForName(name, len);
}

API_EXPORT(void)
LLVMPY_AddFunctionAttr(LLVMValueRef Fn, unsigned AttrKind) {
    LLVMContextRef ctx = LLVMGetModuleContext(LLVMGetGlobalParent(Fn));
    LLVMAttributeRef attr_ref = LLVMCreateEnumAttribute(ctx, AttrKind, 0);
    LLVMAddAttributeAtIndex(Fn, LLVMAttributeReturnIndex, attr_ref);
}

API_EXPORT(unsigned)
LLVMPY_GetEnumAttributeKind(LLVMAttributeRef A) {
    return LLVMGetEnumAttributeKind(A);
}

API_EXPORT(bool)
LLVMPY_AttributeIsType(LLVMAttributeRef A) {
    llvm::Attribute attr = llvm::unwrap(A);
    return attr.isTypeAttribute();
}

API_EXPORT(bool)
LLVMPY_AttributeIsInt(LLVMAttributeRef A) {
    llvm::Attribute attr = llvm::unwrap(A);
    return attr.isIntAttribute();
}

API_EXPORT(bool)
LLVMPY_AttributeIsEnum(LLVMAttributeRef A) {
    llvm::Attribute attr = llvm::unwrap(A);
    return attr.isEnumAttribute();
}

API_EXPORT(bool)
LLVMPY_AttributeIsString(LLVMAttributeRef A) {
    llvm::Attribute attr = llvm::unwrap(A);
    return attr.isStringAttribute();
}

API_EXPORT(const char *)
LLVMPY_GetAttributeAsString(LLVMAttributeRef A) {
    llvm::Attribute attr = llvm::unwrap(A);
    auto str = attr.getAsString();
    return LLVMPY_CreateString(str.c_str());
}

API_EXPORT(void)
LLVMPY_AddFunctionKeyValueAttr(LLVMValueRef Fn, const char *K, size_t KLength,
                               const char *V, size_t VLength) {
    LLVMContextRef ctx = LLVMGetModuleContext(LLVMGetGlobalParent(Fn));
    LLVMAttributeRef attr_ref =
        LLVMCreateStringAttribute(ctx, K, KLength, V, VLength);
    LLVMAddAttributeAtIndex(Fn, LLVMAttributeReturnIndex, attr_ref);
}
}
