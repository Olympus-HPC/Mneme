; RUN: %opt -load-pass-plugin=%plugin -passes=purge-autoinit -S %s | %FILECHECK %s

define void @f(ptr %p) {
entry:
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 64, i1 false), !annotation !0
  call void @llvm.memset.p0.i64(ptr %p, i8 1, i64 64, i1 false), !annotation !1
  call void @llvm.memset.p0.i64(ptr %p, i8 2, i64 64, i1 false), !annotation !2
  call void @llvm.memset.p0.i64(ptr %p, i8 3, i64 64, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr writeonly, i8, i64, i1 immarg)

; !0 and !1 are the two annotation shapes the IR verifier accepts, a flat string
; and a tuple of strings. !2 is an unrelated annotation.
!0 = !{!"auto-init"}
!1 = !{!{!"auto-init"}}
!2 = !{!"other"}

; CHECK-LABEL: @f
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr %p, i8 2
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr %p, i8 3
; CHECK-NEXT: ret void
