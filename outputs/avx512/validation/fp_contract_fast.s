	.file	"fp_contract_test.cpp"
	.text
	.globl	_Z19mil_incr32_contractDv16_fS_RK3C32 # -- Begin function _Z19mil_incr32_contractDv16_fS_RK3C32
	.p2align	4
	.type	_Z19mil_incr32_contractDv16_fS_RK3C32,@function
_Z19mil_incr32_contractDv16_fS_RK3C32:  # 
	.cfi_startproc
# %bb.0:
	vmulps	(%rdi), %zmm0, %zmm3
	vmovups	192(%rdi), %zmm4                # AlignMOV convert to UnAlignMOV 
	vmulps	64(%rdi), %zmm0, %zmm2
	vmulps	%zmm2, %zmm1, %zmm2
	vfmsub213ps	%zmm4, %zmm1, %zmm1     # zmm1 = (zmm1 * zmm1) - zmm4
	vmulps	128(%rdi), %zmm0, %zmm0
	vfmadd231ps	%zmm3, %zmm4, %zmm2     # zmm2 = (zmm4 * zmm3) + zmm2
	vfmadd231ps	%zmm0, %zmm1, %zmm2     # zmm2 = (zmm1 * zmm0) + zmm2
	vmovaps	%zmm2, %zmm0
	retq
.Lfunc_end0:
	.size	_Z19mil_incr32_contractDv16_fS_RK3C32, .Lfunc_end0-_Z19mil_incr32_contractDv16_fS_RK3C32
	.cfi_endproc
                                        # -- End function
	.globl	_Z12mul_then_addDv16_fS_S_      # -- Begin function _Z12mul_then_addDv16_fS_S_
	.p2align	4
	.type	_Z12mul_then_addDv16_fS_S_,@function
_Z12mul_then_addDv16_fS_S_:             # 
	.cfi_startproc
# %bb.0:
	vfmadd213ps	%zmm2, %zmm1, %zmm0     # zmm0 = (zmm1 * zmm0) + zmm2
	retq
.Lfunc_end1:
	.size	_Z12mul_then_addDv16_fS_S_, .Lfunc_end1-_Z12mul_then_addDv16_fS_S_
	.cfi_endproc
                                        # -- End function
	.globl	_Z12explicit_fmaDv16_fS_S_      # -- Begin function _Z12explicit_fmaDv16_fS_S_
	.p2align	4
	.type	_Z12explicit_fmaDv16_fS_S_,@function
_Z12explicit_fmaDv16_fS_S_:             # 
	.cfi_startproc
# %bb.0:
	vfmadd213ps	%zmm2, %zmm1, %zmm0     # zmm0 = (zmm1 * zmm0) + zmm2
	retq
.Lfunc_end2:
	.size	_Z12explicit_fmaDv16_fS_S_, .Lfunc_end2-_Z12explicit_fmaDv16_fS_S_
	.cfi_endproc
                                        # -- End function
	.globl	_Z10mul_reusedDv16_fS_S_PS_     # -- Begin function _Z10mul_reusedDv16_fS_S_PS_
	.p2align	4
	.type	_Z10mul_reusedDv16_fS_S_PS_,@function
_Z10mul_reusedDv16_fS_S_PS_:            # 
	.cfi_startproc
# %bb.0:
	vmulps	%zmm1, %zmm0, %zmm0
	vmovups	%zmm0, (%rdi)                   # AlignMOV convert to UnAlignMOV 
	vaddps	%zmm2, %zmm0, %zmm0
	retq
.Lfunc_end3:
	.size	_Z10mul_reusedDv16_fS_S_PS_, .Lfunc_end3-_Z10mul_reusedDv16_fS_S_PS_
	.cfi_endproc
                                        # -- End function
	.globl	_Z15mul_then_add_phDv32_DF16_S_S_ # -- Begin function _Z15mul_then_add_phDv32_DF16_S_S_
	.p2align	4
	.type	_Z15mul_then_add_phDv32_DF16_S_S_,@function
_Z15mul_then_add_phDv32_DF16_S_S_:      # 
	.cfi_startproc
# %bb.0:
	vfmadd213ph	%zmm2, %zmm1, %zmm0
	retq
.Lfunc_end4:
	.size	_Z15mul_then_add_phDv32_DF16_S_S_, .Lfunc_end4-_Z15mul_then_add_phDv32_DF16_S_S_
	.cfi_endproc
                                        # -- End function
	.ident	"Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0 (2026.0.0.20260331)"
	.section	".note.GNU-stack","",@progbits
