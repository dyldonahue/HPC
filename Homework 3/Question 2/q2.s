	.file	"q2.c"
	.text
	.globl	CLOCK
	.type	CLOCK, @function
CLOCK:
.LFB4379:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	leaq	-16(%rbp), %rax
	movq	%rax, %rsi
	movl	$1, %edi
	call	clock_gettime
	movq	-16(%rbp), %rax
	imulq	$1000, %rax, %rax
	vcvtsi2sdq	%rax, %xmm1, %xmm1
	movq	-8(%rbp), %rax
	vcvtsi2sdq	%rax, %xmm2, %xmm2
	vmovsd	.LC0(%rip), %xmm0
	vmulsd	%xmm0, %xmm2, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4379:
	.size	CLOCK, .-CLOCK
	.globl	matrix_vector_product
	.type	matrix_vector_product, @function
matrix_vector_product:
.LFB4380:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L4
.L7:
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, (%rax)
	movl	$0, -8(%rbp)
	jmp	.L5
.L6:
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	vmovss	(%rax), %xmm1
	movl	-4(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$8, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-8(%rbp), %eax
	cltq
	vmovss	(%rdx,%rax,4), %xmm2
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	vmovss	(%rax), %xmm0
	vmulss	%xmm0, %xmm2, %xmm0
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	vaddss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, (%rax)
	addl	$1, -8(%rbp)
.L5:
	cmpl	$1027, -8(%rbp)
	jle	.L6
	addl	$1, -4(%rbp)
.L4:
	cmpl	$1027, -4(%rbp)
	jle	.L7
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4380:
	.size	matrix_vector_product, .-matrix_vector_product
	.globl	matrix_vector_avx512f
	.type	matrix_vector_avx512f, @function
matrix_vector_avx512f:
.LFB4381:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	subq	$648, %rsp
	movq	%rdi, -64(%rsp)
	movq	%rsi, -72(%rsp)
	movq	%rdx, -80(%rsp)
	movl	$0, 644(%rsp)
	jmp	.L9
.L20:
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%zmm0, 520(%rsp)
	movl	$0, 516(%rsp)
	jmp	.L11
.L15:
	movl	644(%rsp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$8, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rsp), %rax
	addq	%rax, %rdx
	movl	516(%rsp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, 120(%rsp)
	movq	120(%rsp), %rax
	vmovups	(%rax), %zmm0
	vmovaps	%zmm0, 392(%rsp)
	movl	516(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rsp), %rax
	addq	%rdx, %rax
	movq	%rax, 128(%rsp)
	movq	128(%rsp), %rax
	vmovups	(%rax), %zmm0
	vmovaps	%zmm0, 328(%rsp)
	vmovaps	392(%rsp), %zmm0
	vmovaps	%zmm0, 264(%rsp)
	vmovaps	328(%rsp), %zmm0
	vmovaps	%zmm0, 200(%rsp)
	vmovaps	520(%rsp), %zmm0
	vmovaps	%zmm0, 136(%rsp)
	vmovaps	264(%rsp), %zmm0
	vmovaps	200(%rsp), %zmm2
	vmovaps	136(%rsp), %zmm1
	movl	$-1, %eax
	kmovw	%eax, %k1
	vfmadd132ps	%zmm2, %zmm1, %zmm0{%k1}
	nop
	vmovaps	%zmm0, 520(%rsp)
	addl	$16, 516(%rsp)
.L11:
	cmpl	$1012, 516(%rsp)
	jle	.L15
	leaq	-56(%rsp), %rax
	movq	%rax, 112(%rsp)
	vmovaps	520(%rsp), %zmm0
	vmovaps	%zmm0, 8(%rsp)
	vmovaps	8(%rsp), %zmm0
	movq	112(%rsp), %rax
	vmovups	%zmm0, (%rax)
	nop
	vxorps	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, 512(%rsp)
	movl	$0, 508(%rsp)
	jmp	.L16
.L17:
	movl	508(%rsp), %eax
	cltq
	vmovss	-56(%rsp,%rax,4), %xmm0
	vmovss	512(%rsp), %xmm1
	vaddss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 512(%rsp)
	addl	$1, 508(%rsp)
.L16:
	cmpl	$15, 508(%rsp)
	jle	.L17
	movl	516(%rsp), %eax
	movl	%eax, 504(%rsp)
	jmp	.L18
.L19:
	movl	644(%rsp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$8, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rsp), %rax
	addq	%rax, %rdx
	movl	504(%rsp), %eax
	cltq
	vmovss	(%rdx,%rax,4), %xmm1
	movl	504(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rsp), %rax
	addq	%rdx, %rax
	vmovss	(%rax), %xmm0
	vmulss	%xmm0, %xmm1, %xmm0
	vmovss	512(%rsp), %xmm1
	vaddss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 512(%rsp)
	addl	$1, 504(%rsp)
.L18:
	cmpl	$1027, 504(%rsp)
	jle	.L19
	movl	644(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rsp), %rax
	addq	%rdx, %rax
	vmovss	512(%rsp), %xmm0
	vmovss	%xmm0, (%rax)
	addl	$1, 644(%rsp)
.L9:
	cmpl	$1027, 644(%rsp)
	jle	.L20
	nop
	vmovd	%eax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4381:
	.size	matrix_vector_avx512f, .-matrix_vector_avx512f
	.section	.rodata
	.align 8
.LC3:
	.string	"Result from a chosen index (Result[76]): %f \n"
	.align 8
.LC4:
	.string	"The total time for matrix multiplication with AVX = %f ms\n"
	.align 8
.LC5:
	.string	"The total time for matrix multiplication without AVX = %f ms\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4382:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4235392, %rsp
	movl	$0, -4(%rbp)
	jmp	.L22
.L25:
	movl	-4(%rbp), %eax
	cltq
	vmovss	.LC2(%rip), %xmm0
	vmovss	%xmm0, -4231280(%rbp,%rax,4)
	movl	-4(%rbp), %eax
	cltq
	vxorps	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, -4235392(%rbp,%rax,4)
	movl	$0, -8(%rbp)
	jmp	.L23
.L24:
	movl	-8(%rbp), %eax
	movslq	%eax, %rcx
	movl	-4(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$8, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	vmovss	.LC2(%rip), %xmm0
	vmovss	%xmm0, -4227168(%rbp,%rax,4)
	addl	$1, -8(%rbp)
.L23:
	cmpl	$1027, -8(%rbp)
	jle	.L24
	addl	$1, -4(%rbp)
.L22:
	cmpl	$1027, -4(%rbp)
	jle	.L25
	movl	$0, %eax
	call	CLOCK
	vmovq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	leaq	-4235392(%rbp), %rdx
	leaq	-4231280(%rbp), %rcx
	leaq	-4227168(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	matrix_vector_avx512f
	movl	$0, %eax
	call	CLOCK
	vmovq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	vmovsd	-24(%rbp), %xmm0
	vsubsd	-16(%rbp), %xmm0, %xmm0
	vmovsd	%xmm0, -32(%rbp)
	vmovss	-4235088(%rbp), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm1
	vmovq	%xmm1, %rax
	vmovq	%rax, %xmm0
	movl	$.LC3, %edi
	movl	$1, %eax
	call	printf
	movq	-32(%rbp), %rax
	vmovq	%rax, %xmm0
	movl	$.LC4, %edi
	movl	$1, %eax
	call	printf
	movl	$0, %eax
	call	CLOCK
	vmovq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	leaq	-4235392(%rbp), %rdx
	leaq	-4231280(%rbp), %rcx
	leaq	-4227168(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	matrix_vector_product
	movl	$0, %eax
	call	CLOCK
	vmovq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	vmovsd	-24(%rbp), %xmm0
	vsubsd	-16(%rbp), %xmm0, %xmm0
	vmovsd	%xmm0, -32(%rbp)
	vmovss	-4235088(%rbp), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm2
	vmovq	%xmm2, %rax
	vmovq	%rax, %xmm0
	movl	$.LC3, %edi
	movl	$1, %eax
	call	printf
	movq	-32(%rbp), %rax
	vmovq	%rax, %xmm0
	movl	$.LC5, %edi
	movl	$1, %eax
	call	printf
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4382:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	-1598689907
	.long	1051772663
	.align 4
.LC2:
	.long	1065353216
	.ident	"GCC: (GNU) 11.4.1 20230605 (Red Hat 11.4.1-2)"
	.section	.note.GNU-stack,"",@progbits