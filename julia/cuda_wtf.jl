using CUDA
using BenchmarkTools
using LinearAlgebra

CUDA.runtime_version()

SIZE = 4096
a = CUDA.rand(Float16, SIZE, SIZE);
b = CUDA.rand(Float16, SIZE, SIZE);
out = CUDA.zeros(Float16, SIZE, SIZE);

mode = Ref{UInt32}(); CUDA.CUBLAS.cublasGetMathMode(CUDA.CUBLAS.handle(), mode); mode[]

CUDA.DEFAULT_MATH

CUDA.math_mode()
# CUDA.math_mode!(CUDA.PEDANTIC_MATH)
CUDA.math_mode!(CUDA.DEFAULT_MATH)


CUDA.CUBLAS.math_mode!(CUDA.CUBLAS.handle(), CUDA.DEFAULT_MATH)

CUDA.CUBLAS.cublasSetMathMode(CUDA.CUBLAS.handle(), CUDA.CUBLAS.CUBLAS_DEFAULT_MATH)# | CUDA.CUBLAS.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))


using Debugger
Debugger.@enter mul!(out, a, b)


@benchmark begin CUDA.@sync mul!(out, a, b) end