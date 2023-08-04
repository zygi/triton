
using Serialization

using CUDA

# CUDA.run_compute_sanitizer()

# exit(0)


include("compilation.jl")

template_vals = Int32[1, 1, 1, 64, 64, 64, 8]
cdiv(x::T, y::U) where {T <: Integer, U <: Integer} = ((x + y) - one(U)) ÷ y
NUM_WARPS=4

ptx = open(deserialize, "matmul_kernel_ptx.jlso")
compiled = ptx_compile(String(ptx), "test_name")
cufun = link(compiled)

# exit(0)

# CUDA.@device_code_sass cufun

SZ = 512
a = CUDA.rand(Float32, SZ, SZ)
b = CUDA.rand(Float32, SZ, SZ)
out = CUDA.zeros(Float32, SZ, SZ)

@show pointer(a)
@show pointer(b)
@show pointer(out)

CUDA.device()

# attributes(cufun)[CUDA.FUNC_ATTRIBUTE_CONST_SIZE_BYTES]

# CUDA.memory(cufun)

SHMEM=98304
CUDA.cuFuncSetAttribute(cufun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SHMEM)

# strides(a)
num_blocks = cdiv(SZ, template_vals[4]) * cdiv(SZ, template_vals[5])
##
CUDA.@sync CUDA.cudacall(cufun, 
    (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat},Cint,Cint,Cint,Cint,Cint,Cint),#,Cint,Cint,Cint),
    #  a, b, out, SZ, SZ, SZ, 1, SZ, 1, SZ, 1, SZ; 
     pointer(a), pointer(b), pointer(out), SZ, SZ, SZ, SZ, SZ, SZ; 
    #  blocks=prod(size(out)) ÷ NUM_WARPS,
    # blocks=1,
    # blocks=1,
    blocks=num_blocks,
    threads=(32*NUM_WARPS, ),
    # threads=1,
    shmem=SHMEM
    )

out

# CUDA.active_blocks(cufun, Int64(32*NUM_WARPS); shmem=0)

# ,ID,API Name,Details,Func Return,Func Parameter,Start,Duration,Queued,Submitted
# ,633,cuLaunchKernel,,CUDA_ERROR_INVALID_VALUE(1),"(0x14e7cb0, 64, 1, 1, 128, 1, 1, 98304, 0x3349060, 0x7fe5df14b3c0{0x7ffe8561b420}, 0x0)",,,,
#,ID,API Name,Details,Func Return,Func Parameter,Start,Duration,Queued,Submitted
#,1298,cuLaunchKernel,,,                           "(0x55830a2438c0, 64, 1, 1, 128, 1, 1, 98304, 0x0, 0x7fff613ddd90{0x7fff613ddd70}, 0x0)",,,,



# 2^15