include("TritonCxxWrap.jl")
##
using CUDA
using OrderedCollections
using SHA
using Base64
using BenchmarkTools
using LinearAlgebra
using Functors
using BFloat16s
using Parameters

using Debugger

const CT = CppTriton

include("cache.jl")
include("global_implicit.jl")
# include("helpers.jl")
# include("tensor_ops.jl")
include("typed_types.jl")
include("compilation.jl")
include("ops.jl")
include("autotune.jl")
include("overrides/CUBLAS.jl")

##

DATA_TYPE = Float16

dynamic_argument_types = OrderedDict([
    :a_ptr => Ptr{DATA_TYPE},
    :b_ptr => Ptr{DATA_TYPE},
    :c_ptr => Ptr{DATA_TYPE},
    :M => Int32,
    :N => Int32,
    :K => Int32,
    :stride_ak => Int32,
    :stride_bn => Int32,
    :stride_cn => Int32,
])

config_params = ConfigParams(
    8,
    4,
    OrderedDict([
        :BLOCK_SIZE_K => Int32(32),
        :stride_am => Int32(1),
        :stride_bk => Int32(1),
        :stride_cm => Int32(1),
        :BLOCK_SIZE_M => Int32(256),
        :GROUP_SIZE_M => Int32(4),
        :BLOCK_SIZE_N => Int32(128),
    ]),
)

matmul_kernel(; # note: all of these are kwargs
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ak,
    stride_bn,
    stride_cn,
    stride_am,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
) = begin

    pid = program_id(1)
    num_pid_m = cdiv(M, BLOCK_SIZE_M)
    num_pid_n = cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid ÷ num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) ÷ group_size_m

    a_block_ptr = block_ptr(
        a_ptr,
        parent_shape = (M, K),
        parent_strides = (stride_am, stride_ak),
        offsets = (pid_m * BLOCK_SIZE_M, 0),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K),
        order = (2, 1),
    )
    b_block_ptr = block_ptr(
        b_ptr,
        parent_shape = (K, N),
        parent_strides = (stride_bk, stride_bn),
        offsets = (0, pid_n * BLOCK_SIZE_N),
        block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_N),
        order = (2, 1),
    )


    # offs_am = (pid_m * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)) % M
    # offs_bn = (pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)) % N
    # offs_k = arange(0, BLOCK_SIZE_K)
    # a_ptrs =
    #     a_ptr + (expanddims(offs_am, 2) * stride_am + expanddims(offs_k, 1) * stride_ak)
    # b_ptrs =
    #     b_ptr + (expanddims(offs_k, 2) * stride_bk + expanddims(offs_bn, 1) * stride_bn)
    accumulator = zeros(TrVal{Float32}, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    (accumulator, a_block_ptr, b_block_ptr) =
    # control flow functional for now
        triton_for!(
            Int32(0),
            cdiv(K, TrVal(BLOCK_SIZE_K)),
            Int32(1),
            accumulator,
            a_block_ptr,
            b_block_ptr,
        ) do k, accumulator, a_block_ptr, b_block_ptr
            a = load(a_block_ptr; boundary_check = (1, 2))
            b = load(b_block_ptr; boundary_check = (1, 2))

            accumulator = dot(a, b; allow_tf32 = true) + accumulator

            a_block_ptr = advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = advance(b_block_ptr, (BLOCK_SIZE_K, 0))
            return (accumulator, a_block_ptr, b_block_ptr)
        end

    # offs_cm = pid_m * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)

    # c_ptrs =
    #     (c_ptr + stride_cm * expanddims(offs_cm, 2)) + stride_cn * expanddims(offs_cn, 1)
    # c_mask = (expanddims(offs_cm, 2) < M) & (expanddims(offs_cn, 1) < N)
    # store(c_ptrs, cast(accumulator, DATA_TYPE); mask = c_mask)

    c_block_ptr = block_ptr(
        c_ptr;
        parent_shape = (M, N),
        parent_strides = (stride_cm, stride_cn),
        offsets = (pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N),
        order = (2, 1),
    )

    store(c_block_ptr, cast(ccumulator, DATA_TYPE); boundary_check = (1, 2))

    triton_return()
end

# zeros(Float32, (1, 2))

grid_map_fn(dyn_args, static_args) =
    cdiv(dyn_args[:M], static_args[:BLOCK_SIZE_M]) *
    cdiv(dyn_args[:N], static_args[:BLOCK_SIZE_N])
tk = compile_triton_kernel(
    matmul_kernel,
    dynamic_argument_types,
    config_params,
    grid_map_fn;
    print_opt_ttir = true,
)

triton_matmul!(out, a, b) = tk(
    a,
    b,
    out,
    size(a, 1),
    size(b, 2),
    size(a, 2),
    stride(a, 2),
    stride(b, 2),
    stride(out, 2),
)


@testset "Compiles and runs correctly" begin
    SZ = 4096
    a = CUDA.randn(Float16, 4096, 4096)
    b = CUDA.randn(Float16, 4096, 4096)
    out = CUDA.zeros(Float16, 4096, 4096)
    out_blas = CUDA.zeros(Float16, 4096, 4096)
    triton_matmul!(out, a, b)
    mul!(out_blas, a, b)
    @test out ≈ out_blas
end




# DATA_TYPE = Tfp16

# @benchmark begin @CUDA.sync triton_matmul!(out, a, b) end #setup=(a=CUDA.rand(Float16, SZ, SZ); b=CUDA.rand(Float16, SZ, SZ); out=CUDA.zeros(Float16,SZ, SZ))
# @benchmark begin @CUDA.sync mul!(out, a, b) end #setup=(a=CUDA.rand(Float16, SZ, SZ); b=CUDA.rand(Float16, SZ, SZ); out=CUDA.zeros(Float16,SZ, SZ))


# tood(xs) = OrderedDict([k => Int32(v) for (k, v) in xs])

# search_space = [
# ConfigParams(num_warps = 8, num_stages = 5, static_args = tood([
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 64, :BLOCK_SIZE_N => 256, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ])),
# ConfigParams(num_warps = 8, num_stages = 4, static_args = tood([ 
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 128, :BLOCK_SIZE_N => 128, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ])),
#     ConfigParams(num_warps = 8, num_stages = 5, static_args = tood([ 
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 128, :BLOCK_SIZE_N => 128, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ])),
#     ConfigParams(num_warps = 8, num_stages = 4, static_args = tood([ 
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 128, :BLOCK_SIZE_N => 256, :BLOCK_SIZE_K => 64, :GROUP_SIZE_M => 8, ])),


#     ConfigParams(num_warps = 8, num_stages = 4, static_args = tood([
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 128, :BLOCK_SIZE_N => 128, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ])),
#     ConfigParams(num_warps = 8, num_stages = 4, static_args = tood([
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 128, :BLOCK_SIZE_N => 64, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ])),
#     ConfigParams(num_warps = 8, num_stages = 4, static_args = tood([
#     :stride_am => 1, :stride_bk => 1, :stride_cm => 1, :BLOCK_SIZE_M => 64, :BLOCK_SIZE_N => 128, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ])),
# ]

# example_args = (
#     CUDA.rand(Float16, SZ, SZ),
#     CUDA.rand(Float16, SZ, SZ),
#     CUDA.zeros(Float16, SZ, SZ),
#     SZ, SZ, SZ, SZ, SZ, SZ   
# )

# bm_results = optimize(matmul_kernel, dynamic_argument_types, example_args, search_space, grid_map_fn)

##

dynamic_argument_types = OrderedDict([
    :a_ptr => Ptr{DATA_TYPE},
    :b_ptr => Ptr{DATA_TYPE},
    :c_ptr => Ptr{DATA_TYPE},
    :M => Int32,
    :N => Int32,
    :K => Int32,
    :stride_ak => Int32,
    :stride_bn => Int32,
    :stride_cn => Int32,
])


search_space = FullSearchSpace(
    num_warps = Log2DiscreteSearchSpace(Int64, 4, 16),
    num_stages = LinearDiscreteSearchSpace(Int64, 2e0, 5e0),
    static_args = OrderedDict([
        :stride_am => LinearDiscreteSearchSpace(Int32, 1.0, 1.0),
        :stride_bk => LinearDiscreteSearchSpace(Int32, 1.0, 1.0),
        :stride_cm => LinearDiscreteSearchSpace(Int32, 1.0, 1.0),
        :BLOCK_SIZE_M => Log2DiscreteSearchSpace(Int32, 32, 256),
        :BLOCK_SIZE_N => Log2DiscreteSearchSpace(Int32, 32, 256),
        :BLOCK_SIZE_K => Log2DiscreteSearchSpace(Int32, 32, 256),
        :GROUP_SIZE_M => Log2DiscreteSearchSpace(Int32, 4, 8),
    ]),
)


# CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION


SZ = 4096
const example_args2 = (
    CUDA.randn(Float16, SZ, SZ),
    CUDA.randn(Float16, SZ, SZ),
    CUDA.zeros(Float16, SZ, SZ),
    SZ,
    SZ,
    SZ,
    SZ,
    SZ,
    SZ,
)

# init_assignment = ConfigParams(num_warps = 8, num_stages = 5, static_args = tood([
#     :stride_ak => 1, :stride_bn => 1, :stride_cn => 1, :BLOCK_SIZE_M => 64, :BLOCK_SIZE_N => 256, :BLOCK_SIZE_K => 32, :GROUP_SIZE_M => 8, ]))

init_assignment = ConfigParams(
    8,
    3,
    OrderedDict([
        :BLOCK_SIZE_K => Int32(32),
        :stride_am => Int32(1),
        :stride_bk => Int32(1),
        :stride_cm => Int32(1),
        :BLOCK_SIZE_M => Int32(256),
        :GROUP_SIZE_M => Int32(4),
        :BLOCK_SIZE_N => Int32(128),
    ]),
)

using BlackBoxOptim

optimize_bbo(
    matmul_kernel,
    dynamic_argument_types,
    search_space,
    example_args2,
    grid_map_fn;
    init_assignment,
    iterations = 5,
)
