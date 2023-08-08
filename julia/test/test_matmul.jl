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

    offs_am = (pid_m * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)) % N
    offs_k = arange(0, BLOCK_SIZE_K)
    a_ptrs =
        a_ptr + (expanddims(offs_am, 2) * stride_am + expanddims(offs_k, 1) * stride_ak)
    b_ptrs =
        b_ptr + (expanddims(offs_k, 2) * stride_bk + expanddims(offs_bn, 1) * stride_bn)
    accumulator = zeros(TrVal{Float32}, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    (accumulator, a_ptrs, b_ptrs) =
    # control flow functional for now
        triton_for!(
            Int32(0),
            cdiv(K, TrVal(BLOCK_SIZE_K)),
            Int32(1),
            accumulator,
            a_ptrs,
            b_ptrs,
        ) do k, accumulator, a_ptrs, b_ptrs
            a = load(
                a_ptrs;
                mask = expanddims(offs_k, 1) < K - k * BLOCK_SIZE_K,
                other = zero(DATA_TYPE),
            )
            b = load(
                b_ptrs;
                mask = expanddims(offs_k, 2) < K - k * BLOCK_SIZE_K,
                other = zero(DATA_TYPE),
            )

            accumulator = dot(a, b; allow_tf32 = true) + accumulator

            a_ptrs = a_ptrs + BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptrs + BLOCK_SIZE_K * stride_bk
            return (accumulator, a_ptrs, b_ptrs)
        end

    offs_cm = pid_m * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)

    c_ptrs =
        (c_ptr + stride_cm * expanddims(offs_cm, 2)) + stride_cn * expanddims(offs_cn, 1)
    c_mask = (expanddims(offs_cm, 2) < M) & (expanddims(offs_cn, 1) < N)
    store(c_ptrs, cast(accumulator, DATA_TYPE); mask = c_mask)

    triton_return()
end

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

