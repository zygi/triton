# matmul


matmul_kernel_rowmajor_opt(; # note: all of these are kwargs
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ak,
    stride_bn,
    stride_cn,
    stride_am,
    stride_bk, 
    stride_cm,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M) = begin

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
    a_ptrs = a_ptr + (expanddims(offs_am, 2) * stride_am + expanddims(offs_k, 1) * stride_ak)
    b_ptrs = b_ptr + (expanddims(offs_k, 2) * stride_bk + expanddims(offs_bn, 1) * stride_bn)
    accumulator = zeros(Tfp32, [BLOCK_SIZE_M, BLOCK_SIZE_N])
    
    (accumulator, a_ptrs, b_ptrs) =
        # control flow functional for now
        triton_for!(Tensor(Int32(0)), cdiv(K, Tensor(BLOCK_SIZE_K)), Tensor(Int32(1)), 
            accumulator, a_ptrs, b_ptrs) do k, accumulator, a_ptrs, b_ptrs
            a = load(a_ptrs; mask=expanddims(offs_k, 1) < K - k * BLOCK_SIZE_K, other=zero(DATA_TYPE))            
            b = load(b_ptrs; mask=expanddims(offs_k, 2) < K - k * BLOCK_SIZE_K, other=zero(DATA_TYPE))

            accumulator = dot(a, b; allow_tf32=true) + accumulator 

            a_ptrs = a_ptrs + BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptrs + BLOCK_SIZE_K * stride_bk
            return (accumulator, a_ptrs, b_ptrs)
        end

    offs_cm = pid_m * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)

    c_ptrs = (c_ptr + stride_cm * expanddims(offs_cm, 2)) +  stride_cn * expanddims(offs_cn, 1)
    c_mask = (expanddims(offs_cm, 2) < M) & (expanddims(offs_cn, 1) < N)
    store(c_ptrs, cast(accumulator, DATA_TYPE); mask=c_mask)

    triton_return()
end
