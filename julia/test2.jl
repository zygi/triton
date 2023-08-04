include("TritonCxxWrap.jl")

using CUDA
const CT = CppTriton

include("global_implicit.jl")
include("helpers.jl")
include("tensor_ops.jl")
include("compilation.jl")
include("ops.jl")

##


compile_function(fn, arg_types, val_args; print_mlir=false, kwargs...) = begin
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    mod = CppTriton.create_module!(builder)

    fn_op = CT.get_or_insert_function!(builder, mod, "test_name", get_fn_type(builder, arg_types), "public", false)
    CT.push_back!(mod, fn_op)

    entry = CT.add_entry_block!(fn_op)
    insert_pt = CT.get_insertion_block(builder)
    CT.set_insertion_point_to_start!(builder, CT.CxxRef(entry))
    function_args = [CT.arg(CT.CxxRef(entry), i - 1) for i in 1:CT.get_num_arguments(entry)]
    arg_tensors = [Tensor(builder, arg_handle, arg_type) for (arg_handle, arg_type) in zip(function_args, arg_types)]

    # tensorise_val(::Val{x}) where {x} = Tensor(builder, x)

    fn(builder, arg_tensors..., val_args...)
    # fn(builder, arg_tensors..., (Tensor(builder, a) for a in val_args)...)

    if print_mlir
        CT.repr(mod) |> print
    end

    cufun, recommended_sm_size = compile_module!(mod, ctx, get_cc_numeric(), "test_name"; kwargs...)
    cufun, ctx, recommended_sm_size # don't drop the context cuz then it will be garbage collected and mess everything up
end

# Int64(-1)

test_kernel(bd, in_ptr::Tensor, out_ptr::Tensor, n::Tensor, extra_increment::Int32) = begin
    pid = program_id(bd, 1)
    my_ptr = in_ptr + pid

    # @show broadcast_impl_shape(Tensor(bd, Int64(1)), [8,])
    # device_print(bd, "test ", expanddims(arange(bd, 0, 4), 1) + expanddims(arange(bd, 0, 4), 2))
    # device_print(bd, "test ", Tensor(bd, Int64(5)) + broadcast_impl_shape(Tensor(bd, Int64(1)), [8,]))

    # device_print(bd, "test ", cdiv(Tensor(bd, 256), Tensor(bd, 64)))
    # device_print(bd, "test ", Tensor(bd, Int64(5)) - Tensor(bd, Int64(1)))
    # device_print(bd, "test2 ", (Tensor(bd, Int64(-1)) < Tensor(bd, Int64(0))) - Tensor(bd, true))
    # device_print(bd, "test2 ", Tensor(bd, CT.get_int1(bd, true), Tint1))
    # device_print(bd, "test2 ", Tensor(bd, true))
    # device_print(bd, "test2 ", triton_one(bd, Tint64))
    # @show construct_ir_type(bd, Tint64)

    # device_print(bd, "test2 ", Tensor(bd, CT.get_all_ones_value(bd, construct_ir_type(bd, Tint64)), Tint64))
    # device_print(bd, "test2 ", Tensor(bd, Int64(1)))
    # triton_one(builder, Tint64)
    # device_print(bd, "test3 ", Tensor(bd, CppTriton.get_int64(bd, -1), Tint64))
    
    accum = Tensor(bd, Int32(0))
    accum2 = Tensor(bd, Int32(0))
    (final_accum, accum2) = triton_for!(bd, Int32(0), Int32(5), Int32(1), accum, accum2) do i, accum, accum2
        in = load(my_ptr; mask=Tensor(bd, true), other=Tensor(bd, 0.0f0))
        return (accum + i + Tensor(bd, extra_increment), accum2) #+ Tensor(bd, EXTRA_INCREMENT) #+ cast(in, Tint64)
    end

    store(out_ptr + pid, cast(final_accum, Tfp32))
    triton_return(bd)
end


cufun, ctx, shmem = compile_function(test_kernel, [ PointerTritonType(Tfp32), PointerTritonType(Tfp32), Tint32 ], [Int32(1)];
    print_mlir=true, print_opt_ttir=true)

# shmem

@test begin
    cufun, ctx = compile_function(test_kernel, [ PointerTritonType(Tfp32), PointerTritonType(Tfp32), Tint32 ], [Int32(1)];
        print_mlir=true, print_opt_ttir=true)
    test_a = CUDA.ones(Float32, 64)
    test_out = CUDA.zeros(Float32, 64)
    CUDA.cudacall(cufun, (CuPtr{Cfloat},CuPtr{Cfloat},Cint),
        # test_a, test_out, 64; blocks=1, threads=32)
        test_a, test_out, 64; blocks=prod(size(test_a)) ÷ 1, threads=32*8)
    # @show test_out
    test_out ≈ 15 .* test_a
    # true
end






matmul_kernel(builder,
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am,
    stride_bk, 
    stride_cm,

    stride_ak,
    stride_bn,
    stride_cn,

    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M) = begin
    
    pid = program_id(builder, 1)
    num_pid_m = cdiv(M, Tensor(builder, BLOCK_SIZE_M))
    num_pid_n = cdiv(N, Tensor(builder, BLOCK_SIZE_N))
    num_pid_in_group = Tensor(builder, GROUP_SIZE_M) * num_pid_n
    group_id = pid ÷ num_pid_in_group
    first_pid_m = group_id * Tensor(builder, GROUP_SIZE_M)
    group_size_m = min(num_pid_m - first_pid_m, Tensor(builder, GROUP_SIZE_M))
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) ÷ group_size_m

    offs_am = (pid_m * Tensor(builder, BLOCK_SIZE_M) + arange(builder, 0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * Tensor(builder, BLOCK_SIZE_N) + arange(builder, 0, BLOCK_SIZE_N)) % N
    offs_k = arange(builder, 0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (expanddims(offs_am, 2) * stride_am + expanddims(offs_k, 1) * Tensor(builder, stride_ak))
    b_ptrs = b_ptr + (expanddims(offs_k, 2) * stride_bk + expanddims(offs_bn, 1) * Tensor(builder, stride_bn))
    accumulator = triton_zeros(builder, [BLOCK_SIZE_M, BLOCK_SIZE_N], Tfp32)
    
    (accumulator, a_ptrs, b_ptrs) =
        triton_for!(Tensor(builder, Int32(0)), cdiv(K, Tensor(builder, BLOCK_SIZE_K)), Tensor(builder, Int32(1)), 
            accumulator, a_ptrs, b_ptrs) do k, accumulator, a_ptrs, b_ptrs
            a = load(a_ptrs; mask=expanddims(offs_k, 1) < K - k * Tensor(builder, BLOCK_SIZE_K), other=triton_zero(builder, Tfp32))            
            b = load(b_ptrs; mask=expanddims(offs_k, 2) < K - k * Tensor(builder, BLOCK_SIZE_K), other=triton_zero(builder, Tfp32))

            accumulator = accumulator + dot(a, b; allow_tf32=true)

            a_ptrs = a_ptrs + Tensor(builder, BLOCK_SIZE_K) * Tensor(builder, stride_ak)
            b_ptrs = b_ptrs + Tensor(builder, BLOCK_SIZE_K) * stride_bk
            return (accumulator, a_ptrs, b_ptrs)
        end

    offs_cm = pid_m * Tensor(builder, BLOCK_SIZE_M) + arange(builder, 0, BLOCK_SIZE_M)
    offs_cn = pid_n * Tensor(builder, BLOCK_SIZE_N) + arange(builder, 0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * expanddims(offs_cm, 2) + Tensor(builder, stride_cn) * expanddims(offs_cn, 1)
    c_mask = (expanddims(offs_cm, 2) < M) & (expanddims(offs_cn, 1) < N)
    store(c_ptrs, accumulator; mask=c_mask)

    triton_return(builder)
end

DATA_TYPE = Tfp32
arg_types = [
    PointerTritonType(Tfp32),
    PointerTritonType(Tfp32),
    PointerTritonType(Tfp32),
    Tint32, Tint32, Tint32,
    Tint32, Tint32, Tint32,
    # Tint32, Tint32, Tint32,
]

SZ = 4096
template_vals = Int32[1, 1, 1, 128, 64, 32, 8]

NUM_WARPS = 4
cufun, ctx, SHMEM = compile_function(matmul_kernel, arg_types, template_vals; print_mlir=true, num_warps=NUM_WARPS, 
    print_opt_ttir=true,
    num_stages=4
    # print_final_ptx=true
    )

SHMEM
    


# SHMEM=98304
CUDA.cuFuncSetAttribute(cufun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SHMEM)


# CUDA.@device_code_sass cufun

a = CUDA.rand(Float32, SZ, SZ)
b = CUDA.rand(Float32, SZ, SZ)
out = CUDA.zeros(Float32, SZ, SZ)

# strides(a)
num_blocks = cdiv(SZ, template_vals[4]) * cdiv(SZ, template_vals[5])
##

using BenchmarkTools


CUDA.@sync CUDA.cudacall(cufun, 
    (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat},Cint,Cint,Cint,Cint,Cint,Cint),#,Cint,Cint,Cint),
    #  a, b, out, SZ, SZ, SZ, 1, SZ, 1, SZ, 1, SZ; 
     a, b, out, SZ, SZ, SZ, SZ, SZ, SZ; 
    #  blocks=prod(size(out)) ÷ NUM_WARPS,
    # blocks=1,
    shmem=SHMEM,
    # blocks=1,
    blocks=num_blocks,
    threads=(32*NUM_WARPS, )
    # threads=1
    )


convert(CuArray{Float16}, a) * convert(CuArray{Float16}, b) - a * b
# a

using LinearAlgebra

@btime begin
    CUDA.@sync LinearAlgebra.mul!(out, a, b)
end


@btime begin
    CUDA.@sync CUDA.cudacall($cufun, 
    (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat},Cint,Cint,Cint,Cint,Cint,Cint),#,Cint,Cint,Cint),
    #  a, b, out, SZ, SZ, SZ, 1, SZ, 1, SZ, 1, SZ; 
     $a, $b, $out, SZ, SZ, SZ, SZ, SZ, SZ; 
    #  blocks=prod(size(out)) ÷ NUM_WARPS,
    # blocks=1,
    shmem=SHMEM,
    # blocks=1,
    blocks=num_blocks,
    threads=(32*NUM_WARPS, )
    # threads=1
    )
end

collect(a) * collect(b) - collect(a * b)

# CuArray((a' * b')') - out
a * b - out


##


exit(0)

cdiv(5, 2)
cdiv(SZ, 64)

out


    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)




# T = Tfp32
# params = [
#     PointerTritonType(T),
#     PointerTritonType(T),
#     Tint64,
# ]

# fn = CT.get_or_insert_function!(builder, mod, "test_name", get_fn_type(builder, params), "public", false)
# CT.push_back!(mod, fn)

# entry = CT.add_entry_block!(fn)
# insert_pt = CT.get_insertion_block(builder)
# CT.set_insertion_point_to_start!(builder, CT.CxxRef(entry))
# function_args = [CT.arg(CT.CxxRef(entry), i - 1) for i in 1:CT.get_num_arguments(entry)]

# arg_tensors = [Tensor(builder, arg_handle, arg_type) for (arg_handle, arg_type) in zip(function_args, params)]
# test_kernel(in_ptr::Tensor, out_ptr::Tensor, n::Tensor) = begin
#     pid = program_id(builder, 1)
#     my_ptr = in_ptr + cast(pid, Tint64)
#     # device_print(builder, "ptr", my_ptr)
#     accum = Tensor(builder, Int64(0))
#     (final_accum,) = triton_for!(0, 5, 1, accum) do i, accum
#         in = load(my_ptr)
#         return accum + i + cast(in, Tint64)
#     end

#     store(out_ptr + cast(pid, Tint64), cast(final_accum, Tfp32))
#     triton_return(builder)
# end

# test_kernel(arg_tensors...)

# pre_if_entry = Tensor(builder, Int32(0))
# pre_if_entry2 = Tensor(builder, Int32(6))
# accum = Tensor(builder, Int64(0))

# triton_for(Tensor(builder, 0), Tensor(builder, 10), Tensor(builder, 1), accum) do i, accum 
#    return accum + i
# end 

CT.repr(mod) |> print


NUM_WARPS = 1

cufun = compile_module!(mod, ctx, get_cc_numeric(), "test_name"; num_warps=NUM_WARPS)


#


test_a = CUDA.ones(Float32, 64)
# test_b = CUDA.ones(Float32, 64)
test_out = CUDA.zeros(Float32, 64)
CUDA.cudacall(cufun, (CuPtr{Cfloat},CuPtr{Cfloat},Cint),
    test_a, test_out, size(test_a, 1); blocks=length(test_a) ÷ NUM_WARPS, threads=32*NUM_WARPS)

test_out


##

methods(() -> 5)[1]

Base.arg_decl_parts(methods(() -> 5)[1])

using Debugger

pre_if_entry = Tensor(builder, Int32(0))
pre_if_entry2 = Tensor(builder, Int32(6))

if_res_1, if_res_2 = triton_if(pre_if_entry == Tensor(builder, Int32(0))) do 
    store(arg_tensors[1], Tensor(builder, Float32(5.5)))
    return (pre_if_entry, pre_if_entry2)
end

if_res_1 + if_res_2

# starting_condition = (cast(arg_tensors[3], Tint64) == Tensor(builder, Int64(0)))


# orig_iploc = _get_ip_and_loc(builder)

# then_block = CT.create_block!(builder)
# CT.set_insertion_point_to_start!(builder, CT.CxxRef(then_block))

# triton_yield(builder, pre_if_entry + pre_if_entry2)

# # arg_tensors[3] == 0

# _set_ip_and_loc(builder, orig_iploc...)
# if_op = CT.create_if_op!(builder, CT.StdVector([construct_ir_type(builder, t.type) for t in [pre_if_entry]]), 
# starting_condition.handle,
#     false)

# CT.merge_block_before!(CT.CxxRef(then_block), CT.CxxRef(CT.get_then_block(if_op)))




CT.repr(mod) |> print



##

CT.get_type(function_args[1])

construct_ir_type(builder, params[1])



# Tensor(builder, function_args[1], params[1])

test_kernel(x_ptr, y_ptr, output_ptr, n_elements) = begin
    pid = program_id(builder, 1)
    block_start = pid * Tensor(builder, Int32(BLOCK_SIZE))
    offsets = block_start + arange(builder, 0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets_int64 = cast(offsets, Tint64)
    other = triton_zero(builder, base_scalar_type(output_ptr.type))
    x = load(x_ptr + offsets_int64; mask=mask, other=other)
    y = load(y_ptr+ offsets_int64; mask=mask, other=other)
    output = x + y
    store(output_ptr + offsets_int64, output; mask=mask)
    triton_return(builder)
end

test_kernel(arg_tensors...)

CT.repr(mod) |> print

NUM_WARPS = 1

cufun = compile_module!(mod, ctx, get_cc_numeric(), "test_name"; num_warps=NUM_WARPS)


##


test_a = CUDA.ones(Float32, 64)
test_b = CUDA.ones(Float32, 64)
test_out = CUDA.zeros(Float32, 64)
CUDA.cudacall(cufun, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat},Cint),
    test_a, test_b, test_out, 64; blocks=prod(size(test_a)) ÷ BLOCK_SIZE, threads=32*NUM_WARPS)

test_a
test_b
test_out |> print

@assert test_out ≈ 2 .* test_a
# using MLStyle
# using IRTools
# using Cassette
# using Test
# using JET








# PointerTritonType(Tint8)


# fn = CT.get_or_insert_function!(builder, mod, "test_name", get_fn_type(builder, [PointerTritonType(Tfp32)]), "public", false)


##

block_args = [CT.arg(CT.CxxRef(entry), i - 1) for i in 1:CT.get_num_arguments(entry)]

cnst = CT.get_fp32(builder, 1.0f0)
CT.create_store!(builder, block_args[1], cnst, CT.CM_NONE, CT.EP_NORMAL)
CT.ret!(builder, Vector{CT.CxxRef{CT.Value}}())

# Vector{CT.Value}()

# ctx


CT.repr(mod) |> print

inline_triton_ir(mod, ctx)

using CUDA

# CUDA.capability(CUDA.device()).major


# get_cc_numeric()


ttir_compute_capability_rewrite(mod, ctx)
rest_of_ttir_pass(mod, ctx)

ttir_to_ttgir(mod, ctx, 4)


# def optimize_ttgir(mod, num_stages, arch):
#     pm = ir.pass_manager(mod.context)
#     pm.enable_debug()
#     pm.add_tritongpu_coalesce_pass()
#     pm.add_tritongpu_remove_layout_conversions_pass()
#     if isinstance(arch, int):
#         pm.add_tritongpu_accelerate_matmul_pass(arch)
#     pm.add_tritongpu_remove_layout_conversions_pass()
#     pm.add_tritongpu_optimize_dot_operands_pass()
#     pm.add_tritongpu_pipeline_pass(num_stages)
#     pm.add_tritongpu_prefetch_pass()
#     pm.add_tritongpu_optimize_dot_operands_pass()
#     pm.add_tritongpu_remove_layout_conversions_pass()
#     pm.add_tritongpu_decompose_conversions_pass()
#     pm.add_tritongpu_reorder_instructions_pass()
#     pm.add_cse_pass()
#     pm.add_symbol_dce_pass()
#     pm.run(mod)
#     return mod

optimize_ttgir(mod, ctx, 3, get_cc_numeric()) # TODO num_stages


llir = ttgir_to_llir(mod, get_cc_numeric())

print(llir)

# TODO fix version gates

ptx = llir_to_ptx(llir, get_cc_numeric());
print(ptx);

joinpath(dirname(dirname(dirname(pathof(CUDA)))), "bin", "ptxas")

ptx_to_cubin(ptx, arch) = begin
    # find the path to the "ptxas" executable, e.g. using which
    ptxas = read(`which ptxas`, String)[1:end-1]
    CT.compile_ptx_to_cubin(ptx, ptxas, arch)
end

cubin = ptx_to_cubin(ptx, get_cc_numeric())
# write to tmp file
tmpfile = tempname() * ".so"
open(tmpfile, "w") do f
    write(f, cubin)
end

# load the so file
# Base.Libc.Libdl.dlopen(tmpfile, Base.Libc.RTLD_LAZY)


# using LLVM


# @dispose ctx=Context() begin
#     Base.parse(LLVM.Module, String(llir))
# end

# llir



methods(LLVM.name)

compilation_res = ptx_compile(String(ptx), "test_name")


cf = link(compilation_res)

test_array = CUDA.zeros(Float32, 10)

CUDA.cudacall(cf, (CuPtr{Cfloat},), test_array)

test_array

# CUDA.ptxas()

##

entry

CT.repr(entry)

CT.CxxRef(entry)

CT._show(CT.CxxRef(entry))

