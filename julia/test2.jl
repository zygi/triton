include("TritonCxxWrap.jl")
##
using CUDA
using OrderedCollections
const CT = CppTriton

include("global_implicit.jl")
include("helpers.jl")
include("tensor_ops.jl")
include("compilation.jl")
include("ops.jl")

##

# [v for (k, v) in OrderedDict([:a => 1, :b => 2, :c => 3])]

# Make them OrderedDict for debuggability
compile_function(fn, arg_types::OrderedDict, val_args::OrderedDict; print_mlir=false, kwargs...) = begin
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    mod = CppTriton.create_module!(builder)

    fn_op = CT.get_or_insert_function!(builder, mod, "test_name", get_fn_type(builder, values(arg_types)), "public", false)
    CT.push_back!(mod, fn_op)

    entry = CT.add_entry_block!(fn_op)
    insert_pt = CT.get_insertion_block(builder)
    CT.set_insertion_point_to_start!(builder, CT.CxxRef(entry))
    function_args = [CT.arg(CT.CxxRef(entry), i - 1) for i in 1:CT.get_num_arguments(entry)]
    arg_tensors = OrderedDict([arg_sym => Tensor(builder, arg_handle, arg_type) for (arg_handle, (arg_sym, arg_type)) in zip(function_args, arg_types)])

    # tensorise_val(::Val{x}) where {x} = Tensor(builder, x)
    # @assert length(methods(fn)) == 1 "TODO handle multiple methods"
    # _, decls, _, _ =   Base.arg_decl_parts(methods(fn)[1])
    # args = decls[2:end]


    # @show decls

    with_scoped(builder) do; fn(; arg_tensors..., val_args...) end
    # with_scoped(builder) do; fn(arg_tensors..., values(val_args)...) end
    # fn(builder, arg_tensors..., (Tensor(builder, a) for a in val_args)...)

    if print_mlir
        CT.repr(mod) |> print
    end

    cufun, recommended_sm_size = compile_module!(mod, ctx, get_cc_numeric(), "test_name"; kwargs...)
    cufun, ctx, recommended_sm_size # don't drop the context cuz then it will be garbage collected and mess everything up
end

# Int64(-1)

test_kernel(; in_ptr::Tensor, out_ptr::Tensor, n::Tensor, extra_increment::Int32) = begin
    pid = program_id(1)
    my_ptr = in_ptr + pid

    # @show broadcast_impl_shape(Tensor(bd, Int64(1)), [8,])
    # device_print(bd, "test ", expanddims(arange(bd, 0, 4), 1) + expanddims(arange(bd, 0, 4), 2))
    # device_print(bd, "test ", Tensor(bd, Int64(5)) + broadcast_impl_shape(Tensor(bd, Int64(1)), [8,]))

    # device_print(bd, "test ", cdiv(Tensor(bd, 256), Tensor(bd, 64)))
    # device_print(bd, "test ", Tensor(bd, Int64(5)) - Tensor(bd, Int64(1)))
    # device_print(bd, "test2 ", (Tensor(bd, Int64(-1)) < Tensor(bd, Int64(0))) - Tensor(bd, true))
    # device_print(bd, "test2 ", Tensor(bd, CT.get_int1(bd, true), Tint1))
    # device_print(bd, "test2 ", Tensor(bd, true))
    # device_print(bd, "test2 ", one(bd, Tint64))
    # @show construct_ir_type(bd, Tint64)

    # device_print(bd, "test2 ", Tensor(bd, CT.get_all_ones_value(bd, construct_ir_type(bd, Tint64)), Tint64))
    # device_print(bd, "test2 ", Tensor(bd, Int64(1)))
    # one(builder, Tint64)
    # device_print(bd, "test3 ", Tensor(bd, CppTriton.get_int64(bd, -1), Tint64))
    
    accum = zero(Tint32)
    accum2 = zero(Tint32)
    (final_accum, accum2) = triton_for!(Int32(0), Int32(5), Int32(1), accum, accum2) do i, accum, accum2
        in = load(my_ptr; mask=Tensor(true), other=Tensor(0.0f0))
        return (accum + i + Tensor(extra_increment), accum2) #+ Tensor(bd, EXTRA_INCREMENT) #+ cast(in, Tint64)
    end

    store(out_ptr + pid, cast(final_accum, Tfp32))
    triton_return()
end

arg_types = OrderedDict([:in_ptr => PointerTritonType(Tfp32), :out_ptr => PointerTritonType(Tfp32), :n => Tint32])
template_vals = OrderedDict([:extra_increment => Int32(1)])

cufun, ctx, shmem = compile_function(test_kernel, arg_types, template_vals;
    print_mlir=true, print_opt_ttir=true)

# shmem

@test begin
    cufun, ctx = compile_function(test_kernel, arg_types, template_vals;
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



DATA_TYPE = Tfp16
dynamic_argument_types = OrderedDict([
    :a_ptr => PointerTritonType(DATA_TYPE),
    :b_ptr => PointerTritonType(DATA_TYPE),
    :c_ptr => PointerTritonType(DATA_TYPE),
    :M => Tint32,
    :N => Tint32,
    :K => Tint32,
    :stride_ak => Tint32,
    :stride_bn => Tint32,
    :stride_cn => Tint32,
])

static_arg_values = OrderedDict([k => Int32(v) for (k, v) in 
[
    :stride_am => 1,
    :stride_bk => 1, 
    :stride_cm => 1,
    :BLOCK_SIZE_M => 128,
    :BLOCK_SIZE_N => 128,
    :BLOCK_SIZE_K => 32,
    :GROUP_SIZE_M => 8,
]])

matmul_kernel(; # note: all of these are kwargs
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
    accumulator = zeros(DATA_TYPE, [BLOCK_SIZE_M, BLOCK_SIZE_N])
    
    (accumulator, a_ptrs, b_ptrs) =
        # control flow functional for now
        triton_for!(Tensor(Int32(0)), cdiv(K, Tensor(BLOCK_SIZE_K)), Tensor(Int32(1)), 
            accumulator, a_ptrs, b_ptrs) do k, accumulator, a_ptrs, b_ptrs
            a = load(a_ptrs; mask=expanddims(offs_k, 1) < K - k * BLOCK_SIZE_K, other=Float16(0))            
            b = load(b_ptrs; mask=expanddims(offs_k, 2) < K - k * BLOCK_SIZE_K, other=Float16(0))

            accumulator = accumulator + dot(a, b; allow_tf32=true)

            a_ptrs = a_ptrs + BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptrs + BLOCK_SIZE_K * stride_bk
            return (accumulator, a_ptrs, b_ptrs)
        end

    offs_cm = pid_m * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)
    c_ptrs_offset_1 = stride_cm * expanddims(offs_cm, 2)
    offs_cn = pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)    
    c_ptrs_offset_2 = stride_cn * expanddims(offs_cn, 1)

    c_ptrs = c_ptr + c_ptrs_offset_1 + c_ptrs_offset_2
    c_mask = (expanddims(offs_cm, 2) < M) & (expanddims(offs_cn, 1) < N)
    store(c_ptrs, accumulator; mask=c_mask)

    triton_return()
end



compile_and_package(matmul_kernel, dynamic_argument_types, static_arg_values;
    num_warps=8, 
    num_stages=5,
    kwargs...
    ) = begin
    cufun, ctx, SHMEM = compile_function(matmul_kernel, dynamic_argument_types, static_arg_values;
        print_mlir=true,
        num_warps=num_warps, 
        print_opt_ttir=true,
        print_final_ptx=true,
        num_stages=num_stages,
        kwargs...
        )
    
        
end


SZ = 4096

NUM_WARPS = 8
cufun, ctx, SHMEM = compile_function(matmul_kernel, dynamic_argument_types, static_arg_values;
    print_mlir=true,
    num_warps=NUM_WARPS, 
    print_opt_ttir=true,
    print_final_ptx=true,
    num_stages=5
    # print_final_ptx=true
    )
SHMEM 


triton_type_to_julia_ctype(x::ScalarTritonType) = @match x begin
    # Tvoid 
    Tint1 => Cuchar
    Tint8 => Cuchar
    Tuint8 => Cuchar
    Tint16 => Cshort
    Tuint16 => Cushort
    Tint32 => Cint
    Tuint32 => Cuint
    Tint64 => Clonglong
    Tuint64 => Culonglong
    # Tfp8e5  => UInt8
    # Tfp8e4  => Float16
    # Tfp8e4b15 => Float16
    Tfp16 => Cushort # maybe? idk
    # Tbf16 
    Tfp32 => Cfloat
    Tfp64 => Cdouble
end
triton_type_to_julia_ctype(x::PointerTritonType) = CuPtr{triton_type_to_julia_ctype(x.scalar)}

# get_argtypes_t(::Val{triton_type_tuple}) where {triton_type_tuple} = map(x -> zero(triton_type_to_julia_ctype(x)), triton_type_tuple)
# # get_argtypes_t(Val((Tint32,)))
# tpl = (values(dynamic_argument_types)...,)
# get_argtypes_t(Val((PointerTritonType(Tint32),)))

# get_argtypes_t(Val((values(dynamic_argument_types)...,)))


# arg_ctypes = Tuple([triton_type_to_julia_type(x) for x in values(dynamic_argument_types)])

struct TritonKernel{F, M, FF}
    fun::F
    dynarg_ctypes
    dynarg_tritontype_dict
    required_dyn_shmem
    num_warps
    metadata::M
    num_blocks_fn::FF    
end

TritonKernel(fun::F,dynamic_argument_typedict, required_dyn_shmem, num_warps, metadata::M, num_blocks_mapping::FF) where {F, M, FF} = begin
    dynarg_ctypes = [triton_type_to_julia_ctype(x) for x in values(dynamic_argument_typedict)]
    attributes(fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = SHMEM
    TritonKernel{F, M, FF}(fun, dynarg_ctypes, dynamic_argument_typedict, required_dyn_shmem, num_warps, metadata, num_blocks_mapping)
end

_quicktypecheck(::CuPtr{T}, x::CuArray{T}) where T = true
_quicktypecheck(::T, x::T) where T = true
_quicktypecheck(::T, x) where T = false

function (kernel::TritonKernel)(args...; threads::CuDim=(32*kernel.num_warps), blocks::Union{CuDim, Nothing}=nothing)
    @assert length(args) == length(kernel.dynarg_ctypes)
    converted_args = args
    dynarg_mapping = Dict(zip(keys(kernel.dynarg_tritontype_dict), args))
    if isnothing(blocks)
        blocks = kernel.num_blocks_fn(dynarg_mapping, kernel.metadata)
    end
    
    # @show collect(zip(keys(kernel.dynarg_ctypes), args))

    types_match = [_quicktypecheck(x, y) for (x, y) in zip(kernel.dynarg_ctypes, args)]

    @assert all(types_match) "Type mismatch: $types_match"

    # @show blocks

    # converted_args = [cudaconvert(y) for (x, y) in zip(kernel.dynarg_ctypes, args)]
    # @show converted_args
    # @show (kernel.dynarg_ctypes...,)
    CUDA.cudacall(kernel.fun, (kernel.dynarg_ctypes...,), converted_args...; threads=threads, blocks=blocks, shmem=kernel.required_dyn_shmem)
end 

##

tk = TritonKernel(
    cufun,
    dynamic_argument_types,
    SHMEM,
    NUM_WARPS,
    static_arg_values,
    (dyn_args, static_args) -> cdiv(dyn_args[:M], static_args[:BLOCK_SIZE_M]) * cdiv(dyn_args[:N], static_args[:BLOCK_SIZE_N]) 
    )

a = CUDA.rand(Float16, SZ, SZ)
b = CUDA.rand(Float16, SZ, SZ)
out = CUDA.zeros(Float16, SZ, SZ)

# CUDA.cudaconvert(a) |> CUDA.device_pointer

num_blocks = cdiv(SZ, static_arg_values[:BLOCK_SIZE_M]) * cdiv(SZ, static_arg_values[:BLOCK_SIZE_N])

tk(a, b, out, SZ, SZ, SZ, SZ, SZ, SZ)

out - a * b
# Base.to_tuple_type(5)

# exception_ptr = CUDA.create_exceptions!(cufun.mod)
# mm = methods(matmul_kernel)[1]
# hk = CUDA.HostKernel{eltype(mm),arg_ctypes}(mm, cufun, CUDA.KernelState(exception_ptr))

# attributes(hk.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = SHMEM

# dynshmem(hk) = CUDA.attributes(hk.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]

# dynshmem(hk)


CUDA.maxthreads(hk)

CUDA.cufunction(matmul_kernel, )

CUDA.cuFuncSetAttribute(cufun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SHMEM)



##

using BenchmarkTools

function do_mult(a, b, out)
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
end

do_mult(a, b, out)

using NVTX
NVTX.@range "triton" begin do_mult(a, b, out) end



# convert(CuArray{Float16}, a) * convert(CuArray{Float16}, b) - a * b

# a * b - out

# a

using LinearAlgebra
using BenchmarkTools

# triton matmul
println("JuliaTriton bench, DIM=4096, FP16, no tuning")
@benchmark do_mult(a, b, out) setup=( a=CUDA.rand(Float16, SZ, SZ), b=CUDA.rand(Float16, SZ, SZ), out=CUDA.zeros(Float16, SZ, SZ) )

# cuBLAS matmul
println("\n\n")
println("CuBLAS bench, DIM=4096, FP16")
@benchmark begin
    CUDA.@sync LinearAlgebra.mul!(out, a', b')
end setup=( a=CUDA.rand(Float16, SZ, SZ), b=CUDA.rand(Float16, SZ, SZ), out=CUDA.zeros(Float16, SZ, SZ) )

matmul_flops(N) = 2N^3
matmul_flops(SZ) / 0.00282 / 1e12


CUDA.CUBLAS.cublasSetMathMode(CUDA.CUBLAS.handle(), CUDA.CUBLAS.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)



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
    other = zero(builder, base_scalar_type(output_ptr.type))
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

