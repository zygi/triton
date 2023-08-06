inline_triton_ir!(mod, ctx) = begin
    pm = CT.PassManager(CT.CxxPtr(ctx))
    CT.enable_debug!(pm)
    CT.add_inliner_pass!(pm)
    CT.run!(pm, mod)
    mod
end

ttir_compute_capability_rewrite!(mod, ctx) = begin
    # For hardware without support, we must rewrite all load/store
    # with block (tensor) pointers into tensors of pointers
    pm = CT.PassManager(CT.CxxPtr(ctx))
    CT.enable_debug!(pm)
    # if _is_cuda(arch):
    # pm.add_rewrite_tensor_pointer_pass(arch)
    # pm.run(mod)
    CT.add_rewrite_tensor_pointer_pass!(pm, get_cc_numeric())
    CT.run!(pm, mod)
    return mod
end


rest_of_ttir_pass!(mod, ctx) = begin
    # pm = ir.pass_manager(mod.context)
    # pm.enable_debug()
    # pm.add_inliner_pass()
    # pm.add_triton_combine_pass()
    # pm.add_canonicalizer_pass()
    # pm.add_reorder_broadcast_pass()
    # pm.add_cse_pass()
    # pm.add_licm_pass()
    # pm.add_symbol_dce_pass()
    # pm.run(mod)
    pm = CT.PassManager(CT.CxxPtr(ctx))
    CT.enable_debug!(pm)
    CT.add_inliner_pass!(pm)
    CT.add_triton_combine_pass!(pm)
    CT.add_canonicalizer_pass!(pm)
    CT.add_reorder_broadcast_pass!(pm)
    CT.add_cse_pass!(pm)
    CT.add_licm_pass!(pm)
    CT.add_symbol_dce_pass!(pm)
    CT.run!(pm, mod)
    return mod
end

get_cc_numeric() = begin
    if CUDA.has_cuda()
        cap = CUDA.capability(CUDA.device())
        return cap.major * 10 + cap.minor
    else
        return 0
    end
end

ttir_to_ttgir!(mod, ctx, num_warps) = begin
    pm = CT.PassManager(CT.CxxPtr(ctx))
    CT.enable_debug!(pm)
    CT.add_convert_triton_to_tritongpu_pass!(pm, num_warps, 32) # threads per warp
    CT.run!(pm, mod)
    return mod
end

optimize_ttgir!(mod, ctx, num_stages, arch) = begin
    pm = CT.PassManager(CT.CxxPtr(ctx))
    CT.enable_debug!(pm)
    CT.add_tritongpu_coalesce_pass!(pm)
    CT.add_tritongpu_remove_layout_conversions_pass!(pm)
    # if isa(arch, Int)

    CT.add_tritongpu_accelerate_matmul_pass!(pm, arch)
    # end
    CT.add_tritongpu_remove_layout_conversions_pass!(pm)
    CT.add_tritongpu_optimize_dot_operands_pass!(pm)
    CT.add_tritongpu_pipeline_pass!(pm, num_stages)
    CT.add_tritongpu_prefetch_pass!(pm)
    CT.add_tritongpu_optimize_dot_operands_pass!(pm)
    CT.add_tritongpu_remove_layout_conversions_pass!(pm)
    CT.add_tritongpu_decompose_conversions_pass!(pm)
    CT.add_tritongpu_reorder_instructions_pass!(pm)
    CT.add_cse_pass!(pm)
    CT.add_symbol_dce_pass!(pm)
    CT.run!(pm, mod)
    return mod
end

ttgir_to_llir!(mod, arch) = CT.translate_triton_gpu_to_llvmir(mod, arch, false)
llir_to_ptx!(llir::AbstractString, arch) = CT.translate_llvmir_to_ptx(llir, arch, 81)




function ptx_compile(ptx::String, entry::String)
    needs_cudadevrt = false


    # prepare invocations of CUDA compiler tools
    ptxas_opts = String[]
    nvlink_opts = String[]
    ## debug flags
    if Base.JLOptions().debug_level == 1
        push!(ptxas_opts, "--generate-line-info")
    elseif Base.JLOptions().debug_level >= 2
        push!(ptxas_opts, "--device-debug")
        push!(nvlink_opts, "--debug")
    end
    ## relocatable device code
    if needs_cudadevrt
        push!(ptxas_opts, "--compile-only")
    end

    # use the highest device capability that's supported by CUDA. note that we're allowed
    # to query this because the compilation cache is sharded by the device context.
    # XXX: put this in the CompilerTarget to avoid device introspection?
    #      on the other hand, GPUCompiler doesn't care about the actual device capability...
    dev = device()
    caps = filter(toolchain_cap -> toolchain_cap <= capability(dev), CUDA.cuda_compat().cap)
    cap = maximum(caps)
    # NOTE: we should already have warned about compute compatibility mismatches
    #       during TLS state set-up.
    arch = "sm_$(cap.major)$(cap.minor)"

    # compile to machine code
    # NOTE: we use tempname since mktemp doesn't support suffixes, and mktempdir is slow
    ptx_input = tempname(cleanup=false) * ".ptx"
    ptxas_output = tempname(cleanup=false) * ".cubin"
    write(ptx_input, ptx)

    # we could use the driver's embedded JIT compiler, but that has several disadvantages:
    # 1. fixes and improvements are slower to arrive, by using `ptxas` we only need to
    #    upgrade the toolkit to get a newer compiler;
    # 2. version checking is simpler, we otherwise need to use NVML to query the driver
    #    version, which is hard to correlate to PTX JIT improvements;
    # 3. if we want to be able to use newer (minor upgrades) of the CUDA toolkit on an
    #    older driver, we should use the newer compiler to ensure compatibility.
    append!(ptxas_opts, [
        "--verbose",
        "--gpu-name", arch,
        "--output-file", ptxas_output,
        "-v",
        ptx_input
    ])
    proc, log = CUDA.run_and_collect(`$(CUDA.ptxas()) $ptxas_opts`)
    log = strip(log)
    if !success(proc)
        reason = proc.termsignal > 0 ? "ptxas received signal $(proc.termsignal)" :
                 "ptxas exited with code $(proc.exitcode)"
        msg = "Failed to compile PTX code ($reason)"
        msg *= "\nInvocation arguments: $(join(ptxas_opts, ' '))"
        if !isempty(log)
            msg *= "\n" * log
        end
        msg *= "\nIf you think this is a bug, please file an issue and attach $(ptx_input)"
        error(msg)
    elseif !isempty(log)
        @debug "PTX compiler log:\n" * log
    end
    rm(ptx_input)

    # link device libraries, if necessary
    #
    # this requires relocatable device code, which prevents certain optimizations and
    # hurts performance. as such, we only do so when absolutely necessary.
    # TODO: try LTO, `--link-time-opt --nvvmpath /opt/cuda/nvvm`.
    #       fails with `Ignoring -lto option because no LTO objects found`
    if needs_cudadevrt
        nvlink_output = tempname(cleanup=false) * ".cubin"
        append!(nvlink_opts, [
            "--verbose", "--extra-warnings",
            "--arch", arch,
            "--library-path", dirname(libcudadevrt),
            "--library", "cudadevrt",
            "--output-file", nvlink_output,
            ptxas_output
        ])
        proc, log = run_and_collect(`$(nvlink()) $nvlink_opts`)
        log = strip(log)
        if !success(proc)
            reason = proc.termsignal > 0 ? "nvlink received signal $(proc.termsignal)" :
                     "nvlink exited with code $(proc.exitcode)"
            msg = "Failed to link PTX code ($reason)"
            msg *= "\nInvocation arguments: $(join(nvlink_opts, ' '))"
            if !isempty(log)
                msg *= "\n" * log
            end
            msg *= "\nIf you think this is a bug, please file an issue and attach $(ptxas_output)"
            error(msg)
        elseif !isempty(log)
            @debug "PTX linker info log:\n" * log
        end
        rm(ptxas_output)

        image = read(nvlink_output)
        rm(nvlink_output)
    else
        image = read(ptxas_output)
        rm(ptxas_output)
    end

    return (image, entry=entry)
end

function link(compiled)
    # load as an executable kernel object
    ctx = CUDA.context()
    mod = CuModule(compiled.image)
    CuFunction(mod, compiled.entry)
end

const mlir_to_cubin_cache = Cache(String, FileSystemCacheManager("/tmp/juliatriton_cache", Any),
x -> replace(base64encode(sha512(x)), '/' => '_', '=' => '_', '+' => '_')
)

using Serialization
function compile_module!(mod, ctx, arch::Integer, entry::AbstractString;
        num_stages=3, num_warps=32, print_opt_ttir=false, print_opt_ttgir=false, print_final_llir=false, print_final_ptx=false, bypass_cache=true)
    mlir_ast_string = String(CT.string(mod))

    compiled, recommended_shared_memory = if !bypass_cache && haskey(mlir_to_cubin_cache, mlir_ast_string)
        @info "Using cached compilation result"
        mlir_to_cubin_cache[mlir_ast_string]
    else
        inline_triton_ir!(mod, ctx)
        ttir_compute_capability_rewrite!(mod, ctx)
        rest_of_ttir_pass!(mod, ctx)
        # if print_opt_ttir
        #     @info "Optimized TTIR:\n" * CT.repr(mod)
        # end
        if print_opt_ttir
            @info "Optimized TTIR:\n" * CT.repr(mod)
        end


        # mod = mktemp() do path, io
        #     write(io, OVERRIDE_CODE)
        #     # write(io, CT.repr(mod))
        #     close(io)
        #     CT.parse_mlir_module(path, ctx)
        # end

        # mod = CT.parse_mlir_module
    
        ttir_to_ttgir!(mod, ctx, num_warps)
        optimize_ttgir!(mod, ctx, num_stages, arch)
        if print_opt_ttgir
            @info "Optimized TTGIR:\n" * CT.repr(mod)
        end
    
        llir = ttgir_to_llir!(mod, arch)
        if print_final_llir
            @info "Final LLIR:\n" * llir
        end
    
        recommended_shared_memory = CT.get_shared_memory_size(mod)
    
        ptx = llir_to_ptx!(llir, arch)
        if print_final_ptx && isinteractive()
            # @info "Final PTX:\n" * ptx
            clipboard(ptx)
        end
    
        # serialize("matmul_kernel_ptx.jlso", String(ptx))
    
        compiled = ptx_compile(String(ptx), entry)
    
        mlir_to_cubin_cache[mlir_ast_string] = (compiled, recommended_shared_memory)
        compiled, recommended_shared_memory
    end

    cufunction = link(compiled)
    cufunction, recommended_shared_memory
end


# Make arguments OrderedDict for debuggability
compile_function(fn, arg_types::OrderedDict, val_args::OrderedDict; print_immediate_ttir=false, kwargs...) = begin
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    mod = CppTriton.create_module!(builder)

    fn_op = CT.get_or_insert_function!(builder, mod, "test_name", get_fn_type(builder, values(arg_types)), "public", false)
    for (idx, arg) in enumerate(values(arg_types))
        # is_block(arg) && 
        CT.set_arg_attr!(fn_op, idx-1, "tt.divisibility", 16)
        # CT.add_argument!(fn_op, arg)
    end
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

    if print_immediate_ttir
        CT.repr(mod) |> print
    end

    cufun, recommended_sm_size = compile_module!(mod, ctx, get_cc_numeric(), "test_name"; kwargs...)
    cufun, ctx, recommended_sm_size # don't drop the context cuz then it will be garbage collected and mess everything up
end


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



triton_type_to_julia_type(x::ScalarTritonType) = @match x begin
    # Tvoid 
    Tint1 => Bool
    Tint8 => Int8
    Tuint8 => UInt8
    Tint16 => Int16
    Tuint16 => UInt16
    Tint32 => Int32
    Tuint32 => UInt32
    Tint64 => Int64
    Tuint64 => UInt64
    # Tfp8e5  => UInt8
    # Tfp8e4  => Float16
    # Tfp8e4b15 => Float16
    Tfp16 => Float16 # maybe? idk
    # Tbf16 
    Tfp32 => Float32
    Tfp64 => Float64
end
triton_type_to_julia_type(x::PointerTritonType) = CuPtr{triton_type_to_julia_type(x.scalar)}

struct TritonKernel{F, M, FF}
    fun::F
    dynarg_types
    dynarg_tritontype_dict
    required_dyn_shmem
    num_warps
    metadata::M
    num_blocks_fn::FF    
end


TritonKernel(fun::F,dynamic_argument_typedict, required_dyn_shmem, num_warps, metadata::M, num_blocks_fn::FF) where {F, M, FF} = begin
    # required_dyn_shmem = 49152
    shared_total = CUDA.attribute(CUDA.device(), CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    if required_dyn_shmem > shared_total
        throw("The kernel requires $required_dyn_shmem bytes of dynamic shared memory, but only $shared_total bytes are available. Adjust your kernel or compilation parameters.")
    end

    shared_optin_max = attribute(CUDA.device(), CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    if (required_dyn_shmem > 49152 && shared_optin_max > 49152)
        CUDA.cuFuncSetCacheConfig(fun, CUDA.CU_FUNC_CACHE_PREFER_SHARED)
        # shared_total = attribute(CUDA.device(), CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        shared_static = attributes(fun)[CUDA.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        attributes(fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shared_optin_max - shared_static
        # int shared_total, shared_static;
        # CUDA_CHECK(cuDeviceGetAttribute(
        #     &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        #     device));
        # CUDA_CHECK(cuFuncGetAttribute(&shared_static,
        #                               CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
        # CUDA_CHECK(
            # cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            #                    shared_optin - shared_static));
    end
    
    dynarg_types = [triton_type_to_julia_type(x) for x in values(dynamic_argument_typedict)]
    attributes(fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = required_dyn_shmem

    # check mapping, I wonder if we can do this statically
    num_blocks_fn_ret_type = Base.return_types(num_blocks_fn, (eltype(dynarg_types), M))
    @assert !isempty(num_blocks_fn_ret_type) "TODO num_blocks_fn must return Int64, instead got $num_blocks_fn_ret_type"

    TritonKernel{F, M, FF}(fun, dynarg_types, dynamic_argument_typedict, required_dyn_shmem, num_warps, metadata, num_blocks_fn)
end

_quicktypecheck(::CuPtr{T}, x::CuArray{T}) where T = true
_quicktypecheck(::T, x::T) where T = true
_quicktypecheck(::T, x) where T = false

_convert_to_callarg(::Type{CuPtr{T}}, x::CuArray{T}) where T = pointer(x)
_convert_to_callarg(::Type{T}, x) where T = convert(T, x)


# NamedTuple{(:a, :b)}((1, "")) |> flatten


struct ConfigParams
    num_warps::Int
    num_stages::Int
    static_args::OrderedDict{Symbol, Any}
end
# @functor ConfigParams
ConfigParams(;num_warps, num_stages, static_args) = ConfigParams(num_warps, num_stages, static_args)
flatten(p::ConfigParams, spec) = [p.num_warps, p.num_stages, values(p.static_args)...]

cp = ConfigParams(1, 2, OrderedDict(:a => 1, :b => 2))

function (kernel::TritonKernel)(args...; threads::CuDim=(32*kernel.num_warps), blocks::Union{CuDim, Nothing}=nothing)
    @assert length(args) == length(kernel.dynarg_types)
    converted_args = args
    dynarg_mapping = Dict(zip(keys(kernel.dynarg_tritontype_dict), args))
    if isnothing(blocks)
        blocks = kernel.num_blocks_fn(dynarg_mapping, kernel.metadata)
    end

    # TODO check mat arg divisibility by 16
    
    converted_args = [_convert_to_callarg(x, y) for (x, y) in zip(kernel.dynarg_types, args)]
    CUDA.cudacall(kernel.fun, (kernel.dynarg_types...,), converted_args...; threads=threads, blocks=blocks, shmem=kernel.required_dyn_shmem)

    # TODO should we sync here?
end 


function compile_triton_kernel(kernel_fn, dynamic_argument_types, static_arg_values, grid_fn; num_warps=4, kwargs...)
    cufun, _, shmem = compile_function(kernel_fn, dynamic_argument_types, static_arg_values; num_warps, kwargs...)
    TritonKernel(cufun, dynamic_argument_types, shmem, num_warps, static_arg_values, grid_fn)
end

compile_triton_kernel(kernel_fn, dynamic_argument_types, config_params::ConfigParams, grid_fn; kwargs...) = 
    compile_triton_kernel(kernel_fn, dynamic_argument_types, config_params.static_args, grid_fn; num_stages=config_params.num_stages, num_warps=config_params.num_warps, kwargs...)