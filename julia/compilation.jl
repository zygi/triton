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

using Serialization
function compile_module!(mod, ctx, arch::Integer, entry::AbstractString;
        num_stages=3, num_warps=32, print_opt_ttgir=false, print_opt_ttir=false, print_final_llir=false, print_final_ptx=false)
    inline_triton_ir!(mod, ctx)
    ttir_compute_capability_rewrite!(mod, ctx)
    rest_of_ttir_pass!(mod, ctx)
    if print_opt_ttir
        @info "Optimized TTIR:\n" * CT.repr(mod)
    end

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


    serialize("matmul_kernel_ptx.jlso", String(ptx))

    compiled = ptx_compile(String(ptx), entry)
    cufunction = link(compiled)
    cufunction, recommended_shared_memory
end
