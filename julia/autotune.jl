using ThreadsX
using Evolutionary

# Simple dumb optimization from a list of configurations
optimize(fn, dynamic_argument_types, dynamic_argument_instances, search_space, grid_map_fn) = begin
    into_compiled(cfg) = try
        compile_triton_kernel(fn, dynamic_argument_types, cfg, grid_map_fn)
    catch e
        @info e
        nothing
    end
    
    # First compile all kernels concurrently
    kernels = ThreadsX.map(into_compiled, search_space) |> filter(!isnothing)
    
    do_benchmark(fn, params...) = begin
        b = @benchmarkable begin CUDA.@sync $fn($(params)...) end seconds=0.3
        run(b)
    end 
    [do_benchmark(kern, dynamic_argument_instances...) for kern in kernels]
end



## TODO document
# Julia doesn't have much support for discrete-valued black box optimizers, so we "invent" our own
# We want to map our kernel params into an optimizable R^n box. Most of the params will be integers
# (in which case we map by identity and unmap by rounding)
# or powers-of-2 (in which case we map by log2 and unmap by 2^round)

abstract type IntervalSearchSpace end
struct LinearDiscreteSearchSpace{T} <: IntervalSearchSpace
    lb
    ub
end
LinearDiscreteSearchSpace(::Type{T}, lb, ub) where T = LinearDiscreteSearchSpace{T}(lb, ub)
lowerbound(space::LinearDiscreteSearchSpace) = space.lb
upperbound(space::LinearDiscreteSearchSpace) = space.ub
transformsoln(space::LinearDiscreteSearchSpace{T}, soln) where T = T(round(soln))
projectsoln(space::LinearDiscreteSearchSpace, soln) = Float64(soln)

struct Log2DiscreteSearchSpace{T} <: IntervalSearchSpace
    lb::Float64
    ub::Float64
    Log2DiscreteSearchSpace{T}(lb, ub) where T = new{T}(log2(lb), log2(ub))
end
Log2DiscreteSearchSpace(::Type{T}, lb, ub) where T = Log2DiscreteSearchSpace{T}(lb, ub)
lowerbound(space::Log2DiscreteSearchSpace) = space.lb
upperbound(space::Log2DiscreteSearchSpace) = space.ub
transformsoln(space::Log2DiscreteSearchSpace{T}, soln) where T = T(2^T(round(soln)))
projectsoln(space::Log2DiscreteSearchSpace, soln) = log2(soln)

@test let sp = Log2DiscreteSearchSpace(Int32, 32, 256); transformsoln(sp, sp.lb) end == 32.0

@kwdef struct FullSearchSpace
    num_warps
    num_stages
    static_args::OrderedDict{Symbol, <:IntervalSearchSpace}
end

get_optim_grid(search_space::FullSearchSpace) = begin
    optim_spaces = [ search_space.num_warps, search_space.num_stages, values(search_space.static_args)... ]
    # map(x -> [lowerbound(x), upperbound(x)], optim_spaces)
    map(lowerbound, optim_spaces), map(upperbound, optim_spaces)
end


reverse_point(x, search_space::FullSearchSpace) = ConfigParams(
    num_warps=transformsoln(search_space.num_warps, x[1]),
    num_stages=transformsoln(search_space.num_stages, x[2]),
    static_args=OrderedDict(zip(keys(search_space.static_args), map(transformsoln, values(search_space.static_args), x[3:end])))
)

orig_space_to_optim_space(params::ConfigParams, search_space::FullSearchSpace) = begin
    aligned_param_static_args = OrderedDict(k => params.static_args[k] for k in keys(search_space.static_args))
    [
        projectsoln(search_space.num_warps, params.num_warps),
        projectsoln(search_space.num_stages, params.num_stages),
        map(projectsoln, values(search_space.static_args), values(aligned_param_static_args))...
    ]
end


# TODO don't keep reusing dynamic args
optimize_bbo(matmul_kernel, dynamic_argument_types, search_space::FullSearchSpace, dynamic_args, grid_map_fn; init_assignment=nothing, iterations=nothing, kwargs...) = begin
    kernel_cache = Dict{ConfigParams, Any}()

    do_experiment(assignment) = begin
        try
            rev = reverse_point(assignment, search_space)
            @show rev
            kern = if haskey(kernel_cache, rev)
                kernel_cache[rev]
            else
                kern = compile_triton_kernel(matmul_kernel, dynamic_argument_types, rev.static_args, grid_map_fn;
                    num_warps=rev.num_warps, num_stages=rev.num_stages, assert_enough_shmem=true)
                kernel_cache[rev] = kern
                kern
            end

            bench = @benchmarkable begin CUDA.@sync $kern($(dynamic_args)...) end seconds=0.5
            # tune!(bench)
            timing = run(bench)
            res = Float64(mean(timing).time)
            @info "$(res / 1e6) ms"
            res
        catch e
            if isa(e, InsufficientSharedMemoryError)
                return (e.required - e.available) * 10e9 # penalized by 1 second for every extra byte
            end
            if isa(e, InterruptException)
                throw(e)
            end
            throw(e)
            @show e
            return Inf
        end
    end

    init_point = if isnothing(init_assignment)
        mean(get_optim_grid(search_space))
    else
        orig_space_to_optim_space(init_assignment, search_space)
    end

    # @show init_point

    # optim_grid_lbs, optim_grid_ubs = get_optim_grid(search_space)
    # bboptimize(do_experiment, init_point; SearchRange=collect(zip(optim_grid_lbs, optim_grid_ubs)), StartingMethod=:resampling_inheritance_memetic_search)#, init_point)

    # @show get_optim_grid(search_space)
    opt_result = Evolutionary.optimize(do_experiment, BoxConstraints(get_optim_grid(search_space)...), init_point, CMAES(μ=3), Evolutionary.Options(; iterations, successive_f_tol=0, kwargs...))
    reverse_point(Evolutionary.minimizer(opt_result), search_space), opt_result
end
