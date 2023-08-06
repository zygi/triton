using ThreadsX
using Evolutionary

# Simple dumb optimization from a list of configurations
optimize(fn, dynamic_argument_types, search_space, grid_map_fn) = begin
    into_compiled(cfg) = try
        compile_triton_kernel(fn, dynamic_argument_types, cfg.static_args, grid_map_fn; num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    catch e
        nothing
    end
    kernels = ThreadsX.map(into_compiled, search_space) |> filter(!isnothing)
    
    example_args = (
        CUDA.rand(Float16, SZ, SZ),
        CUDA.rand(Float16, SZ, SZ),
        CUDA.zeros(Float16, SZ, SZ),
        SZ, SZ, SZ, SZ, SZ, SZ   
    )
    

    # @show "Constructing benchmarks"
    do_benchmark(fn, params...) = begin
        b = @benchmarkable begin CUDA.@sync $fn($(params)...) end seconds=0.3
        run(b)
    end 

    # benchmarks = [begin @benchmarkable kern(example_args...) seconds=0.5 end for kern in kernels]
    
    @show "Constructing results"
    results = [do_benchmark(kern, example_args...) for kern in kernels]

    results
end



## TODO document
# Julia doesn't have much support for discrete-valued black box optimizers 

abstract type IntervalSearchSpace end
struct LinearDiscreteSearchSpace <: IntervalSearchSpace
    lb
    ub
end
lowerbound(space::LinearDiscreteSearchSpace) = space.lb
upperbound(space::LinearDiscreteSearchSpace) = space.ub
transformsoln(space::LinearDiscreteSearchSpace, soln) = Int(round(soln))
projectsoln(space::LinearDiscreteSearchSpace, soln) = Float64(soln)

struct Log2DiscreteSearchSpace <: IntervalSearchSpace
    lb::Float64
    ub::Float64
    Log2DiscreteSearchSpace(lb, ub) = new(log2(lb), log2(ub))
end
lowerbound(space::Log2DiscreteSearchSpace) = space.lb
upperbound(space::Log2DiscreteSearchSpace) = space.ub
transformsoln(space::Log2DiscreteSearchSpace, soln) = 2^Int(round(soln))
projectsoln(space::Log2DiscreteSearchSpace, soln) = log2(soln)

@test let sp = Log2DiscreteSearchSpace(32, 256); transformsoln(sp, sp.lb) end == 32.0



get_optim_grid(search_space) = begin
    optim_spaces = [ search_space.num_warps, search_space.num_stages, values(search_space.static_args)... ]
    # map(x -> [lowerbound(x), upperbound(x)], optim_spaces)
    map(lowerbound, optim_spaces), map(upperbound, optim_spaces)
end


reverse_point(x, search_space) = ConfigParams(
    num_warps=transformsoln(search_space.num_warps, x[1]),
    num_stages=transformsoln(search_space.num_stages, x[2]),
    static_args=Dict(zip(keys(search_space.static_args), map(transformsoln, values(search_space.static_args), x[3:end])))
)

orig_space_to_optim_space(params::ConfigParams, search_space) = [
    projectsoln(search_space.num_warps, params.num_warps),
    projectsoln(search_space.num_stages, params.num_stages),
    map(projectsoln, values(search_space.static_args), values(params.static_args))...
]


# TODO don't keep reusing dynamic args
optimize_bbo(matmul_kernel, dynamic_argument_types, search_space, dynamic_args, grid_map_fn; init_assignment=nothing) = begin
    kernel_cache = Dict{ConfigParams, Any}()

    do_experiment(assignment) = begin
        try
            rev = reverse_point(assignment, search_space)
            @show rev
            kern = if haskey(kernel_cache, rev)
                kernel_cache[rev]
            else
                kern = compile_triton_kernel(matmul_kernel, dynamic_argument_types, rev.static_args, grid_map_fn; num_warps=rev.num_warps, num_stages=rev.num_stages)
                kernel_cache[rev] = kern
                kern
            end

            bench = @benchmarkable begin CUDA.@sync $kern($(dynamic_args)...) end seconds=0.3
            # tune!(bench)
            timing = run(bench)
            res = Float64(mean(timing).time)
            @info "$(res / 1e6) ms"
            res
        catch e
            1e9
        end
    end

    init_point = if isnothing(init_assignment)
        mean(get_optim_grid(search_space))
    else
        orig_space_to_optim_space(init_assignment, search_space)
    end

    # @show init_point


    Evolutionary.optimize(do_experiment, BoxConstraints(get_optim_grid(search_space)...), init_point, CMAES(μ=5))
end
