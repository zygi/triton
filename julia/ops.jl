
get_fn_type(builder, param_types) = begin
    ir_params = CT.CxxRef.([construct_ir_type(builder, t) for t in param_types])
    CT.get_function_ty!(builder, ir_params, Vector{eltype(ir_params)}())
end

_get_ip_and_loc(builder) = (CT.get_loc(builder), CT.get_insertion_point!(builder))
_set_ip_and_loc(builder, ip, loc) = begin CT.set_loc!(builder, ip); CT.restore_insertion_point!(builder, loc) end
@test @wcok _set_ip_and_loc(builder, _get_ip_and_loc(builder)...)

get_op_results(builder, op, result_types) = begin
    count = CT.get_num_results(op)
    @assert count == length(result_types)
    results = [CT.get_result(op, i) for i in 0:count-1]
    [Tensor(builder, r, t) for (r, t) in zip(results, result_types)]
end



tupleize(xs::T) where {T <: Tuple} = xs
tupleize(x) = (x,)
# tupleize(nothing) = (x,)

triton_for!(fn, lb::IntoTensor, ub::IntoTensor, step::IntoTensor, init_arguments::Vararg{Tensor}) = begin
    bd = get_builder_ref()
    lb = Tensor(lb)
    ub = Tensor(ub)
    step = Tensor(step)

    # bd = lb.builder
    through_var_types = trtype.(init_arguments)

    for_op = CT.create_for_op!(bd, lb.handle, ub.handle, step.handle, CT.StdVector([ia.handle for ia in init_arguments]))
    # for_op = CT.create_for_op!(bd, lb.handle, ub.handle, step.handle, CT.StdVector{CT.ValueAllocated}([ia.handle for ia in init_arguments]))
    ind_var = Tensor(bd, CT.get_induction_var(for_op), trtype(lb))
    
    orig_iploc = _get_ip_and_loc(bd)
    
    body_block = CT.create_block!(bd)
    CT.set_insertion_point_to_start!(bd, CT.CxxRef(body_block))

    through_arg_handles = [CT.get_region_iter_arg(for_op, i) for i in 0:CT.get_num_region_iter_arg(for_op)-1]
    @assert length(through_arg_handles) == length(init_arguments)
    through_args = [Tensor(bd, h, t) for (h, t) in zip(through_arg_handles, through_var_types)]

    res_vars = fn(ind_var, through_args...) |> tupleize 

    # @show res_vars
    # @show trtype.(res_vars) .== through_var_types
    # @show objectid.(trtype.(res_vars))
    # @show objectid.(through_var_types)
    
    @assert length(res_vars) == length(through_var_types)
    @assert trtype.(res_vars) == through_var_types "Expected \n$(through_var_types)\n  got \n$(trtype.(res_vars))\n"
    triton_yield(res_vars...)
    CT.merge_block_before!(CT.CxxRef(body_block), CT.CxxRef(CT.get_body(for_op, 0)))

    _set_ip_and_loc(bd, orig_iploc...)
    return get_op_results(bd, for_op, through_var_types)
end

(TritonBlockType{Tuple{128, 64}, TrFloat32}, TritonBlockType{Tuple{128, 32}, TritonPointerType{TrFloat16}}, TritonBlockType{Tuple{32, 64}, TritonPointerType{TrFloat16}}
    ) == (TritonBlockType{Tuple{128, 64}, TrFloat32}, TritonBlockType{Tuple{128, 32}, TritonPointerType{TrFloat16}}, TritonBlockType{Tuple{32, 64}, TritonPointerType{TrFloat16}})
# triton_for!(fn, builder, lb::Number, ub::Number, step::Number, init_arguments::Vararg{Tensor}) = begin
#     triton_for!(fn, Tensor(builder, lb), Tensor(builder, ub), Tensor(builder, step), init_arguments...)
# end

# triton_for!(fn, lb::Number, ub::Number, step::Number, init_arguments::Vararg{Tensor}) = begin

#     triton_for!(fn, Tensor(builder, lb), Tensor(builder, ub), Tensor(builder, step), init_arguments...)
# end



triton_if!(then_thunk, condition::Tensor) = begin
    bd = condition.builder
    orig_iploc = _get_ip_and_loc(bd)
    then_block = CT.create_block!(bd)
    CT.set_insertion_point_to_start!(bd, CT.CxxRef(then_block))
    res_vars = then_thunk()
    res_types::Vector{Tensor} = if isnothing(res_vars)
        res_vars = Vector{Tensor}()
    else
        collect(map(x -> x.type, tupleize(res_vars)))
    end
    # @assert typeof(res_vars) == Tuple{Vararg{Tensor}} 
    triton_yield(bd, res_vars...)
    _set_ip_and_loc(bd, orig_iploc...)
    if_op = CT.create_if_op!(bd, CT.StdVector(collect(CT.MLIRTypeAllocated, map(x -> construct_ir_type(bd, x), res_types))), condition.handle, false)
    CT.merge_block_before!(CT.CxxRef(then_block), CT.CxxRef(CT.get_then_block(if_op)))
    return get_op_results(bd, if_op, res_types)
end
