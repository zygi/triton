
get_fn_type(builder, param_types) = begin
    ir_params = CT.CxxRef.([construct_ir_type(t) for t in param_types])
    CT.get_function_ty!(builder, ir_params, Vector{eltype(ir_params)}())
end

_get_ip_and_loc(builder) = (CT.get_loc(builder), CT.get_insertion_point!(builder))
_set_ip_and_loc(builder, ip, loc) = begin CT.set_loc!(builder, ip); CT.restore_insertion_point!(builder, loc) end
@test @ok _set_ip_and_loc(builder, _get_ip_and_loc(builder)...)

get_op_results(op, result_objs) = begin
    count = CT.get_num_results(op)
    @assert count == length(result_objs)
    results = [CT.get_result(op, i) for i in 0:count-1]
    [trval_like(o, r) for (r, o) in zip(results, result_objs)]
end



tupleize(xs::T) where {T <: Tuple} = xs
tupleize(x) = (x,)
# tupleize(nothing) = (x,)

triton_for!(fn, lb::IntoTrVal, ub::IntoTrVal, step::IntoTrVal, init_arguments::Vararg{<:TrVal}) = begin
    bd = get_builder_ref()
    lb = TrVal(lb)
    ub = TrVal(ub)
    step = TrVal(step)

    # bd = lb.builder
    # through_var_types = trtype.(init_arguments)

    for_op = CT.create_for_op!(bd, lb.handle, ub.handle, step.handle, CT.StdVector([ia.handle for ia in init_arguments]))
    ind_var = trval_like(lb, CT.get_induction_var(for_op))
    
    orig_iploc = _get_ip_and_loc(bd)
    
    body_block = CT.create_block!(bd)
    CT.set_insertion_point_to_start!(bd, CT.CxxRef(body_block))

    through_arg_handles = [CT.get_region_iter_arg(for_op, i) for i in 0:CT.get_num_region_iter_arg(for_op)-1]
    @assert length(through_arg_handles) == length(init_arguments) "You must pass the initial value of every loop argument as an argument to the triton_for! function"
    through_args = [trval_like(arg, h) for (h, arg) in zip(through_arg_handles, init_arguments)]

    res_vars = fn(ind_var, through_args...) |> tupleize 

    # @show res_vars
    # @show trtype.(res_vars) .== through_var_types
    # @show objectid.(trtype.(res_vars))
    # @show objectid.(through_var_types)
    
    @assert length(res_vars) == length(through_arg_handles) "You must return the same number of variables as you pass in"
    @assert trtype.(res_vars) == trtype.(init_arguments) "You must return the same types that you take in. Types passed in:\n$(trtype.(init_arguments))\n, types returned: \n$(trtype.(res_vars))\n"
    triton_yield(res_vars...)
    CT.merge_block_before!(CT.CxxRef(body_block), CT.CxxRef(CT.get_body(for_op, 0)))

    _set_ip_and_loc(bd, orig_iploc...)
    return get_op_results(for_op, res_vars)
end

# (TrBlock{Tuple{128, 64}, TrFloat32}, TrBlock{Tuple{128, 32}, TritonPointerType{TrFloat16}}, TrBlock{Tuple{32, 64}, TritonPointerType{TrFloat16}}
#     ) == (TrBlock{Tuple{128, 64}, TrFloat32}, TrBlock{Tuple{128, 32}, TritonPointerType{TrFloat16}}, TrBlock{Tuple{32, 64}, TritonPointerType{TrFloat16}})
# triton_for!(fn, builder, lb::Number, ub::Number, step::Number, init_arguments::Vararg{Tensor}) = begin
#     triton_for!(fn, Tensor(builder, lb), Tensor(builder, ub), Tensor(builder, step), init_arguments...)
# end

# triton_for!(fn, lb::Number, ub::Number, step::Number, init_arguments::Vararg{Tensor}) = begin

#     triton_for!(fn, Tensor(builder, lb), Tensor(builder, ub), Tensor(builder, step), init_arguments...)
# end



# triton_if!(then_thunk, condition::Tensor) = begin
#     bd = condition.builder
#     orig_iploc = _get_ip_and_loc(bd)
#     then_block = CT.create_block!(bd)
#     CT.set_insertion_point_to_start!(bd, CT.CxxRef(then_block))
#     res_vars = then_thunk()
#     res_types::Vector{Tensor} = if isnothing(res_vars)
#         res_vars = Vector{Tensor}()
#     else
#         collect(map(x -> x.type, tupleize(res_vars)))
#     end
#     # @assert typeof(res_vars) == Tuple{Vararg{Tensor}} 
#     triton_yield(bd, res_vars...)
#     _set_ip_and_loc(bd, orig_iploc...)
#     if_op = CT.create_if_op!(bd, CT.StdVector(collect(CT.MLIRTypeAllocated, map(x -> construct_ir_type(bd, x), res_types))), condition.handle, false)
#     CT.merge_block_before!(CT.CxxRef(then_block), CT.CxxRef(CT.get_then_block(if_op)))
#     return get_op_results(bd, if_op, res_types)
# end
