
# Tensor

struct Tensor{T <: TritonType}
    builder
    handle
    type::T

    Tensor{T}(builder, handle, type::T) where {T<:TritonType} = begin
        # hacky way to double check types since it's annoying to extract a julia-rep of a type from a handle
        t1 = CT.get_type(handle)
        t2 = construct_ir_type(builder, type)
        t1_repr = CT.repr(CT.CxxRef(t1))
        t2_repr = CT.repr(CT.CxxRef(t2))
        @assert t1_repr == t2_repr "Type mismatch when wrapping Tensor: expected $t1_repr, got $t2_repr"
        
        new(builder, handle, type)
    end
end
Tensor(builder, handle, type::T) where {T<:TritonType} = Tensor{T}(builder, handle, type)

Tensor(x::T) where {T<:Tensor} = x

Base.show(io::IO, x::Tensor) = begin
    hd = CT.repr(CT.CxxRef(x.handle))
    print(io, "Tensor($hd)")
    
end
##
Tensor(builder, b::Bool) = Tensor(builder, CppTriton.get_int1(builder, b), Tint1)
Tensor(builder, x::Int64) = Tensor(builder, CppTriton.get_int64(builder, x), Tint64)
Tensor(builder, x::UInt64) = Tensor(builder, CppTriton.get_int64(builder, reinterpret(Int64, x)), Tuint64)
Tensor(builder, x::Int32) = Tensor(builder, CppTriton.get_int32(builder, x), Tint32)
Tensor(builder, x::UInt32) = Tensor(builder, CppTriton.get_int32(builder, reinterpret(Int32, x)), Tuint32)

@test @wcok Tensor(builder, Int64(1))
@test @wcok Tensor(builder, Int64(-2^63))
@test @wcok Tensor(builder, Int64(2^63-1))
@test @wcok Tensor(builder, typemax(UInt64))

Tensor(builder, x::Float32) = Tensor(builder, CppTriton.get_fp32(builder, x), Tfp32)
Tensor(builder, x::Float64) = Tensor(builder, CppTriton.get_fp64(builder, x), Tfp64)


IntoTensor = Union{Bool, Int64, UInt64, Int32, UInt32, Float32, Float64}
Tensor(x::IntoTensor) = begin Tensor(get_builder_ref(), x) end

@test @wcok begin
    with_scoped(builder) do 
        Tensor(5.0)
    end
end

is_floating(x::Tensor) = is_floating(x.type)
is_integer(x::Tensor) = is_integer(x.type)
is_pointer(x::Tensor) = is_pointer(x.type)
numel(x::Tensor) = numel(x.type)

Base.size(x::Tensor) = size(x.type)
Base.size(x::Tensor, dim) = size(x.type, dim)

cast(input::Tensor, dst_ty::Union{TritonType, PointerTritonType}) = begin
    builder = input.builder
    src_ty = input.type
    if is_block(src_ty) && !is_block(dst_ty)
        dst_ty = BlockTritonType(dst_ty, src_ty.dims)
    end
    (src_ty == dst_ty) && return input

    src_sca_ty = scalar_type(src_ty)
    dst_sca_ty = scalar_type(dst_ty)

    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    # if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
    #     (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()):
    #         return tl.tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder)),
    #                         dst_ty)
    if (is_fp8(src_sca_ty) && is_floating(dst_sca_ty)) ||
        (is_floating(src_sca_ty) && is_fp8(dst_sca_ty))
            return Tensor(builder, CppTriton.create_fp_to_fp!(builder, input.handle, construct_ir_type(builder, dst_ty)), construct_ir_type(builder, dst_ty))
    end
    
    # bf16 <=> (not fp32)
    # if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
    #     (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
    #     return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)
    if (src_sca_ty == Tfp16 && dst_sca_ty != Tfp32) ||
        (src_sca_ty == Tbf16 && dst_sca_ty != Tfp32)
        return cast(cast(input, Tfp32), dst_sca_ty)
    end

        # Standard floating types' casting: truncation
    #   fp64 => fp32, fp16, bf16
    #   fp32 => fp16, bf16
    # truncate_fp = src_sca_ty.is_floating() and \
    #     dst_sca_ty.is_floating() and \
    #     src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    # if truncate_fp:
    #     return tl.tensor(builder.create_fp_trunc(input.handle,
    #                                              dst_ty.to_ir(builder)),
    #                      dst_ty)

    truncate_fp = is_floating(src_sca_ty) &&
        is_floating(dst_sca_ty) &&
        primitive_bandwidth(src_sca_ty) > primitive_bandwidth(dst_sca_ty)
    if truncate_fp
        return Tensor(builder, CppTriton.create_fp_trunc!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
    end

        # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    # ext_fp = src_sca_ty.is_floating() and \
    #     dst_sca_ty.is_floating() and \
    #     src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    # if ext_fp:
    #     return tl.tensor(builder.create_fp_ext(input.handle,
    #                                            dst_ty.to_ir(builder)),
    #                      dst_ty)
    ext_fp = is_floating(src_sca_ty) &&
        is_floating(dst_sca_ty) &&
        primitive_bandwidth(src_sca_ty) < primitive_bandwidth(dst_sca_ty)
    if ext_fp
        return Tensor(builder, CppTriton.create_fp_ext!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
    end

    # Casting between integer types
    # if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
    #     (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
    #      sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
    #      if dst_sca_ty.is_bool():
    #          ty = input.dtype.to_ir(builder)
    #          _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
    #          return not_equal(input, _0, builder)
    #      else:
    #          return tl.tensor(builder.create_int_cast(input.handle,
    #                                                   dst_ty.to_ir(builder), sign_extend),
    #                           dst_ty)

    if is_integer(src_sca_ty) && is_integer(dst_sca_ty) &&
        (primitive_bandwidth(src_sca_ty) != primitive_bandwidth(dst_sca_ty) || is_signed(src_sca_ty) != is_signed(dst_sca_ty))
        sign_extend = is_signed(src_sca_ty) && !is_bool(src_sca_ty)
        if is_bool(dst_sca_ty)
            ty = construct_ir_type(builder, dst_ty)
            _0 = Tensor(builder, CppTriton.get_null_value!(builder, ty), dst_ty)
            # TODO add not_equal
            return not_equal(input, _0)
        else
            return Tensor(builder, CppTriton.create_int_cast!(builder, input.handle, construct_ir_type(builder, dst_ty), sign_extend), dst_ty)
        end
    end

    # Casting standard floating types to integer types
    # if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
    #     if dst_sca_ty.is_bool():
    #         ty = input.dtype.to_ir(builder)
    #         _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
    #         return not_equal(input, _0, builder)
    #     elif dst_sca_ty.is_int_signed():
    #         return tl.tensor(builder.create_fp_to_si(input.handle,
    #                                                  dst_ty.to_ir(builder)),
    #                          dst_ty)
    #     else:
    #         return tl.tensor(builder.create_fp_to_ui(input.handle,
    #                                                  dst_ty.to_ir(builder)),
    #                          dst_ty)
    if is_standard_floating(src_sca_ty) && is_integer(dst_sca_ty)
        if is_bool(dst_sca_ty)
            ty = construct_ir_type(builder, dst_ty)
            _0 = Tensor(builder, CppTriton.get_null_value!(builder, ty), dst_ty)
            # TODO add not_equal
            return not_equal(input, _0)
        elseif is_signed(dst_sca_ty)
            return Tensor(builder, CppTriton.create_fp_to_si!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
        else
            return Tensor(builder, CppTriton.create_fp_to_ui!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
        end
    end

    # # Casting integer types to standard floating types
    # if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
    #     if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
    #         return tl.tensor(builder.create_ui_to_fp(input.handle,
    #                                                     dst_ty.to_ir(builder)),
    #                             dst_ty)
    #     else:
    #         return tl.tensor(builder.create_si_to_fp(input.handle,
    #                                                     dst_ty.to_ir(builder)),
    #                             dst_ty)
    if is_integer(src_sca_ty) && is_standard_floating(dst_sca_ty)
        if is_bool(src_sca_ty) || !is_signed(src_sca_ty)
            return Tensor(builder, CppTriton.create_ui_to_fp!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
        else
            return Tensor(builder, CppTriton.create_si_to_fp!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
        end
    end

#     # Casting pointer types to integer types
#     if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
#         bitwidth = dst_sca_ty.int_bitwidth
#         if bitwidth == 64:
#             return tl.tensor(builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)),
#                                 dst_ty)
#         if bitwidth == 1:
#             return not_equal(cast(input, tl.int64, builder),
#                                 tl.tensor(builder.get_int64(0), tl.int64),
#                                 builder)
    if is_pointer(src_sca_ty) && is_integer(dst_sca_ty)
        bitwidth = primitive_bandwidth(dst_sca_ty)
        if bitwidth == 64
            return Tensor(builder, CppTriton.create_ptr_to_int!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
        elseif bitwidth == 1
            # what is this for
            @assert false
            return not_equal(cast(input, Tint64), Tensor(builder, CppTriton.get_int64!(builder, 0), Tint64))
        end
    end

    # # Casting integer types to pointer types
    # if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
    #     return tl.tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)
    if is_integer(src_sca_ty) && is_pointer(dst_sca_ty)
        return Tensor(builder, CppTriton.create_int_to_ptr!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
    end
    
    # # Casting pointer types to pointer types
    # if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
    #     return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)
    if is_pointer(src_sca_ty) && is_pointer(dst_sca_ty)
        return Tensor(builder, CppTriton.create_bitcast!(builder, input.handle, construct_ir_type(builder, dst_ty)), dst_ty)
    end

    throw("Unsupported cast from $src_ty to $dst_ty")    
end

@test_throws "Unsupported" @wc cast(Tensor(builder, 5.0), PointerTritonType(Tfp64))
@test @wc (cast(Tensor(builder, 3), PointerTritonType(Tint64)); true)

arange(builder, start::Integer, endd::Integer) = begin
    start = Int32(start)
    endd = Int32(endd)
    shape = [endd - start,]
    ret_ty = BlockTritonType(Tint32, shape)
    Tensor(builder, CT.create_make_range!(builder, start, endd), ret_ty)
end
@declare_alias_using_scoped arange
@test @wcok arange(builder, 0, 5)

full(builder, dims::Vector{Int64}, value::Tensor) = begin
    @assert numel(value) == 1 "Value must be a scalar"
    # value = cast(value, dtype)
    ret_ty = BlockTritonType(value.type, dims)
    Tensor(builder, CT.create_splat!(builder, value.handle, collect(dims)), ret_ty)
end

full(builder, dims::Vector{Int64}, value::T, dtype) where T = begin
    @assert is_scalar(dtype) "Value's target type must be a scalar"
    value_ir = if iszero(value)
        Tensor(builder, CT.get_null_value!(builder, construct_ir_type(builder, dtype)), dtype)
    else
        tensor = Tensor(builder, value)
        cast(tensor, dtype)
    end
    Tensor(builder, CT.create_splat!(builder, value_ir.handle, collect(dims)), BlockTritonType(dtype, dims))
end

@test @wcok begin
    t1 = Tensor(builder, 1.0)
    full(builder, [10,20,], t1)

    t2 = full(builder, [10,], 5, Tint32)
end

function triton_broadcast(lhs::Tensor, rhs::Tensor)
    @match lhs.type, rhs.type begin
        (BlockTritonType(lhs_sca_ty, lhs_dims, _), BlockTritonType(rhs_sca_ty, rhs_dims, _)) => begin
            if lhs_dims == rhs_dims
                return lhs, rhs
            end
            target_shape = Int64[]
            for (l, r) in zip(lhs_dims, rhs_dims)
                if l == r
                    push!(target_shape, l)
                elseif l == 1
                    push!(target_shape, r)
                elseif r == 1
                    push!(target_shape, l)
                else
                    throw("Trying to broadcast incompatible shapes, shapes lhs: $lhs_dims, rhs: $rhs_dims")
                end
            end
            lhs = if lhs_dims != target_shape
                Tensor(lhs.builder, CT.create_broadcast!(lhs.builder, lhs.handle, collect(target_shape)), BlockTritonType(lhs_sca_ty, target_shape))
            else
                lhs
            end

            rhs = if rhs_dims != target_shape
                Tensor(rhs.builder, CT.create_broadcast!(rhs.builder, rhs.handle, collect(target_shape)), BlockTritonType(rhs_sca_ty, target_shape))
            else
                rhs
            end

            return lhs, rhs

            # throw("TODO, shapes lhs: $lhs_dims, rhs: $rhs_dims")
        end
        (BlockTritonType(lhs_sca_ty, lhs_dims, _), rhs_ty) => begin
            rhs_block_ty = BlockTritonType(rhs_ty, lhs_dims)
            lhs, Tensor(lhs.builder, CT.create_splat!(lhs.builder, rhs.handle, collect(rhs_block_ty.dims)), rhs_block_ty)
        end
        (lhs_ty, BlockTritonType(rhs_sca_ty, rhs_dims, _)) => begin
            lhs_block_ty = BlockTritonType(lhs_ty, rhs_dims)
            Tensor(rhs.builder, CT.create_splat!(rhs.builder, lhs.handle, collect(lhs_block_ty.dims)), lhs_block_ty), rhs
        end
        (_, _) => (lhs, rhs)
    end 
end

triton_broadcast(a, b, c) = begin
    a, b = triton_broadcast(a, b)
    b, c = triton_broadcast(b, c)
    a, b = triton_broadcast(a, b)
    a, b, c
end

@test @wc begin
    t1 = Tensor(builder, 1.0)
    t2 = full(builder, [10, 20], 5, Tint32)
    
    t1, t2 = triton_broadcast(t1, t2)
    t1.type.dims == t2.type.dims
end

shapes_match(x::Tensor, y::Tensor) = size(x) == size(y)

types_shapes_match(x::Tensor, y::Tensor) = begin
    (x.type == y.type) && (size(x) == size(y))
end
@test @wc types_shapes_match(Tensor(builder, 1.0), Tensor(builder, 2.0))
@test @wc !types_shapes_match(Tensor(builder, 1.0), Tensor(builder, 2))
@test @wc !types_shapes_match(Tensor(builder, 1.0), Tensor(builder, 1.0f0))


_replace_ptr_with_int32(x::ScalarTritonType) = x
_replace_ptr_with_int32(x::PointerTritonType) = Tint32
_replace_ptr_with_int32(x::BlockTritonType) = BlockTritonType(_replace_ptr_with_int32(x.scalar), x.dims)

# identifies integers with pointers
types_shapes_match_uptopointer(x::Tensor, y::Tensor) = begin
    x_type = _replace_ptr_with_int32(x.type)
    y_type = _replace_ptr_with_int32(y.type)

    (x_type == y_type) && (size(x) == size(y))
    # @match x_type, y_type begin
    #     (BlockTritonType(x_sca_ty, dims, _), BlockTritonType(y_sca_ty, dims, _)) => begin
    #         base_scalar_type(x_sca_ty) == base_scalar_type(y_sca_ty) && (size(x) == size(y))
    #     end
    #     (_, _) => base_scalar_type(x.type) == base_scalar_type(y.type) && (size(x) == size(y))
    # end
    # (base_scalar_type(x) ) && (size(x) == size(y))
end
@test @wc types_shapes_match_uptopointer(Tensor(builder, Int32(4)), cast(Tensor(builder, 3), PointerTritonType(Tint32)))
@test @wc !types_shapes_match_uptopointer(Tensor(builder, Int64(4)), cast(Tensor(builder, 3), PointerTritonType(Tint64)))
@test @wc !types_shapes_match_uptopointer(full(builder, [2,], 4, Tint64), cast(Tensor(builder, 3), PointerTritonType(Tint64)))


points_to_type(ptr::Tensor, val::Tensor) = @match ptr.type, val.type begin
    (BlockTritonType(PointerTritonType(lhs_ty), lhs_dims, _), BlockTritonType(rhs_ty::ScalarTritonType, rhs_dims, _)) => begin
        return lhs_ty == rhs_ty && lhs_dims == rhs_dims
    end
    # (BlockTritonType(PointerTritonType(lhs_ty), _, _), rhs_ty::ScalarTritonType) => begin
    #     return lhs_ty == rhs_ty
    # end
    (PointerTritonType(lhs_ty), rhs_ty::ScalarTritonType) => begin
        return lhs_ty == rhs_ty
    end
end
@test @wc points_to_type(cast(Tensor(builder, 3), PointerTritonType(Tint32)), Tensor(builder, Int32(3)))
@test @wc !points_to_type(cast(Tensor(builder, 3), PointerTritonType(Tint32)), Tensor(builder, Int64(3)))

program_id(builder, axis) = try
    Tensor(builder, CT.create_get_program_id!(builder, axis-1), Tint32)
catch e
    if isa(e, BoundsError)
        throw("Axis must be between 1 and 3")
    end
    throw(e)
end

@test @wcok program_id(builder, 1)

num_programs(builder, axis) = Tensor(builder, CT.create_get_num_programs!(builder, axis-1), Tint32)
@test @wcok num_programs(builder, 1)


# for now, let's have no implicit casting



Base.:+(lhs::Tensor, rhs::Tensor) = begin
    lhs, rhs = triton_broadcast(lhs, rhs)
    @assert types_shapes_match_uptopointer(lhs, rhs) "Types and shapes must match, got x: $lhs, y: $rhs"

    # offset + ptr
    # ptr + offset
    if is_pointer(rhs.type) && !is_pointer(lhs.type)
        lhs, rhs = rhs, lhs
    end
    if is_pointer(lhs.type)
        return Tensor(lhs.builder, CT.create_addptr!(lhs.builder, lhs.handle, rhs.handle), lhs.type)
    end

    # float + float
    if is_floating(lhs.type) && is_floating(rhs.type)
        return Tensor(lhs.builder, CT.create_fadd!(lhs.builder, lhs.handle, rhs.handle), lhs.type)
    end

    if is_integer(lhs.type) && is_integer(rhs.type)
        return Tensor(lhs.builder, CT.create_add!(lhs.builder, lhs.handle, rhs.handle), lhs.type)
    end

    throw("Can't add $lhs and $rhs")
end
@test @wcok Tensor(builder, 1.0) + Tensor(builder, 2.0)
@test @wcok full(builder, [5,], 1.0, Tfp32) + full(builder, [5,], 2.0, Tfp32)
@test_throws "" @wc Tensor(builder, 1.0) + Tensor(builder, 2)
@test_throws "" @wc Tensor(builder, 1.0) + full(builder, [2,], 2.0, Tfloat64)

triton_zero(builder, ty) = Tensor(builder, CT.get_null_value(builder, construct_ir_type(builder, ty)), ty)
@test @wcok triton_zero(builder, Tint32)

triton_all_ones(builder, ty) = Tensor(builder, CT.get_all_ones_value(builder, construct_ir_type(builder, ty)), ty)
@test @wcok triton_all_ones(builder, Tint32)

triton_one(builder, ty) = @match ty begin
    Tint64 => Tensor(builder, Int64(1))
    Tint32 => Tensor(builder, Int32(1))
    Tfp64 => Tensor(builder, Float64(1.0))
    Tfp32 => Tensor(builder, Float32(1.0))
    Tuint64 => Tensor(builder, UInt64(1))
    Tuint32 => Tensor(builder, UInt32(1))
    Tuint8 => Tensor(builder, UInt8(1))
    Tint1 => Tensor(builder, true)
end
@test @wcok triton_one(builder, Tint32)

Base.:-(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    @assert types_shapes_match_uptopointer(x, y) "Types and shapes must match, got x: $x, y: $y"
    if is_pointer(x)
        return Tensor(x.builder, CT.create_addptr!(x.builder, x.handle, (-y).handle), x.type)
    end
    if is_floating(x.type) && is_floating(y.type)
        return Tensor(x.builder, CT.create_fsub!(x.builder, x.handle, y.handle), x.type)
    end
    if is_integer(x.type) && is_integer(y.type)
        return Tensor(x.builder, CT.create_sub!(x.builder, x.handle, y.handle), x.type)
    end
    throw("Can't subtract $x and $y")
end
@test @wcok Tensor(builder, 1.0) - Tensor(builder, 2.0)
@test_throws "" @wc Tensor(builder, 1.0) - Tensor(builder, 5)


Base.:-(x::Tensor) = begin
    is_pointer(x) && throw("Can't negate a pointer")
    triton_zero(x.builder, x.type) - x
end 
@test @wcok -Tensor(builder, 1.0)
@test @wcok cast(Tensor(builder, 1), PointerTritonType(Tint64)) - Tensor(builder, Int32(2))

Base.:*(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    @assert types_shapes_match(x, y) "Types and shapes must match, got x: $x and y: $y"
    if is_floating(x.type) && is_floating(y.type)
        return Tensor(x.builder, CT.create_fmul!(x.builder, x.handle, y.handle), x.type)
    end
    if is_integer(x.type) && is_integer(y.type)
        return Tensor(x.builder, CT.create_mul!(x.builder, x.handle, y.handle), x.type)
    end
    throw("Can't multiply $x and $y")
end
@test @wcok Tensor(builder, 1.0) * Tensor(builder, 2.0)

Base.:/(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    @assert shapes_match(x, y) "Shapes must match, got x: $x and y: $y"
    if is_floating(x.type) && is_integer(y.type)
        y = cast(y, x.type)
    elseif is_integer(x.type) && is_floating(y.type)
        x = cast(x, y.type)
    elseif is_floating(x.type) && is_floating(y.type)
        if fp_mantissa_width(x.type) > fp_mantissa_width(y.type)
            y = cast(y, x.type)
        else
            x = cast(x, y.type)
        end
    else
        # TODO think about int/int
        throw("Can't divide $x and $y")
    end
    return Tensor(x.builder, CT.create_fdiv!(x.builder, x.handle, y.handle), x.type)
end
@test @wc (Tensor(builder, 1.0) / Tensor(builder, 2.0f0)).type == Tfp64
@test @wcok Tensor(builder, 1.0) / Tensor(builder, 2)

Base.div(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    @assert types_shapes_match(x, y) "Shapes must match, got x: $x and y: $y"
    if is_integer(x.type) && is_integer(y.type)
       if is_signed(x.type)
            return Tensor(x.builder, CT.create_sdiv!(x.builder, x.handle, y.handle), x.type)
        else
            return Tensor(x.builder, CT.create_udiv!(x.builder, x.handle, y.handle), x.type)
        end
    end
    throw("Can't divide $x and $y")    
end
@test @wcok Tensor(builder, 1) ÷ Tensor(builder, 2)
@test_throws "" @wc Tensor(builder, 1.0) ÷ Tensor(builder, 2.0)

cdiv(x::Tensor, y::Tensor) = (x + y - triton_one(x.builder, x.type)) ÷ y
cdiv(x::T, y::U) where {T <: Integer, U <: Integer} = ((x + y) - one(U)) ÷ y
@test @wcok cdiv(Tensor(builder, 5), Tensor(builder, 2))

Base.rem(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    @assert types_shapes_match(x, y) "Types and shapes must match"
    if is_integer(x.type) && is_integer(y.type)
        @assert is_signed(x.type) == is_signed(y.type) "Types must be both signed or both unsigned"
        if is_signed(x.type)
            return Tensor(x.builder, CT.create_srem!(x.builder, x.handle, y.handle), x.type)
        else
            return Tensor(x.builder, CT.create_urem!(x.builder, x.handle, y.handle), x.type)
        end
    end
    # TODO think about float % float
    throw("Can't divide $x and $y")    
end
@test @wcok Tensor(builder, 5) % Tensor(builder, 2)

base_eq(x::Tensor, y::Tensor) = x == y
base_neq(x::Tensor, y::Tensor) = x != y

# Comparison ops: <, <=, >, >=, ==, !=

COMPARISON_OPS = [
    (:<, :create_fcmpOLT!, :create_icmpSLT!, :create_icmpULT!),
    (:<=, :create_fcmpOLE!, :create_icmpSLE!, :create_icmpULE!),
    (:>, :create_fcmpOGT!, :create_icmpSGT!, :create_icmpUGT!),
    (:>=, :create_fcmpOGE!, :create_icmpSGE!, :create_icmpUGE!),
    (:(==), :create_fcmpOEQ!, :create_icmpEQ!, :create_icmpEQ!),
    (:!=, :create_fcmpUNE!, :create_icmpNE!, :create_icmpNE!)
]
for (op_name, float_op, signed_op, unsigned_op) in COMPARISON_OPS
    eval(quote
        Base.$op_name(x::Tensor, y::Tensor) = begin
            x, y = triton_broadcast(x, y)
            @assert types_shapes_match(x, y) "Types and shapes must match, got x: $x and y: $y"
            return_ty = change_scalar_type(x.type, Tint1)

            if is_floating(x.type) && is_floating(y.type)
                return Tensor(x.builder, CT.$float_op(x.builder, x.handle, y.handle), return_ty)
            end
            if is_integer(x.type) && is_integer(y.type)
                if is_signed(x.type)
                    return Tensor(x.builder, CT.$signed_op(x.builder, x.handle, y.handle), return_ty)
                else
                    return Tensor(x.builder, CT.$unsigned_op(x.builder, x.handle, y.handle), return_ty)
                end
            end
            if is_pointer(x.type) && is_pointer(y.type)
                # TODO decide once and for all if pointers are signed or unsigned 
                return Tensor(x.builder, CT.$signed_op(x.builder, x.handle, y.handle), return_ty)
            end
            throw("Can't compare $x and $y")
        end

        @test @wc ($op_name(Tensor(builder, 1.0), Tensor(builder, 2.0))).type == Tint1
        @test @wc ($op_name(Tensor(builder, Int32(1)), Tensor(builder, Int32(1)))).type == Tint1
        @test @wc ($op_name(Tensor(builder, UInt32(1)), Tensor(builder, UInt32(1)))).type == Tint1
        @test @wc ($op_name(full(builder, [5,], 5.0, Tfp64), full(builder, [5,], 4.0, Tfp64))).type == BlockTritonType(Tint1, [5,])
    end)
end

# bitwise and, or, xor
for (op_name, create_op) in [(:&, :create_and!), (:|, :create_or!), (:^, :create_xor!)]
    eval(quote
        Base.$op_name(x::Tensor, y::Tensor) = begin
            x, y = triton_broadcast(x, y)
            @assert types_shapes_match(x, y) "Types and shapes must match, got x: $x and y: $y"
            @assert is_integer(x.type) && is_integer(y.type) "Both operands must be integers, got x: $(x.type) and y: $(y.type)"
            Tensor(x.builder, CT.$create_op(x.builder, x.handle, y.handle), x.type)
        end

        @test @wc ($op_name(Tensor(builder, Int32(1)), Tensor(builder, Int32(1)))).type == Tint32
    end)
end


expanddims(x::Tensor, axis::Int) = begin
    @assert is_block(x.type)
    @assert axis >= 1 && axis <= length(x.type.dims)+1 "Axis must be in range [1, length(x.type.dims)+1]"
    new_shape = similar(x.type.dims, length(x.type.dims) + 1)
    new_shape[1:(axis-1)] .= x.type.dims[1:axis-1]
    new_shape[axis] = 1
    new_shape[axis + 1:end] .= x.type.dims[axis:end]
    new_type = BlockTritonType(x.type.scalar, new_shape)
    Tensor(x.builder, CT.create_expand_dims!(x.builder, x.handle, axis-1), new_type)
end
@test @wc size(expanddims(full(builder, [2, 3], 1.0, Tfp64), 2)) == [2, 1, 3]

broadcast_impl_shape(x::Tensor, shape) = let
    shape = collect(Int64, shape)
    builder = x.builder
    @match x.type begin
        BlockTritonType(scalar_ty, src_shape, _) => begin
            @assert length(src_shape) == length(shape) "Shapes must have the same length, got $src_shape and $shape"
            new_shape = similar(src_shape, length(src_shape))
            for i in 1:length(src_shape)
                if src_shape[i] == 1
                    new_shape[i] = shape[i]
                else
                    @assert src_shape[i] == shape[i] "Shapes must be compatible, got $src_shape and $shape"
                    new_shape[i] = src_shape[i]
                end
            end
            ret_ty = BlockTritonType(scalar_ty, new_shape)
            Tensor(x.builder, CT.create_broadcast!(builder, x.handle, new_shape), ret_ty)
        end
        _ => begin
            ret_ty = BlockTritonType(x.type, shape)
            Tensor(x.builder, CT.create_splat!(builder, x.handle, shape), ret_ty)
        end
    end
end
@test @wc size(broadcast_impl_shape(Tensor(builder, 1.0), [2, 3])) == [2, 3]
@test @wc begin
    res = broadcast_impl_shape(full(builder, [2, 1, 3], 1.0, Tfp32), [2, 5, 3])
    size(res) == [2, 5, 3] && scalar_type(res.type) == Tfp32
end

triton_zeros(builder, dims, ty) = broadcast_impl_shape(triton_zero(builder, ty), dims)
@test @wcok triton_zeros(builder, [2,], Tint32)

_string_to_load_cache_modifier(x) = begin
    if x == ".ca" return CppTriton.CM_CA end
    if x == ".cg" return CppTriton.CM_CG end
    if x == "" return CppTriton.CM_NONE end
    throw("Unknown cache modifier $x")
end

_string_to_store_cache_modifier(x) = begin
    if x == ".wb" return CppTriton.CM_WB end
    if x == ".wt" return CppTriton.CM_WT end
    if x == ".cg" return CppTriton.CM_CG end
    if x == ".cs" return CppTriton.CM_CS end
    if x == "" return CppTriton.CM_NONE end
    throw("Unknown cache modifier $x")
end

_string_to_eviction_policy(x) = begin
    if x == "evict_last" return CppTriton.EP_EVICT_LAST end
    if x == "evict_first" return CppTriton.EP_EVICT_FIRST end
    if x == "" return CppTriton.EP_NORMAL end
    throw("Unknown eviction policy $x")
end

_load_legacy(ptr::Tensor, mask::Union{Tensor, Nothing}, other::Union{Tensor, Nothing}, cache, eviction, is_volatile) = begin
    @assert is_pointer(ptr.type) "ptr must be a pointer or a pointer block"
    @assert primitive_bandwidth(base_scalar_type(ptr.type)) != 1 "TODO ptr can't point to bools"
    @assert isnothing(mask) == isnothing(other) "mask and other must be either both nothing or both tensors"

    # ptr, val = triton_broadcast(ptr, val)
    if !isnothing(mask)
        # if is_block(other.type)
        #     ptr, mask, other = triton_broadcast(ptr, mask, other)
        # else
        #     ptr, mask = triton_broadcast(ptr, mask)
        # end
        ptr, mask, other = triton_broadcast(ptr, mask, other)
            # mask, other = triton_broadcast(mask, other)
    end

    @assert isnothing(mask) || (is_bool(mask.type) && shapes_match(ptr, mask)) "mask must be a boolean tensor with the same shape as ptr, got $mask and $ptr"
    @assert isnothing(other) || points_to_type(ptr, other) "other must have the same type as ptr, got $other and $ptr"

    result_ty = if is_block(ptr.type) BlockTritonType(base_scalar_type(ptr.type), ptr.type.dims) else base_scalar_type(ptr.type) end

    if isnothing(mask)
        Tensor(ptr.builder, CT.create_load!(ptr.builder, ptr.handle, cache, eviction, is_volatile), result_ty)
    else
        Tensor(ptr.builder, CT.create_masked_load!(ptr.builder, ptr.handle, mask.handle, other.handle, cache, eviction, is_volatile), result_ty)
    end
end

@test @wcok begin
    ptr = cast(Tensor(builder, 1), PointerTritonType(Tfp32))
    mask = Tensor(builder, true)
    other = Tensor(builder, 2.0f0)
    _load_legacy(ptr, mask, other, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    _load_legacy(ptr, nothing, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
end
@test @wcok begin
    ptr = cast(full(builder, [2, 3], 5, Tint64), PointerTritonType(Tint64))
    mask = full(builder, [2, 3], true, Tint1)
    other = full(builder, [2, 3], 2, Tint64)
    _load_legacy(ptr, nothing, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    _load_legacy(ptr, mask, other, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
end


load(ptr::Tensor; mask::Union{Tensor, Nothing}=nothing, other::Union{Tensor, Nothing}=nothing, cache="", eviction="", is_volatile=false) = begin
    _load_legacy(ptr, mask, other, _string_to_load_cache_modifier(cache), _string_to_eviction_policy(eviction), is_volatile)
end



_store_legacy(ptr::Tensor, val::Tensor, mask::Union{Tensor, Nothing}, cache, eviction) = begin
    @assert is_pointer(ptr.type) "ptr must be a pointer or a pointer block"
    @assert primitive_bandwidth(base_scalar_type(ptr.type)) != 1 "TODO ptr can't point to bools"

    if !isnothing(mask) && is_block(mask.type)
        ptr, val, mask = triton_broadcast(ptr, val, mask)
    else
        ptr, val = triton_broadcast(ptr, val)
    end
    
    @assert points_to_type(ptr, val) "ptr must be ptr<T> where val is <T>, got ptr: $ptr and val: $val"
    @assert isnothing(mask) || (scalar_type(mask.type) == Tint1) "mask must be a boolean tensor, got $mask"
    @assert isnothing(mask) || shapes_match(ptr, mask) "mask must be a boolean tensor with the same shape as ptr, got $mask and $ptr"
    
    if isnothing(mask)
        CT.create_store!(ptr.builder, ptr.handle, val.handle, cache, eviction)
    else
        CT.create_masked_store!(ptr.builder, ptr.handle, val.handle, mask.handle, cache, eviction)
    end
end

@test @wcok begin
    ptr = cast(Tensor(builder, 1), PointerTritonType(Tfp32))
    val = Tensor(builder, 2.0f0)
    mask = Tensor(builder, true)
    _store_legacy(ptr, val, mask, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    _store_legacy(ptr, val, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
end
@test @wcok begin
    ptr = cast(full(builder, [2, 3], 5, Tint64), PointerTritonType(Tint64))
    val = full(builder, [2, 3], 2, Tint64)
    mask = full(builder, [2, 3], true, Tint1)
    _store_legacy(ptr, val, mask, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    _store_legacy(ptr, val, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
end

store(ptr::Tensor, val::Tensor; mask::Union{Tensor, Nothing}=nothing, cache="", eviction="") = begin
    _store_legacy(ptr, val, mask, _string_to_store_cache_modifier(cache), _string_to_eviction_policy(eviction))
end


triton_return(builder) = begin
    CT.ret!(builder, CT.CxxRef{CT.Value}[])
end
@test @wcok triton_return(builder)

device_print(builder, prefix, args...) = begin
    handles = collect(map(x -> Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(x.handle)), args))
    CT.create_print!(builder, prefix, handles)
end
@test @wcok device_print(builder, "hello", full(builder, [2, 3], 5, Tint64), full(builder, [2, 3], 5, Tint64))


subtypes(CT.ValueAllocated)

triton_yield(builder, vs::Vararg{Tensor}) = begin
    if isempty(vs)
        CT.create_yield_op!(builder, CT.CxxRef{CT.Value}[])
    else
        CT.create_yield_op!(builder, collect(map(x -> Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(x.handle)), vs)))
    end
end
@test @wcok triton_yield(builder, full(builder, [2, 3], 5, Tint64), full(builder, [2, 3], 5, Tint64))


triton_where(cond::Tensor, x::Tensor, y::Tensor) = begin
    cond = cast(cond, Tint1)

    if is_block(cond.type)
        cond, x, y = triton_broadcast(cond, x, y)
    else
        x, y = triton_broadcast(x, y)
    end

    @assert types_shapes_match(x, y)
    Tensor(cond.builder, CT.create_select!(cond.builder, cond.handle, x.handle, y.handle), x.type)
end
@test @wcok begin
    cond = full(builder, [2, 3], true, Tint1)
    x = full(builder, [2, 3], 5, Tint64)
    y = full(builder, [2, 3], 2, Tint64)
    triton_where(cond, x, y)
end
@test @wcok begin
    cond = Tensor(builder, true)
    x = full(builder, [2, 3], 5, Tint64)
    y = full(builder, [2, 3], 2, Tint64)
    triton_where(cond, x, y)
end


Base.min(x::Tensor, y::Tensor) = triton_where(x < y, x, y)
Base.max(x::Tensor, y::Tensor) = triton_where(x > y, x, y)
@test @wcok begin
    x = full(builder, [2, 3], 5, Tint64)
    y = full(builder, [2, 3], 2, Tint64)
    min(x, y)
    max(x, y)
end


dot(x::Tensor, y::Tensor; allow_tf32 = true) = begin
    @assert is_block(x.type) && is_block(y.type) "x and y must be block tensors, got $x and $y"
    @assert base_scalar_type(x.type) == base_scalar_type(y.type) "x and y must have the same type, got $x and $y"
    @assert length(size(x)) == 2 && length(size(y)) == 2 "x and y must be 2D tensors, got $x and $y"
    @assert size(x, 2) == size(y, 1) "x and y must have compatible shapes, got $x and $y"
    @assert size(x, 1) >= 16 && size(x, 2) >= 16 && size(y, 1) >= 16 && size(y, 2) >= 16 "x and y must be at least 16x16, got $x and $y"
    @assert is_floating(x.type) && is_floating(y.type) "TODO x and y must be floating point tensors, got $x and $y"
    accum_type = @match base_scalar_type(x.type), base_scalar_type(y.type) begin
        (Tfp32, _) => Tfp32
        (Tbf16, _) => Tfp32
        (_, Tfp16) => Tfp16
        (_, _) => Tfp32
    end
    accum = triton_zero(x.builder, accum_type)
    M = size(x, 1)
    N = size(y, 2)
    accum_splat = CT.create_splat!(x.builder, accum.handle, [M, N])
    ret_ty = BlockTritonType(accum_type, [M, N])
    Tensor(x.builder, CT.create_dot!(x.builder, x.handle, y.handle, accum_splat, allow_tf32), ret_ty)
end
@test @wc begin
    x = full(builder, [16, 32], 5, Tfp32)
    y = full(builder, [32, 64], 2, Tfp32)
    size(dot(x, y)) == [16, 64]
end

dot_fma(x::Tensor, y::Tensor, accum::Tensor; allow_tf32 = true) = begin
    @assert is_block(x.type) && is_block(y.type) && is_block(accum.type) "x, y, and accum must be block tensors, got $x, $y, and $accum"
    @assert base_scalar_type(x.type) == base_scalar_type(y.type) == base_scalar_type(accum.type) "x, y, and accum must have the same type, got $x, $y, and $accum"
    @assert length(size(x)) == 2 && length(size(y)) == 2 && length(size(accum)) == 2 "x, y, and accum must be 2D tensors, got $x, $y, and $accum"
    @assert size(x, 2) == size(y, 1) "x and y must have compatible shapes, got $x and $y"
    @assert size(x, 1) == size(accum, 1) && size(y, 2) == size(accum, 2) "y and accum must have compatible shapes, got $y and $accum"
    @assert size(x, 1) >= 16 && size(x, 2) >= 16 && size(y, 1) >= 16 && size(y, 2) >= 16 "x and y must be at least 16x16, got $x and $y"
    @assert is_floating(x.type) && is_floating(y.type) && is_floating(accum.type) "TODO x, y, and accum must be floating point tensors, got $x, $y, and $accum"
    # accum_type = @match base_scalar_type(x.type), base_scalar_type(y.type) begin
    #     (Tfp32, _) => Tfp32
    #     (Tbf16, _) => Tfp32
    #     (_, Tfp16) => Tfp16
    #     (_, _) => Tfp32
    # end
    # accum = triton_zero(x.builder, accum_type)
    # M = size(x, 1)
    # N = size(y, 2)
    # accum_splat = CT.create_splat!(x.builder, accum.handle, [M, N])
    # ret_ty = BlockTritonType(accum_type, [M, N])
    Tensor(x.builder, CT.create_dot!(x.builder, x.handle, y.handle, accum.handle, allow_tf32), accum.type)
end