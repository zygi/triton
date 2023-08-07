# include("TritonCxxWrap.jl")
# const CT = CppTriton
##
include("global_implicit.jl")
using Test
using BFloat16s

macro wbc(expr)
    quote
        let ctx = CppTriton.MLIRContext()
            CppTriton.load_triton!(ctx)
            builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
            set_builder_ref(builder)
            res = $expr
            set_builder_ref(nothing)
            res
        end
    end
end


macro wc(expr)
    quote
        let ctx = CppTriton.MLIRContext()
            CppTriton.load_triton!(ctx)
            builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
            $expr
        end
    end
end

macro wc(criterion, expr)
    quote
        let ctx = CppTriton.MLIRContext()
            CppTriton.load_triton!(ctx)
            builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
            $expr
        end
    end
end

# test that the block doesn't throw the exception
macro wcok(expr)
    quote
        let ctx = CppTriton.MLIRContext()
            CppTriton.load_triton!(ctx)
            builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
            $expr
            true
        end
    end
end


# Tvoid
#     Tint1
#     Tint8
#     Tuint8
#     Tint16
#     Tuint16
#     Tint32
#     Tuint32
#     Tint64
#     Tuint64
#     Tfp8e5
#     Tfp8e4
#     Tfp8e4b15
#     Tfp16
#     Tbf16
#     Tfp32
#     Tfp64

using BFloat16s
using StaticArrays

abstract type TritonType end

const TritonTypeable = Union{
    Nothing, Bool,
    Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
    BFloat16, Float16, Float32, Float64,
#     Tfp8e5
#     Tfp8e4
#     Tfp8e4b15
    }

abstract type TritonSimpleType{T <: TritonTypeable} <: TritonType end
abstract type TritonPointerType{T <: TritonType} <: TritonType end

# TODO write note about how constructing tuples with Int32 etc can lead to very confusing and undebuggable errors
abstract type TritonBlockType{Size <: Tuple, T <: TritonType} <: TritonType end

# Arrgh I shouldn't need to do this, hopefully it won't bite me in the butt later
# Base.:==(::Type{<:}, ::Type{<:TritonType}) = true

const TrNothing = TritonSimpleType{Nothing}
const TrBool = TritonSimpleType{Bool}
const TrInt8 = TritonSimpleType{Int8}
const TrUInt8 = TritonSimpleType{UInt8}
const TrInt16 = TritonSimpleType{Int16}
const TrUInt16 = TritonSimpleType{UInt16}
const TrInt32 = TritonSimpleType{Int32}
const TrUInt32 = TritonSimpleType{UInt32}
const TrInt64 = TritonSimpleType{Int64}
const TrUInt64 = TritonSimpleType{UInt64}
const TrBFloat16 = TritonSimpleType{BFloat16}
const TrFloat16 = TritonSimpleType{Float16}
const TrFloat32 = TritonSimpleType{Float32}
const TrFloat64 = TritonSimpleType{Float64}

const TritonSimpleTypes = Union{TrNothing, TrBool, TrInt8, TrUInt8, TrInt16, TrUInt16, TrInt32, TrUInt32, TrInt64, TrUInt64, TrBFloat16, TrFloat16, TrFloat32, TrFloat64}
const TritonScalarTypes = Union{TritonSimpleTypes, TritonPointerType}

dimtuple_to_vec(::Type{Tuple{}}) = Int64[]
dimtuple_to_vec(::Type{X}) where {X <: Tuple} = collect(fieldtypes(X))

# Helper functions
Base.size(::Type{TritonBlockType{Size, T}}) where {Size <: Tuple, T <: TritonType} = dimtuple_to_vec(Size)
Base.size(::Type{T}) where {T <: TritonType} = Int64[]

numel(::Type{T}) where {T <: TritonType} = prod(size(T))
 
# Can I do this generically? Or more importantly, should I?
_build_tuple_type(::Val{T}) where T = Tuple{T}
_build_tuple_type(::Val{T}, ::Val{U}) where {T, U} = Tuple{T, U}
_build_tuple_type(::Val{T}, ::Val{U}, ::Val{V}) where {T, U, V} = Tuple{T, U, V}
_build_tuple_type(::Val{T}, ::Val{U}, ::Val{V}, ::Val{W}) where {T, U, V, W} = Tuple{T, U, V, W}

vec_to_dimtuple(xs) = _build_tuple_type(Val.(collect(Int64, xs))...)

TritonBlockType(::Type{T}, dims...) where {T <: TritonType} = TritonBlockType{vec_to_dimtuple(dims), T}
TritonBlockType(::Type{T}, dims) where {T <: TritonType} = TritonBlockType{vec_to_dimtuple(dims), T}


construct_ir_type(builder, ::Type{TrNothing}) = CppTriton.get_void_ty!(builder)
construct_ir_type(builder, ::Type{TrBool}) = CppTriton.get_int1_ty!(builder)
construct_ir_type(builder, ::Type{TrInt8}) = CppTriton.get_int8_ty!(builder)
construct_ir_type(builder, ::Type{TrUInt8}) = CppTriton.get_int8_ty!(builder)
construct_ir_type(builder, ::Type{TrInt16}) = CppTriton.get_int16_ty!(builder)
construct_ir_type(builder, ::Type{TrUInt16}) = CppTriton.get_int16_ty!(builder)
construct_ir_type(builder, ::Type{TrInt32}) = CppTriton.get_int32_ty!(builder)
construct_ir_type(builder, ::Type{TrUInt32}) = CppTriton.get_int32_ty!(builder)
construct_ir_type(builder, ::Type{TrInt64}) = CppTriton.get_int64_ty!(builder)
construct_ir_type(builder, ::Type{TrUInt64}) = CppTriton.get_int64_ty!(builder)
construct_ir_type(builder, ::Type{TrBFloat16}) = CppTriton.get_bf16_ty!(builder)
construct_ir_type(builder, ::Type{TrFloat16}) = CppTriton.get_half_ty!(builder)
construct_ir_type(builder, ::Type{TrFloat32}) = CppTriton.get_float_ty!(builder)
construct_ir_type(builder, ::Type{TrFloat64}) = CppTriton.get_double_ty!(builder)

construct_ir_type(builder, ::Type{TritonPointerType{T}}) where {T <: TritonType} = CppTriton.get_ptr_ty!(builder, construct_ir_type(builder, T), 1)
construct_ir_type(builder, ::Type{TritonBlockType{Size, T}}) where {Size <: Tuple, T <: TritonType} = CppTriton.get_block_ty!(builder, construct_ir_type(builder, T), dimtuple_to_vec(Size))

construct_ir_type(::Type{T}) where {T <: TritonType} = construct_ir_type(get_builder_ref(), T)
@test @wbc begin construct_ir_type(TrFloat32); true end
@test @wbc begin construct_ir_type(TritonPointerType{TrUInt8}); true end
@test @wbc begin construct_ir_type(TritonBlockType{Tuple{1, 2}, TrUInt8}); true end
@test @wbc begin construct_ir_type(TritonBlockType{Tuple{}, TrUInt8}); true end



# _bi stands for block included, that is, it returns true for scalar floating types and block floating types
is_floating_bi(::Type{TritonSimpleType{T}}) where {T <: TritonTypeable} = T <: Float16 || T <: Float32 || T <: Float64 || T <: BFloat16
is_floating_bi(::Type{TritonBlockType{Size, T}}) where {Size <: Tuple, T <: TritonType} = is_floating_bi(T) 
is_floating_bi(::Type{T}) where {T <: TritonType} = false

# pointers are NOT integers
is_integer_bi(::Type{TritonSimpleType{T}}) where {T <: TritonTypeable} = T <: Bool || T <: Int8 || T <: UInt8 || T <: Int16 || T <: UInt16 || T <: Int32 || T <: UInt32 || T <: Int64 || T <: UInt64
is_integer_bi(::Type{TritonBlockType{Size, T}}) where {Size <: Tuple, T <: TritonType} = is_integer_bi(T)
is_integer_bi(::Type{T}) where {T <: TritonType} = false

is_pointer_bi(::Type{T}) where {T <: TritonPointerType} = true
is_pointer_bi(::Type{TritonBlockType{Size, T}}) where {Size <: Tuple, T <: TritonType} = is_pointer_bi(T)
is_pointer_bi(::Type{T}) where {T <: TritonType} = false

# pointers ARE scalars
is_scalar(::Type{T}) where {T <: TritonSimpleType} = true
is_scalar(::Type{T}) where {T <: TritonPointerType} = true
is_scalar(::Type{T}) where {T <: TritonType} = false

is_block(::Type{T}) where {T <: TritonBlockType} = true
is_block(::Type{T}) where {T <: TritonType} = false

scalar_type_of(::Type{TritonBlockType{Size, T}}) where {Size <: Tuple, T <: TritonType} = T
scalar_type_of(::Type{T}) where {T <: TritonType} = T

points_to(::Type{TritonPointerType{T}}) where {T <: TritonType} = T
points_to(::Type{TritonBlockType{Size, TritonPointerType{T}}}) where {Size <: Tuple, T <: TritonType} = T

# replace_block_scalar(::Type{TritonBlockType{Size, T}}, ::Type{U}) where {Size <: Tuple, T <: TritonType, U <: TritonScalarTypes} = TritonBlockType{Size, U}
# @test replace_block_scalar(TritonBlockType{Tuple{2, 3}, TrInt32}, TrFloat32) == TritonBlockType{Tuple{2, 3}, TrFloat32}

change_scalar_type(::Type{TritonBlockType{Size, T}}, ::Type{U}) where {Size <: Tuple, T <: TritonType, U <: TritonScalarTypes} = TritonBlockType{Size, U}
change_scalar_type(::Type{T}, ::Type{U}) where {T <: TritonType, U <: TritonScalarTypes} = U

primitive_bandwidth(::Type{T}) where {T <: TritonType} = primitive_bandwidth(scalar_type_of(x))
primitive_bandwidth(::Type{TrBool}) = 1
primitive_bandwidth(::Type{TrInt8}) = 8
primitive_bandwidth(::Type{TrUInt8}) = 8
primitive_bandwidth(::Type{TrInt16}) = 16
primitive_bandwidth(::Type{TrUInt16}) = 16
primitive_bandwidth(::Type{TrInt32}) = 32
primitive_bandwidth(::Type{TrUInt32}) = 32
primitive_bandwidth(::Type{TrInt64}) = 64
primitive_bandwidth(::Type{TrUInt64}) = 64
primitive_bandwidth(::Type{TrBFloat16}) = 16
primitive_bandwidth(::Type{TrFloat16}) = 16
primitive_bandwidth(::Type{TrFloat32}) = 32
primitive_bandwidth(::Type{TrFloat64}) = 64
primitive_bandwidth(::Type{T}) where {T <: TritonPointerType} = 64


fp_mantissa_width(x::Type{TrBFloat16}) = 10
fp_mantissa_width(x::Type{TrFloat16}) = 7
fp_mantissa_width(x::Type{TrFloat32}) = 23
fp_mantissa_width(x::Type{TrFloat64}) = 53

is_signed_bi(::Type{TritonBlockType{S, T}}) where {S, T} = is_signed_bi(scalar_type_of(T))
is_signed_bi(::Type{TrBool}) = true
is_signed_bi(::Type{TrInt8}) = true
is_signed_bi(::Type{TrInt16}) = true
is_signed_bi(::Type{TrInt32}) = true
is_signed_bi(::Type{TrInt64}) = true
is_signed_bi(::Type{T}) where {T <: TritonType} = false

is_standard_floating(::Type{TritonBlockType{S, T}}) where {S, T} = is_standard_floating(scalar_type_of(T))
is_standard_floating(::Type{TrBFloat16}) = true
is_standard_floating(::Type{TrFloat16}) = true
is_standard_floating(::Type{TrFloat32}) = true
is_standard_floating(::Type{TrFloat64}) = true
is_standard_floating(::Type{T}) where {T <: TritonType} = false


const TypeMLIRToJuliaNameLookup = let
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    Dict([CT.repr(CT.CxxRef(construct_ir_type(builder, x))) => x for x in Base.uniontypes(TritonSimpleTypes)])
end

_parse_type_from_ptrscalar(x::AbstractString) = begin
    if startswith(x, "!tt.ptr<")
        inner = x[9:end-1]
        TritonPointerType{TypeMLIRToJuliaNameLookup[inner]}
    else
        TypeMLIRToJuliaNameLookup[x]
    end
end
@test _parse_type_from_ptrscalar("!tt.ptr<i32>") == TritonPointerType{TrUInt32}

_parse_type_from_repr(x::AbstractString) = begin
    if startswith(x, "tensor<")
        notensor = x[8:end-1]
        parts = split(String(notensor), 'x')
        if length(parts) == 1
            # a 0-dim tensor
            TritonBlockType{Tuple{}, _parse_type_from_ptrscalar(parts[1]), Int64[]}
        else
            dims = parse.(Int64, parts[1:end-1])
            dims_type = vec_to_dimtuple(dims)
            TritonBlockType{dims_type, _parse_type_from_ptrscalar(parts[end])}
        end
    else
        _parse_type_from_ptrscalar(x)
    end
end
@test _parse_type_from_repr("tensor<1x2x3x!tt.ptr<i32>>") == TritonBlockType{Tuple{1, 2, 3}, TritonPointerType{TrUInt32}}


# Tuple{1, 2} == vec_to_dimtuple([1, 2])

# Now, SymTensors!
struct Tensor{T <: TritonType}
    builder
    handle
    Tensor{T}(builder, handle) where {T<:TritonType} = begin
        # hacky way to double check types since it's annoying to extract a julia-rep of a type from a handle
        t1 = CT.get_type(handle)
        t2 = construct_ir_type(builder, T)
        t1_repr = CT.repr(CT.CxxRef(t1))
        t2_repr = CT.repr(CT.CxxRef(t2))
        @assert t1_repr == t2_repr "Type mismatch when wrapping Tensor: expected $t1_repr, got $t2_repr"
        
        new{T}(builder, handle)
    end
end
Tensor(builder, handle, ::Type{T}) where {T<:TritonType} = Tensor{T}(builder, handle)
Tensor(handle, ::Type{T}) where {T<:TritonType} = Tensor{T}(get_builder_ref(), handle)
Tensor(x::T) where {T<:Tensor} = x

trtype(t::Tensor{T}) where {T} = T

Base.show(io::IO, x::Tensor) = begin
    hd = CT.repr(CT.CxxRef(x.handle))
    print(io, "Tensor($hd)")
end

Tensor(builder, b::Bool) = Tensor(builder, CppTriton.get_int1!(builder, b), TrBool)
Tensor(builder, x::Int64) = Tensor(builder, CppTriton.get_int64!(builder, x), TrInt64)
Tensor(builder, x::UInt64) = Tensor(builder, CppTriton.get_int64!(builder, reinterpret(Int64, x)), TrUInt64)
Tensor(builder, x::Int32) = Tensor(builder, CppTriton.get_int32!(builder, x), TrInt32)
Tensor(builder, x::UInt32) = Tensor(builder, CppTriton.get_int32!(builder, reinterpret(Int32, x)), TrUInt32)

@test @wcok Tensor(builder, Int64(1))
@test @wcok Tensor(builder, Int64(-2^63))
@test @wcok Tensor(builder, Int64(2^63-1))
@test @wcok Tensor(builder, typemax(UInt64))

Tensor(builder, x::Float32) = Tensor(builder, CppTriton.get_fp32!(builder, x), TrFloat32)
Tensor(builder, x::Float64) = Tensor(builder, CppTriton.get_fp64!(builder, x), TrFloat64)
Tensor(builder, x::Float16) = Tensor(builder, CppTriton.get_fp16!(builder, Float32(x)), TrFloat16)
Tensor(builder, x::BFloat16) = Tensor(builder, CppTriton.get_bf16!(builder, Float32(x)), TyBFloat16)

IntoTensor = Union{Bool, Int64, UInt64, Int32, UInt32, Float32, Float64, Float16, Tensor}
Tensor(x::IntoTensor) = begin Tensor(get_builder_ref(), x) end
@test @wcok begin
    with_scoped(builder) do 
        Tensor(5.0)
    end
end

Base.size(t::Tensor{T}) where T = Base.size(T)
Base.size(t::Tensor{T}, idx) where T = Base.size(T)[idx]
numel(t::Tensor{T}) where T = numel(T)

glb, glb_ctx = let ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    builder, ctx
end

cast(input::Tensor{TritonBlockType{S, T}}, to::Type{U}) where {S, T, U<:TritonScalarTypes} = cast(input, TritonBlockType{S, U})
cast(input::Tensor{T}, to::Type{T}) where {T <: TritonType} = input
cast(input::Tensor{T}, to::Type{U}) where {T, U<:TritonType} = begin
    builder = get_builder_ref()
    T_scalar = scalar_type_of(T)
    U_scalar = scalar_type_of(U)
    if (T_scalar == TrBFloat16 || T_scalar == TrFloat16) && U_scalar != TrFloat32
        return Tensor(CT.create_fp_to_fp!(builder, input.handle, construct_ir_type(input.builder, U)), U)
    end

    if is_floating_bi(T_scalar) && is_floating_bi(U_scalar)
        if primitive_bandwidth(T_scalar) > primitive_bandwidth(U_scalar)
            return Tensor(CT.create_fp_trunc!(builder, input.handle, construct_ir_type(input.builder, U)), U)
        else
            return Tensor(CT.create_fp_ext!(builder, input.handle, construct_ir_type(input.builder, U)), U)
        end
    end

    if is_integer_bi(T_scalar) && is_integer_bi(U_scalar) && (
        (primitive_bandwidth(T_scalar) != primitive_bandwidth(U_scalar)) ||
        (is_signed_bi(T_scalar) != is_signed_bi(U_scalar))
    )
        # is this correct? looks suspicious
        needs_sign_extension = is_signed_bi(T_scalar) && !is_signed_bi(U_scalar)
        if U_scalar == TrBool
            return input != Tensor(CT.get_null_value!(builder, construct_ir_type(U)), U)
        else
            return Tensor(CT.create_int_cast!(builder, input.handle, construct_ir_type(input.builder, U), needs_sign_extension), U)
        end
    end

    if is_standard_floating(T_scalar) && is_integer_bi(U_scalar)
        if U_scalar == TrBool
            return input != Tensor(CT.get_null_value!(builder, construct_ir_type(U)), U)
        elseif is_signed_bi(U_scalar)
            return Tensor(CT.create_fp_to_si!(builder, input.handle, construct_ir_type(U)), U)
        else
            return Tensor(CT.create_fp_to_ui!(builder, input.handle, construct_ir_type(U)), U)
        end
    end

    if is_integer_bi(T_scalar) && is_standard_floating(U_scalar)
        if T_scalar == TrBool || !is_signed_bi(T_scalar)
            return Tensor(CT.create_ui_to_fp!(builder, input.handle, construct_ir_type(U)), U)
        else
            return Tensor(CT.create_si_to_fp!(builder, input.handle, construct_ir_type(U)), U)
        end
    end

    if is_pointer_bi(T_scalar) && is_integer_bi(U_scalar)
        @assert primitive_bandwidth(U_scalar) == 64 "can onnly cast pointers to 64-bit integers"
        return Tensor(CT.create_ptr_to_int!(builder, input.handle, construct_ir_type(U)), U)
    end

    if is_integer_bi(T_scalar) && is_pointer_bi(U_scalar)
        return Tensor(CT.create_int_to_ptr!(builder, input.handle, construct_ir_type(U)), U)
    end

    if is_pointer_bi(T_scalar) && is_pointer_bi(U_scalar)
        return Tensor(CT.create_bitcast!(builder, input.handle, construct_ir_type(U)), U)
    end

    throw(ArgumentError("unsupported cast from $T to $U"))
end
@test @wbc begin cast(Tensor(5.0), TritonBlockType{Tuple{}, TrFloat64}); true end
@test @wbc begin cast(Tensor(5), TrBool); true end

function arange(start::Integer, endd::Integer)
    start = Int32(start)
    endd = Int32(endd)
    shape = [endd - start,]
    ret_ty = TritonBlockType{vec_to_dimtuple(shape), TrInt32}
    Tensor(CT.create_make_range!(get_builder_ref(), start, endd), ret_ty)
end
@test @wbc begin arange(0, 5); true end

function full(dims::Tuple{Vararg{Int64}}, value::T) where {T <: IntoTensor}
    tens = Tensor(value)
    @assert numel(tens) == 1 "only scalar values are supported"
    Tensor(get_builder_ref(), CT.create_splat!(get_builder_ref(), tens.handle, collect(dims)), TritonBlockType{vec_to_dimtuple(dims), scalar_type_of(trtype(tens))})
end
@test @wbc begin full((2, 3), 5); true end

# Tuple{1, 2} == Tuple{1, 2}

function triton_broadcast(lhs::Tensor{TritonBlockType{S1, T1}}, rhs::Tensor{TritonBlockType{S2, T2}}) where {S1, S2, T1, T2}
    builder = get_builder_ref()
    if S1 == S2 return lhs, rhs end
    # do it dynamically
    lhs_dims = dimtuple_to_vec(S1)
    rhs_dims = dimtuple_to_vec(S2)
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
        Tensor(CT.create_broadcast!(builder, lhs.handle, collect(target_shape)), TritonBlockType{vec_to_dimtuple(target_shape), T1})
    else
        lhs
    end

    rhs = if rhs_dims != target_shape
        Tensor(CT.create_broadcast!(builder, rhs.handle, collect(target_shape)), TritonBlockType{vec_to_dimtuple(target_shape), T2})
    else
        rhs
    end

    return lhs, rhs
end
function triton_broadcast(lhs::Tensor{TritonBlockType{S1, T1}}, rhs::Tensor{T2}) where {S1, T1, T2}
    lhs, Tensor(CT.create_splat!(get_builder_ref(), rhs.handle, dimtuple_to_vec(S1)), TritonBlockType{S1, T2})
end
function triton_broadcast(lhs::Tensor{T1}, rhs::Tensor{TritonBlockType{S2, T2}}) where {S2, T1, T2}
    Tensor(CT.create_splat!(get_builder_ref(), lhs.handle, dimtuple_to_vec(S2)), TritonBlockType{S2, T1}), rhs
end
function triton_broadcast(lhs::Tensor{T1}, rhs::Tensor{T2}) where {T1 <: TritonScalarTypes, T2 <: TritonScalarTypes}
    lhs, rhs
end

triton_broadcast(a, b, c) = begin
    a, b = triton_broadcast(a, b)
    b, c = triton_broadcast(b, c)
    a, b = triton_broadcast(a, b)
    a, b, c
end

@test @wbc begin
    t1 = Tensor(1.0)
    t2 = full((10, 20), 5)    
    t1, t2 = triton_broadcast(t1, t2)
    size(t1) == size(t2)
end

program_id(axis) = Tensor(CT.create_get_program_id!(get_builder_ref(), axis-1), TrInt32)
num_programs(axis) = Tensor(CT.create_get_num_programs!(get_builder_ref(), axis-1), TrInt32)
@test @wbc begin program_id(1); true end
@test @wbc begin num_programs(1); true end


using Expronicon
using MLStyle

_split_arg(e) = @match e begin
    Expr(:(::), name, Tsym) => (name, Tsym) 
    x => begin x.head; throw("Binary op must take two tensors") end
end
macro binary_op_implicit_casting(fn)
    jlfn = JLFunction(fn)
    @assert _split_arg(jlfn.args[1])[2] == :Tensor && _split_arg(jlfn.args[2])[2] == :Tensor "Binary op must take two tensors"
    @assert length(jlfn.args) == 2 "Binary op must take two tensors"

    arg_names = map(x -> _split_arg(x)[1], jlfn.args)

    orig_args = copy(jlfn.args)
    jlfn.args[1] = Expr(:(::), arg_names[1], :IntoTensor)
    jlfn.body = quote
        $(jlfn.name)(Tensor($(arg_names[1])), $(arg_names[2])) 
    end
    left_fn = codegen_ast(jlfn)

    jlfn.args = orig_args
    jlfn.args[2] = Expr(:(::), _split_arg(jlfn.args[2])[1], :IntoTensor)
    jlfn.body = quote
        $(jlfn.name)($(arg_names[1]), Tensor($(arg_names[2]))) 
    end
    right_fn = codegen_ast(jlfn)


    quote
        $(esc(fn))
        $(esc(left_fn))
        $(esc(right_fn))
    end
end

@binary_op_implicit_casting Base.:+(lhs::Tensor, rhs::Tensor) = begin
    lhs, rhs = triton_broadcast(lhs, rhs)
    # @assert types_shapes_match_uptopointer(lhs, rhs) "Types and shapes must match, got x: $lhs, y: $rhs"

    # offset + ptr
    # ptr + offset
    if is_pointer_bi(trtype(rhs)) && !is_pointer_bi(trtype(lhs))
        lhs, rhs = rhs, lhs
    end
    if is_pointer_bi(trtype(lhs))
        return Tensor(CT.create_addptr!(get_builder_ref(), lhs.handle, rhs.handle), trtype(lhs))
    end

    # float + float
    if is_floating_bi(trtype(lhs)) && is_floating_bi(trtype(rhs))
        return Tensor(CT.create_fadd!(get_builder_ref(), lhs.handle, rhs.handle), trtype(lhs))
    end

    if is_integer_bi(trtype(lhs)) && is_integer_bi(trtype(rhs))
        return Tensor(CT.create_add!(get_builder_ref(), lhs.handle, rhs.handle), trtype(lhs))
    end

    throw("Can't add $lhs and $rhs")
end

@test @wbc begin Tensor(2.0) + Tensor(5.0); true end
@test @wbc begin Tensor(2.0) + 1.0; true end
@test @wbc begin full((5,), 3) + Tensor(1); true end

Base.zero(::Type{T}) where {T <: TritonType} = Tensor(CT.get_null_value!(get_builder_ref(), construct_ir_type(get_builder_ref(), T)), T)
Base.zero(x::Tensor{T}) where {T <: TritonType} = zero(T)
@test @wbc begin zero(TrInt32); true end

Base.one(::Type{T}) where {U, T <: TritonSimpleType{U}} = Tensor(one(U))
Base.one(x::Tensor{T}) where {T <: TritonSimpleType} = one(T)
@test @wbc begin one(TrFloat32); true end


@binary_op_implicit_casting Base.:-(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    if is_pointer_bi(trtype(x))
        return Tensor(x.builder, CT.create_addptr!(x.builder, x.handle, (-y).handle), trtype(x))
    end
    if is_floating_bi(trtype(x)) && is_floating_bi(trtype(y))
        return Tensor(x.builder, CT.create_fsub!(x.builder, x.handle, y.handle), trtype(x))
    end
    if is_integer_bi(trtype(x)) && is_integer_bi(trtype(y))
        return Tensor(x.builder, CT.create_sub!(x.builder, x.handle, y.handle), trtype(x))
    end
    throw("Can't subtract $x and $y")
end
Base.:-(x::Tensor) = begin
    is_pointer_bi(trtype(x)) && throw("Can't negate a pointer")
    zero(trtype(x)) - x
end 
@test @wbc begin -Tensor(1.0); true end
@test @wbc begin cast(Tensor(1), TritonPointerType{TrInt64}) - Tensor(Int32(2)); true end

@binary_op_implicit_casting Base.:*(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    if is_floating_bi(trtype(x)) && is_floating_bi(trtype(y))
        return Tensor(x.builder, CT.create_fmul!(x.builder, x.handle, y.handle), trtype(x))
    end
    if is_integer_bi(trtype(x)) && is_integer_bi(trtype(y))
        return Tensor(x.builder, CT.create_mul!(x.builder, x.handle, y.handle), trtype(x))
    end
    throw("Can't multiply $x and $y")
end
@test @wbc begin Tensor(1.0) * Tensor(2.0); true end

@binary_op_implicit_casting Base.:/(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    if is_floating_bi(trtype(x)) && is_integer_bi(trtype(y))
        y = cast(y, trtype(x))
    elseif is_integer_bi(trtype(x)) && is_floating_bi(trtype(y))
        x = cast(x, trtype(y))
    elseif is_floating_bi(trtype(x)) && is_floating_bi(trtype(y))
        if fp_mantissa_width(trtype(x)) > fp_mantissa_width(trtype(y))
            y = cast(y, trtype(x))
        else
            x = cast(x, trtype(y))
        end
    else
        # TODO think about int/int
        throw("Can't divide $x and $y")
    end
    return Tensor(x.builder, CT.create_fdiv!(x.builder, x.handle, y.handle), trtype(x))
end
@test @wbc (Tensor(1.0) / Tensor(2.0f0)) |> trtype == TrFloat64

@binary_op_implicit_casting Base.div(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    if is_integer_bi(trtype(x)) && is_integer_bi(trtype(y))
       if is_signed_bi(trtype(x))
            return Tensor(x.builder, CT.create_sdiv!(x.builder, x.handle, y.handle), trtype(x))
        else
            return Tensor(x.builder, CT.create_udiv!(x.builder, x.handle, y.handle), trtype(x))
        end
    end
    throw("Can't divide $x and $y")    
end
@test @wbc begin Tensor(1) ÷ Tensor(2); true end

cdiv(x, y) = (x + y - one(x)) ÷ y
@test @wbc begin cdiv(Tensor(5), Tensor(2)); true end

##

@binary_op_implicit_casting Base.rem(x::Tensor, y::Tensor) = begin
    x, y = triton_broadcast(x, y)
    if is_integer_bi(trtype(x)) && is_integer_bi(trtype(y))
        @assert is_signed_bi(trtype(x)) == is_signed_bi(trtype(y)) "Types must be both signed or both unsigned"
        if is_signed_bi(trtype(x))
            return Tensor(CT.create_srem!(x.builder, x.handle, y.handle), trtype(x))
        else
            return Tensor(CT.create_urem!(x.builder, x.handle, y.handle), trtype(x))
        end
    end
    # TODO think about float % float
    throw("Can't divide $x and $y")    
end
@test @wbc begin Tensor(5) % Tensor(2) ; true end

base_eq(x::Tensor, y::Tensor) = x == y
base_neq(x::Tensor, y::Tensor) = x != y


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
        @binary_op_implicit_casting Base.$op_name(x::Tensor, y::Tensor) = begin
            x, y = triton_broadcast(x, y)
            return_ty = change_scalar_type(trtype(x), TrBool)

            if is_floating_bi(trtype(x)) && is_floating_bi(trtype(y))
                return Tensor(CT.$float_op(x.builder, x.handle, y.handle), return_ty)
            end
            if is_integer_bi(trtype(x)) && is_integer_bi(trtype(y))
                if is_signed_bi(trtype(x))
                    return Tensor(CT.$signed_op(x.builder, x.handle, y.handle), return_ty)
                else
                    return Tensor(CT.$unsigned_op(x.builder, x.handle, y.handle), return_ty)
                end
            end
            if is_pointer_bi(trtype(x)) && is_pointer_bi(trtype(y))
                # TODO decide once and for all if pointers are signed or unsigned 
                return Tensor(CT.$signed_op(x.builder, x.handle, y.handle), return_ty)
            end
            throw("Can't compare $x and $y")
        end

        @test @wbc ($op_name(Tensor(1.0), Tensor(2.0))) |> trtype == TrBool
        @test @wbc ($op_name(Tensor(Int32(1)), Tensor(Int32(1)))) |> trtype == TrBool
        @test @wbc ($op_name(Tensor(UInt32(1)), Tensor(UInt32(1)))) |> trtype == TrBool
        @test @wbc ($op_name(full((5,), 5.0), full((5,), 4.0))) |> trtype == TritonBlockType{Tuple{5}, TrBool}
    end)
end

for (op_name, create_op) in [(:&, :create_and!), (:|, :create_or!), (:^, :create_xor!)]
    eval(quote
        @binary_op_implicit_casting Base.$op_name(x::Tensor, y::Tensor) = begin
            x, y = triton_broadcast(x, y)
            @assert is_integer_bi(trtype(x)) && is_integer_bi(trtype(y)) "Both operands must be integers, got x: $(trtype(x)) and y: $(trtype(y))"
            Tensor(CT.$create_op(x.builder, x.handle, y.handle), trtype(x))
        end

        @test @wbc ($op_name(Tensor(Int32(1)), Tensor(Int32(1)))) |> trtype == TrInt32
    end)
end

expanddims(x::Tensor{TritonBlockType{S, T}}, axis::Int) where {S, T} = begin
    # @assert is_block(trtype(x))
    dims = dimtuple_to_vec(S)
    @assert axis >= 1 && axis <= length(dims)+1 "Axis must be in range [1, length(dims)+1]"
    new_shape = similar(dims, length(dims) + 1)
    new_shape[1:(axis-1)] .= dims[1:axis-1]
    new_shape[axis] = 1
    new_shape[axis + 1:end] .= dims[axis:end]
    new_type = TritonBlockType{vec_to_dimtuple(new_shape), T}
    Tensor(CT.create_expand_dims!(x.builder, x.handle, axis-1), new_type)
end
@test @wbc size(expanddims(full((2, 3), 1.0), 2)) == [2, 1, 3]


function broadcast_impl_shape(x::Tensor{TritonBlockType{S, T}}, shape) where {S, T}
    shape = collect(Int64, shape)
    src_shape = dimtuple_to_vec(S)
    @assert length(src_shape) == length(shape) "Shapes must have the same length, got $src_shape and $shape"
    new_shape = similar(src_shape, length(src_shape))
    for i in eachindex(src_shape)
        if src_shape[i] == 1
            new_shape[i] = shape[i]
        else
            @assert src_shape[i] == shape[i] "Shapes must be compatible, got $src_shape and $shape"
            new_shape[i] = src_shape[i]
        end
    end
    Tensor(CT.create_broadcast!(x.builder, x.handle, new_shape), TritonBlockType{vec_to_dimtuple(new_shape), T})
end
function broadcast_impl_shape(x::Tensor{T}, shape) where T <: TritonScalarTypes
    Tensor(CT.create_splat!(x.builder, x.handle, collect(Int64, shape)), TritonBlockType{vec_to_dimtuple(shape), T})
end
broadcast_impl_shape(x::IntoTensor, shape) = broadcast_impl_shape(Tensor(x), shape)

@test @wbc size(broadcast_impl_shape(Tensor(1.0), [2, 3])) == [2, 3]
@test @wbc begin
    res = broadcast_impl_shape(full((2, 1, 3), 1.0f0), [2, 5, 3])
    size(res) == [2, 5, 3] && scalar_type_of(trtype(res)) == TrFloat32
end


Base.zeros(::Type{T}, dims) where {T <: TritonType} = broadcast_impl_shape(zero(T), dims)
@test @wbc begin zeros(TrFloat32, [2,]); true end


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

shapes_match(x::Tensor, y::Tensor) = size(x) == size(y)

points_to_type(ptr::Tensor{TritonBlockType{S, TritonPointerType{T}}}, dst::Tensor{TritonBlockType{S, T}}) where {S, T} = true
points_to_type(ptr::Tensor{TritonPointerType{T}}, dst::Tensor{T}) where {T <: TritonType} = true
points_to_type(ptr::Tensor, dst::Tensor) = false


_load_legacy(ptr::Tensor, mask::Union{Tensor, Nothing}, other::Union{Tensor, Nothing}, cache, eviction, is_volatile) = begin
    @assert is_pointer_bi(trtype(ptr)) "ptr must be a pointer or a pointer block"
    @assert primitive_bandwidth(scalar_type_of(trtype(ptr))) != 1 "TODO ptr can't point to bools"
    @assert isnothing(mask) == isnothing(other) "mask and other must be either both nothing or both tensors"

    if !isnothing(mask)
        ptr, mask, other = triton_broadcast(ptr, mask, other)
    end

    @assert isnothing(mask) || (scalar_type_of(trtype(mask)) == TrBool && shapes_match(ptr, mask)) "mask must be a boolean tensor with the same shape as ptr, got $mask and $ptr"
    @assert isnothing(other) || points_to_type(ptr, other) "other must have the same type as ptr, got $other and $ptr"

    result_ty = change_scalar_type(trtype(ptr), points_to(trtype(ptr)))

    if isnothing(mask)
        Tensor(CT.create_load!(ptr.builder, ptr.handle, cache, eviction, is_volatile), result_ty)
    else
        Tensor(CT.create_masked_load!(ptr.builder, ptr.handle, mask.handle, other.handle, cache, eviction, is_volatile), result_ty)
    end
end

@test @wbc begin
    ptr = cast(Tensor(1), TritonPointerType{TrFloat32})
    mask = Tensor(true)
    other = Tensor(2.0f0)
    _load_legacy(ptr, mask, other, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    _load_legacy(ptr, nothing, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    true
end
@test @wbc begin
    ptr = cast(full((2, 3), 5), TritonPointerType{TrInt64})
    mask = full((2, 3), true)
    other = full((2, 3), 2)
    _load_legacy(ptr, nothing, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    _load_legacy(ptr, mask, other, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    true
end


load(ptr::IntoTensor; mask::Union{IntoTensor, Nothing}=nothing, other::Union{IntoTensor, Nothing}=nothing, cache="", eviction="", is_volatile=false) = begin
    ptr = Tensor(ptr)
    if !isnothing(mask); mask = Tensor(mask) end
    if !isnothing(other); other = Tensor(other) end
    _load_legacy(ptr, mask, other, _string_to_load_cache_modifier(cache), _string_to_eviction_policy(eviction), is_volatile)
end



_store_legacy(ptr::Tensor, val::Tensor, mask::Union{Tensor, Nothing}, cache, eviction) = begin
    @assert is_pointer_bi(trtype(ptr)) "ptr must be a pointer or a pointer block"
    @assert primitive_bandwidth(scalar_type_of(trtype(ptr))) != 1 "TODO ptr can't point to bools"

    if !isnothing(mask) && is_block(trtype(mask))
        ptr, val, mask = triton_broadcast(ptr, val, mask)
    else
        ptr, val = triton_broadcast(ptr, val)
    end
    
    @assert points_to_type(ptr, val) "ptr must be ptr<T> where val is <T>, got ptr: $ptr and val: $val"
    @assert isnothing(mask) || (scalar_type_of(trtype(mask)) == TrBool) "mask must be a boolean tensor, got $mask"
    @assert isnothing(mask) || shapes_match(ptr, mask) "mask must be a boolean tensor with the same shape as ptr, got $mask and $ptr"
    
    if isnothing(mask)
        CT.create_store!(ptr.builder, ptr.handle, val.handle, cache, eviction)
    else
        CT.create_masked_store!(ptr.builder, ptr.handle, val.handle, mask.handle, cache, eviction)
    end
end

@test @wbc begin
    ptr = cast(Tensor(1), TritonPointerType{TrFloat32})
    val = Tensor(2.0f0)
    mask = Tensor(true)
    _store_legacy(ptr, val, mask, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    _store_legacy(ptr, val, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    true
end
@test @wbc begin
    ptr = cast(full((2, 3), 5), TritonPointerType{TrInt64})
    val = full((2, 3), 2)
    mask = full((2, 3), true)
    _store_legacy(ptr, val, mask, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    _store_legacy(ptr, val, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    true
end

store(ptr::IntoTensor, val::IntoTensor; mask::Union{IntoTensor, Nothing}=nothing, cache="", eviction="") = begin
    _store_legacy(Tensor(ptr), Tensor(val), isnothing(mask) ? nothing : Tensor(mask), _string_to_store_cache_modifier(cache), _string_to_eviction_policy(eviction))
end



triton_return() = CT.ret!(get_builder_ref(), CT.CxxRef{CT.Value}[])
@test @wbc begin triton_return(); true end

device_print(prefix, args...) = begin
    handles = collect(map(x -> Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(x.handle)), args))
    CT.create_print!(get_builder_ref(), prefix, handles)
end
@test @wbc begin device_print("hello", full((2, 3), 5), full((2, 3), 5)); true end


triton_yield(vs...) = begin
    if isempty(vs)
        CT.create_yield_op!(get_builder_ref(), CT.CxxRef{CT.Value}[])
    else
        CT.create_yield_op!(get_builder_ref(), collect(map(x -> Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(Tensor(x).handle)), vs)))
    end
end
@test @wbc begin triton_yield(full((2, 3), 5), full((2, 3), 5), 2.0); true end

types_shapes_match(x, y) = shapes_match(x, y) && scalar_type_of(trtype(x)) == scalar_type_of(trtype(y))

triton_where(cond::IntoTensor, x::IntoTensor, y::IntoTensor) = begin
    cond = Tensor(cond); x = Tensor(x); y = Tensor(y)
    cond = cast(cond, TrBool)
    if is_block(trtype(cond))
        cond, x, y = triton_broadcast(cond, x, y)
    else
        x, y = triton_broadcast(x, y)
    end
    @assert types_shapes_match(x, y) "x and y must have the same type and shape, got $x and $y"
    Tensor(CT.create_select!(cond.builder, cond.handle, x.handle, y.handle), trtype(x))
end
@test @wbc begin
    cond = full((2, 3), true)
    x = full((2, 3), 5)
    y = full((2, 3), 2)
    triton_where(cond, x, y)
    true
end
@test @wbc begin
    cond = Tensor(true)
    x = full((2, 3), 5)
    y = full((2, 3), 2)
    triton_where(cond, x, y)
    true
end


@binary_op_implicit_casting Base.min(x::Tensor, y::Tensor) = triton_where(x < y, x, y)
@binary_op_implicit_casting Base.max(x::Tensor, y::Tensor) = triton_where(x > y, x, y)
@test @wbc begin
    x = full((2, 3), 5)
    y = full((2, 3), 2)
    min(x, y)
    max(x, y)
    true
end


_dot_accum_type(::Type{TrFloat32}, ::Type, output_type) = TrFloat32
_dot_accum_type(::Type{TrFloat16}, ::Type, output_type) = TrFloat32
_dot_accum_type(::Type, ::Type, output_type) = output_type

dot(x::Tensor, y::Tensor; allow_tf32 = true, output_ty::Type{T}=TrFloat32) where {T <: TritonSimpleTypes} = begin
    @assert is_block(trtype(x)) && is_block(trtype(y)) "x and y must be block tensors, got $x and $y"
    @assert scalar_type_of(trtype(x)) == scalar_type_of(trtype(y)) "x and y must have the same type, got $x and $y"
    @assert length(size(x)) == 2 && length(size(y)) == 2 "x and y must be 2D tensors, got $x and $y"
    @assert size(x, 2) == size(y, 1) "x and y must have compatible shapes, got $x and $y"
    @assert size(x, 1) >= 16 && size(x, 2) >= 16 && size(y, 1) >= 16 && size(y, 2) >= 16 "x and y must be at least 16x16, got $x and $y"
    @assert is_floating_bi(trtype(x)) && is_floating_bi(trtype(y)) "TODO x and y must be floating point tensors, got $x and $y"
    accum_type = _dot_accum_type(trtype(x), trtype(y), output_ty)
    accum = zero(accum_type)
    M = size(x, 1)
    N = size(y, 2)
    accum_splat = CT.create_splat!(x.builder, accum.handle, [M, N])
    ret_ty = TritonBlockType{Tuple{M, N}, accum_type}
    Tensor(x.builder, CT.create_dot!(x.builder, x.handle, y.handle, accum_splat, allow_tf32), ret_ty)
end
@test @wbc begin
    x = full((16, 32), 5.0)
    y = full((32, 64), 2.0)
    size(dot(x, y)) == [16, 64]
end

# dot_fma(x::Tensor, y::Tensor, accum::Tensor; allow_tf32 = true) = begin
#     @assert is_block(trtype(x)) && is_block(trtype(y)) && is_block(accum.type) "x, y, and accum must be block tensors, got $x, $y, and $accum"
#     @assert base_scalar_type(trtype(x)) == base_scalar_type(trtype(y)) == base_scalar_type(accum.type) "x, y, and accum must have the same type, got $x, $y, and $accum"
#     @assert length(size(x)) == 2 && length(size(y)) == 2 && length(size(accum)) == 2 "x, y, and accum must be 2D tensors, got $x, $y, and $accum"
#     @assert size(x, 2) == size(y, 1) "x and y must have compatible shapes, got $x and $y"
#     @assert size(x, 1) == size(accum, 1) && size(y, 2) == size(accum, 2) "y and accum must have compatible shapes, got $y and $accum"
#     @assert size(x, 1) >= 16 && size(x, 2) >= 16 && size(y, 1) >= 16 && size(y, 2) >= 16 "x and y must be at least 16x16, got $x and $y"
#     @assert is_floating(trtype(x)) && is_floating(trtype(y)) && is_floating(accum.type) "TODO x, y, and accum must be floating point tensors, got $x, $y, and $accum"
#     # accum_type = @match base_scalar_type(trtype(x)), base_scalar_type(trtype(y)) begin
#     #     (Tfp32, _) => Tfp32
#     #     (Tbf16, _) => Tfp32
#     #     (_, Tfp16) => Tfp16
#     #     (_, _) => Tfp32
#     # end
#     # accum = zero(x.builder, accum_type)
#     # M = size(x, 1)
#     # N = size(y, 2)
#     # accum_splat = CT.create_splat!(x.builder, accum.handle, [M, N])
#     # ret_ty = BlockTritonType(accum_type, [M, N])
#     Tensor(x.builder, CT.create_dot!(x.builder, x.handle, y.handle, accum.handle, allow_tf32), accum.type)
# end




# MATH OPERATIONS

TritonBlockOrSimple{S, T} = Union{TritonBlockType{S, T}, T}

# TODO should I make this more julian and require broadcasting on block types?

for (fn, node_create_fn) in [(:exp, :create_exp!), (:log, :create_log!), (:cos, :create_cos!), (:sin, :create_sin!), (:sqrt, :create_sqrt!)]
    @eval begin
        function $fn(x::Tensor{T}) where {S, T <: Union{TritonBlockOrSimple{S, TrFloat32}, TritonBlockOrSimple{S, TrFloat64}}}
            Tensor(CT.$node_create_fn(get_builder_ref(), x.handle), trtype(x))
        end
    end
end
@test @wbc begin x = full((16, 32), 5.0); size(sqrt(x)) == [16, 32] end
@test @wbc begin x = Tensor(3.0); size(sqrt(x)) == [] end

function abs(x::Tensor{T}) where T
    if is_floating_bi(T)
        Tensor(CT.create_fabs!(get_builder_ref(), x.handle), T)
    elseif is_signed_bi(T)
        Tensor(CT.create_iabs!(get_builder_ref(), x.handle), T)
    elseif is_integer_bi(T)
        x
    else
        error("Unexpected type $T")
    end
end

# function find_common_broadcast_size(vec_of_dims)
#     lengths = unique(length.(vec_of_dims))

# end

# EXTERNAL CALLS

# String(:asdf)

function external_call(lib_name, lib_path, fn_symbol, ret_type, is_pure, args...)
    builder = get_builder_ref()
    arg_type_objects = [construct_ir_type(builder, trtype(arg)) for arg in args]

    # TODO implicitly broadcast first

    Tensor(CT.create_extern_elementwise!(lib_name, lib_path, String(fn_symbol), args, construct_ir_type(builder, ret_type), is_pure), ret_type)
end