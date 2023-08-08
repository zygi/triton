# include("TritonCxxWrap.jl")
# const CT = CppTriton
##
# include("global_implicit.jl")
# using Test
# using BFloat16s

# using BFloat16s
# using StaticArrays
# using Test

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

macro ok(expr)
    quote
        let ctx = CppTriton.MLIRContext()
            CppTriton.load_triton!(ctx)
            builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
            set_builder_ref(builder)
            res = $expr
            set_builder_ref(nothing)
            true
        end
    end
end

const TrTypeableSimple = Union{
    Nothing,
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    BFloat16,
    Float16,
    Float32,
    Float64,
}

const TrTypeablePtr = Union{
    Ptr{Nothing},
    Ptr{Bool},
    Ptr{Int8},
    Ptr{UInt8},
    Ptr{Int16},
    Ptr{UInt16},
    Ptr{Int32},
    Ptr{UInt32},
    Ptr{Int64},
    Ptr{UInt64},
    Ptr{BFloat16},
    Ptr{Float16},
    Ptr{Float32},
    Ptr{Float64},
}

const TrTypeable = Union{TrTypeableSimple,TrTypeablePtr}

# TODO write note about how constructing tuples with Int32 etc can lead to very confusing and undebuggable errors
abstract type TrBlock{T<:TrTypeable,Size<:Tuple} end

# A new object that's semantically different from a block of pointers
# In principle we might represent it as Ptr{TrBlock{...}} but that would be both imprecise
# and activate a lot of dispatches whitch we don't want to activate. So instead we use a new type.
abstract type TrBlockPtr{T<:TrTypeable,Size<:Tuple} end

const TrType{T} = Union{T,TrBlock{T},TrBlockPtr{T}} where {T<:TrTypeable}

dimtuple_to_vec(::Type{Tuple{}}) = Int64[]
dimtuple_to_vec(::Type{X}) where {X<:Tuple} = collect(fieldtypes(X))

dimtuple_to_tuple(::Type{Tuple{}}) = ()
dimtuple_to_tuple(::Type{X}) where {X<:Tuple} = Tuple(fieldtypes(X))

# Helper functions
Base.size(::Type{TrBlock{T,Size}}) where {Size<:Tuple,T<:TrTypeable} =
    dimtuple_to_tuple(Size)
Base.size(::Type{TrBlockPtr{T,Size}}) where {Size<:Tuple,T<:TrTypeable} =
    dimtuple_to_tuple(Size)
Base.size(::Type{T}) where {T<:TrType} = ()
@test size(TrBlock{Float32,Tuple{2,3}}) == (2, 3)

# numel(::Type{TrType})= prod(size(T))
numel(::Type{T}) where {T<:TrType} = prod(size(T))
@test numel(Float32) == 1
@test numel(TrBlock{Float32,Tuple{2,3}}) == 6

# Can I do this generically? Or more importantly, should I?
_build_tuple_type(::Val{T}) where {T} = Tuple{T}
_build_tuple_type(::Val{T}, ::Val{U}) where {T,U} = Tuple{T,U}
_build_tuple_type(::Val{T}, ::Val{U}, ::Val{V}) where {T,U,V} = Tuple{T,U,V}
_build_tuple_type(::Val{T}, ::Val{U}, ::Val{V}, ::Val{W}) where {T,U,V,W} = Tuple{T,U,V,W}
vec_to_dimtuple(xs) = _build_tuple_type(Val.(collect(Int64, xs))...)

TrBlock(::Type{T}, dims...) where {T<:TrTypeable} = TrBlock{T,vec_to_dimtuple(dims)}
TrBlock(::Type{T}, dims) where {T<:TrTypeable} = TrBlock{T,vec_to_dimtuple(dims)}


construct_ir_type(::Type{Nothing}) = CppTriton.get_void_ty!(get_builder_ref())
construct_ir_type(::Type{Bool}) = CppTriton.get_int1_ty!(get_builder_ref())
construct_ir_type(::Type{Int8}) = CppTriton.get_int8_ty!(get_builder_ref())
construct_ir_type(::Type{UInt8}) = CppTriton.get_int8_ty!(get_builder_ref())
construct_ir_type(::Type{Int16}) = CppTriton.get_int16_ty!(get_builder_ref())
construct_ir_type(::Type{UInt16}) = CppTriton.get_int16_ty!(get_builder_ref())
construct_ir_type(::Type{Int32}) = CppTriton.get_int32_ty!(get_builder_ref())
construct_ir_type(::Type{UInt32}) = CppTriton.get_int32_ty!(get_builder_ref())
construct_ir_type(::Type{Int64}) = CppTriton.get_int64_ty!(get_builder_ref())
construct_ir_type(::Type{UInt64}) = CppTriton.get_int64_ty!(get_builder_ref())
construct_ir_type(::Type{BFloat16}) = CppTriton.get_bf16_ty!(get_builder_ref())
construct_ir_type(::Type{Float16}) = CppTriton.get_half_ty!(get_builder_ref())
construct_ir_type(::Type{Float32}) = CppTriton.get_float_ty!(get_builder_ref())
construct_ir_type(::Type{Float64}) = CppTriton.get_double_ty!(get_builder_ref())

construct_ir_type(::Type{Ptr{T}}) where {T<:TrTypeableSimple} =
    CppTriton.get_ptr_ty!(get_builder_ref(), construct_ir_type(T), 1)
construct_ir_type(::Type{TrBlock{T,Size}}) where {Size<:Tuple,T<:TrTypeable} =
    CppTriton.get_block_ty!(get_builder_ref(), construct_ir_type(T), dimtuple_to_vec(Size))

@test @ok construct_ir_type(Float32)
@test @ok construct_ir_type(Ptr{UInt8})
@test @ok construct_ir_type(TrBlock{UInt8,Tuple{1,2}})
@test @ok construct_ir_type(TrBlock{UInt8,Tuple{}})

base_scalar_type(::Type{T}) where {T<:TrTypeable} = T
base_scalar_type(::Type{TrBlock{T,S}}) where {T<:TrTypeable,S} = T
@test @ok base_scalar_type(TrBlock{UInt8,Tuple{1,2}}) == UInt8

# _bi stands for block included, that is, it returns true for scalar floating types and block floating types
is_floating(::Type{T}) where {U,T<:TrType{U}} =
    U <: Float16 || U <: Float32 || U <: Float64 || U <: BFloat16
@test @wbc is_floating(TrBlock{Float32,Tuple{1}})
@test @wbc !is_floating(Int32)

# pointers are NOT integers
is_integer(::Type{T}) where {U,T<:TrType{U}} =
    U <: Bool ||
    U <: Int8 ||
    U <: UInt8 ||
    U <: Int16 ||
    U <: UInt16 ||
    U <: Int32 ||
    U <: UInt32 ||
    U <: Int64 ||
    U <: UInt64


is_pointer(::Type{<:TrBlockPtr}) = true
is_pointer(::Type{T}) where {U,T<:TrType{U}} = U <: Ptr
@test @wbc is_pointer(Ptr{Int32})
@test @wbc is_pointer(TrBlockPtr{Int32,Tuple{1,2}})
@test @wbc !is_pointer(TrBlock{Int32,Tuple{1,2}})

# pointers ARE scalars
is_scalar(::Type{T}) where {T<:TrTypeable} = true
is_scalar(::Type{T}) where {T<:TrType} = false

is_block(::Type{T}) where {T<:TrType} = !is_scalar(T)

scalar_type_of(::Type{U}) where {T<:TrTypeable,U<:TrType{T}} = T
scalar_type_of(::Type{T}) where {T<:TrTypeable} = T
@test @wbc scalar_type_of(TrBlock{Float32,Tuple{1}}) == Float32

# points_to(::Type{T}) where {U <: TrTypeableSimple, T <: TrType{Ptr{U}}} = U
points_to(::Type{TrBlock{Ptr{T},Size}}) where {Size<:Tuple,T<:TrTypeableSimple} =
    TrBlock{T,Size}
points_to(::Type{TrBlockPtr{T,Size}}) where {Size<:Tuple,T<:TrTypeableSimple} =
    TrBlock{T,Size}
points_to(::Type{Ptr{T}}) where {T<:TrTypeableSimple} = T
@test @wbc points_to(Ptr{Float32}) == Float32
@test @wbc points_to(TrBlock{Ptr{Int32},Tuple{5}}) == TrBlock{Int32,Tuple{5}}

change_scalar_type(
    ::Type{Z},
    ::Type{U},
) where {Size<:Tuple,T<:TrTypeable,U<:TrTypeable,Z<:TrBlock{T,Size}} = TrBlock{U,Size}
change_scalar_type(
    ::Type{Z},
    ::Type{U},
) where {Size<:Tuple,T<:TrTypeable,U<:TrTypeable,Z<:TrBlockPtr{T,Size}} = TrBlockPtr{U,Size}
change_scalar_type(::Type{T}, ::Type{U}) where {T<:TrTypeable,U<:TrTypeable} = U
@test @wbc change_scalar_type(TrBlock{Float32,Tuple{1}}, Float64) ==
           TrBlock{Float64,Tuple{1}}
@test @wbc change_scalar_type(TrBlockPtr{Float32,Tuple{1}}, Float64) ==
           TrBlockPtr{Float64,Tuple{1}}

# primitive_bandwidth(::Type{T}) where {T <: TrTypeable} = primitive_bandwidth(scalar_type_of(x))
primitive_bandwidth(::Type{Bool}) = 1
primitive_bandwidth(::Type{Int8}) = 8
primitive_bandwidth(::Type{UInt8}) = 8
primitive_bandwidth(::Type{Int16}) = 16
primitive_bandwidth(::Type{UInt16}) = 16
primitive_bandwidth(::Type{Int32}) = 32
primitive_bandwidth(::Type{UInt32}) = 32
primitive_bandwidth(::Type{Int64}) = 64
primitive_bandwidth(::Type{UInt64}) = 64
primitive_bandwidth(::Type{BFloat16}) = 16
primitive_bandwidth(::Type{Float16}) = 16
primitive_bandwidth(::Type{Float32}) = 32
primitive_bandwidth(::Type{Float64}) = 64
primitive_bandwidth(::Type{U}) where {U<:Ptr} = 64
@test @ok primitive_bandwidth(Ptr{Float32})

fp_mantissa_width(::Type{BFloat16}) = 10
fp_mantissa_width(::Type{Float16}) = 7
fp_mantissa_width(::Type{Float32}) = 23
fp_mantissa_width(::Type{Float64}) = 53
fp_mantissa_width(::Type{U}) where {U<:TrBlock} = fp_mantissa_width(scalar_type_of(U))
@test @ok fp_mantissa_width(TrBlock{Float32,Tuple{1,2}})

# is_signed(::Type{TrBlock{S, T}}) where {S, T} = is_signed(scalar_type_of(T))
is_signed(::Type{Bool}) = true
is_signed(::Type{Int8}) = true
is_signed(::Type{Int16}) = true
is_signed(::Type{Int32}) = true
is_signed(::Type{Int64}) = true
is_signed(::Type{T}) where {T<:TrTypeable} = false
is_signed(::Type{T}) where {T<:TrBlock} = is_signed(scalar_type_of(T))


is_standard_floating(::Type{TrBlock}) = is_standard_floating(scalar_type_of(T))
is_standard_floating(::Type{BFloat16}) = true
is_standard_floating(::Type{Float16}) = true
is_standard_floating(::Type{Float32}) = true
is_standard_floating(::Type{Float64}) = true
is_standard_floating(::Type{T}) where {T<:TrTypeable} = false


const TypeMLIRToJuliaNameLookup = let
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    with_scoped(builder) do
        Dict([
            CT.repr(CT.CxxRef(construct_ir_type(x))) => x for
            x in Base.uniontypes(TrTypeableSimple)
        ])
    end
end

_parse_type_from_ptrscalar(x::AbstractString) = begin
    if startswith(x, "!tt.ptr<")
        inner = x[9:end-1]
        Ptr{TypeMLIRToJuliaNameLookup[inner]}
    else
        TypeMLIRToJuliaNameLookup[x]
    end
end
@test _parse_type_from_ptrscalar("!tt.ptr<i32>") == Ptr{UInt32}

_parse_type_from_repr(x::AbstractString) = begin
    if startswith(x, "tensor<")
        notensor = x[8:end-1]
        parts = split(String(notensor), 'x')
        if length(parts) == 1
            # a 0-dim tensor
            TrBlock{_parse_type_from_ptrscalar(parts[1]),Int64[]}
        else
            dims = parse.(Int64, parts[1:end-1])
            dims_type = vec_to_dimtuple(dims)
            TrBlock{_parse_type_from_ptrscalar(parts[end]),dims_type}
        end
    else
        _parse_type_from_ptrscalar(x)
    end
end
@test _parse_type_from_repr("tensor<1x2x3x!tt.ptr<i32>>") ==
      TrBlock{Ptr{UInt32},Tuple{1,2,3}}

# _checking_target_type(x::TrBlockPtr{U}) where {U} = U

struct TrVal{T<:TrType}
    handle::Any
    TrVal{T}(handle) where {T<:TrType} = begin
        if !isnothing(handle) && !(T <: TrBlockPtr) # TODO add support for TrBlockPtr checking
            # hacky way to double check types since it's annoying to extract a julia-rep of a type from a handle
            t1 = CT.get_type(handle)
            t2 = construct_ir_type(T)
            t1_repr = CT.repr(CT.CxxRef(t1))
            # @show T
            # @show t1_repr
            t2_repr = CT.repr(CT.CxxRef(t2))
            @assert t1_repr == t2_repr "Type mismatch when wrapping TrVal: passed handle with $t1_repr, type with $t2_repr"
        end
        new{T}(handle)
    end
end
# TrVal(builder, handle, ::Type{T}) where {T<:TrTypeable} = TrVal{T}(builder, handle)
TrVal(::Type{T}, handle) where {T<:TrType} = TrVal{T}(handle)
TrVal(::Type{T}, size, handle) where {T<:TrType} = TrVal{TrBlock(T, size)}(handle)
TrVal(x::T) where {T<:TrVal} = x

trtype(t::TrVal{T}) where {T} = T

Base.show(io::IO, x::TrVal) = begin
    hd = CT.repr(CT.CxxRef(x.handle))
    print(io, "TrVal($hd)")
end

TrVal(b::Bool) = TrVal(Bool, CppTriton.get_int1!(get_builder_ref(), b))
TrVal(x::Int64) = TrVal(Int64, CppTriton.get_int64!(get_builder_ref(), x))
TrVal(x::UInt64) =
    TrVal(UInt64, CppTriton.get_int64!(get_builder_ref(), reinterpret(Int64, x)))
TrVal(x::Int32) = TrVal(Int32, CppTriton.get_int32!(get_builder_ref(), x))
TrVal(x::UInt32) =
    TrVal(UInt32, CppTriton.get_int32!(get_builder_ref(), reinterpret(Int32, x)))

@test @ok TrVal(Int64(1))
@test @ok TrVal(Int64(-2^63))
@test @ok TrVal(Int64(2^63 - 1))
@test @ok TrVal(typemax(UInt64))

TrVal(x::Float32) = TrVal(Float32, CppTriton.get_fp32!(get_builder_ref(), x))
TrVal(x::Float64) = TrVal(Float64, CppTriton.get_fp64!(get_builder_ref(), x))
TrVal(x::Float16) = TrVal(Float16, CppTriton.get_fp16!(get_builder_ref(), Float32(x)))
TrVal(x::BFloat16) = TrVal(BFloat16, CppTriton.get_bf16!(get_builder_ref(), Float32(x)))

IntoTrVal = Union{Bool,Int64,UInt64,Int32,UInt32,Float32,Float64,Float16,TrVal}
# TrVal(x::IntoTrVal) = TrVal(get_builder_ref(), x) end
@test @ok TrVal(5.0)
@test @ok TrVal(TrVal(5.0))


Base.size(t::TrVal{T}) where {T<:TrType} = Base.size(T)
Base.size(t::TrVal{T}, idx) where {T<:TrType} = Base.size(T)[idx]

numel(t::TrVal{T}) where {T<:TrType} = numel(T)

# Tuple{1, 2, 3} |> fieldcount

Base.ndims(::TrVal{TrBlock{T,S}}) where {T,S} = fieldcount(S)
Base.ndims(::TrVal{TrBlockPtr{T,S}}) where {T,S} = fieldcount(S)
Base.ndims(::TrVal{T}) where {T<:TrTypeable} = 0


# glb, glb_ctx = let ctx = CppTriton.MLIRContext()
#     CppTriton.load_triton!(ctx)
#     builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
#     builder, ctx
# end

# cast(input::TrVal{TrBlock{T, S}}, to::Type{U}) where {T, S, U<:TrTypeable} = cast(input, TrBlock{U, S})
# cast(input::TrVal{T}, to::Type{T}) where {T <: TrTypeable} = input

# trval_like(x::TrVal{TrBlock{T, S}}, handle, scalar_type::Type{U}=T) where {T <: TrTypeable, S, U <: TrTypeable} = TrVal{TrBlock{U, S}}(handle)
trval_like(
    x::TrVal,
    handle,
    scalar_type::Type{U} = scalar_type_of(trtype(x)),
) where {U<:TrTypeable} = begin
    TrVal{change_scalar_type(trtype(x), U)}(handle)
end
@test @ok trval_like(TrVal(5), CppTriton.get_fp32!(get_builder_ref(), 5.0f0), Float32)

function arange(start::Integer, endd::Integer)
    start = Int32(start)
    endd = Int32(endd)
    shape = [endd - start]
    TrVal(TrBlock(Int32, shape), CT.create_make_range!(get_builder_ref(), start, endd))
end
@test @wbc begin
    arange(0, 5)
    true
end

function full(dims::Tuple{Vararg{Int64}}, value::T) where {T<:IntoTrVal}
    tens = TrVal(value)
    @assert numel(tens) == 1 "only scalar values are supported"
    TrVal(
        TrBlock(trtype(tens), dims),
        CT.create_splat!(get_builder_ref(), tens.handle, collect(dims)),
    )
end
@test @wbc begin
    full((2, 3), 5)
    true
end

# cast(input::TrVal{TrBlock}, ::Type{T}) where {T <: TrTypeable} = cast(input, TrBlock{T, input.shape})
# cast should not change the shape of input
cast(input::TrVal{H1}, to::Type{H2}) where {T,U,H1<:TrType{T},H2<:TrType{U}} = begin
    if T == U
        return input
    end
    builder = get_builder_ref()
    # T_scalar = scalar_type_of(T)
    # U_scalar = scalar_type_of(U)
    if (T == BFloat16 || T == Float16) && U != Float32
        return trval_like(
            input,
            CT.create_fp_to_fp!(
                builder,
                input.handle,
                construct_ir_type(change_scalar_type(H1, U)),
            ),
            U,
        )
    end

    if is_floating(T) && is_floating(U)
        if primitive_bandwidth(T) > primitive_bandwidth(U)
            return trval_like(
                input,
                CT.create_fp_trunc!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                ),
                U,
            )
        else
            # return TrVal(CT.create_fp_ext!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
            return trval_like(
                input,
                CT.create_fp_ext!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                ),
                U,
            )
        end
    end

    if is_integer(T) &&
       is_integer(U) &&
       (
           (primitive_bandwidth(T) != primitive_bandwidth(U)) ||
           (is_signed(T) != is_signed(U))
       )
        # is this correct? looks suspicious
        needs_sign_extension = is_signed(T) && !is_signed(U)
        if U == Bool
            return input != zero(TrVal{U})
        else
            # return TrVal(CT.create_int_cast!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U)), needs_sign_extension), U)
            return trval_like(
                input,
                CT.create_int_cast!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                    needs_sign_extension,
                ),
                U,
            )
        end
    end

    if is_standard_floating(T) && is_integer(U)
        if U == Bool
            return input != zero(TrVal{U})
        elseif is_signed(U)
            # return TrVal(CT.create_fp_to_si!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
            return trval_like(
                input,
                CT.create_fp_to_si!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                ),
                U,
            )
        else
            # return TrVal(CT.create_fp_to_ui!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
            return trval_like(
                input,
                CT.create_fp_to_ui!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                ),
                U,
            )
        end
    end

    if is_integer(T) && is_standard_floating(U)
        if T == Bool || !is_signed(T)
            # return TrVal(CT.create_ui_to_fp!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
            return trval_like(
                input,
                CT.create_ui_to_fp!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                ),
                U,
            )
        else
            # return TrVal(CT.create_si_to_fp!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
            return trval_like(
                input,
                CT.create_si_to_fp!(
                    builder,
                    input.handle,
                    construct_ir_type(change_scalar_type(H1, U)),
                ),
                U,
            )
        end
    end

    if is_pointer(T) && is_integer(U)
        @assert primitive_bandwidth(U) == 64 "can onnly cast pointers to 64-bit integers"
        # return TrVal(CT.create_ptr_to_int!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
        return trval_like(
            input,
            CT.create_ptr_to_int!(
                builder,
                input.handle,
                construct_ir_type(change_scalar_type(H1, U)),
            ),
            U,
        )
    end

    if is_integer(T) && is_pointer(U)
        # return TrVal(CT.create_int_to_ptr!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
        return trval_like(
            input,
            CT.create_int_to_ptr!(
                builder,
                input.handle,
                construct_ir_type(change_scalar_type(H1, U)),
            ),
            U,
        )
    end

    if is_pointer(T) && is_pointer(U)
        # return TrVal(CT.create_bitcast!(builder, input.handle, construct_ir_type(change_scalar_type(H1, U))), U)
        return trval_like(
            input,
            CT.create_bitcast!(
                builder,
                input.handle,
                construct_ir_type(change_scalar_type(H1, U)),
            ),
            U,
        )
    end

    throw(ArgumentError("unsupported cast from $T to $U"))
end
@test @ok cast(TrVal(5.0), Float32)
@test @ok cast(TrVal(5), Ptr{Bool})
@test @ok cast(full((2, 3), 5), Ptr{Int64})


# Tuple{1, 2} == Tuple{1, 2}

function triton_broadcast(
    lhs::TrVal{TrBlock{T1,S1}},
    rhs::TrVal{TrBlock{T2,S2}},
) where {S1,S2,T1,T2}
    builder = get_builder_ref()
    if S1 == S2
        return lhs, rhs
    end
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
            throw(
                "Trying to broadcast incompatible shapes, shapes lhs: $lhs_dims, rhs: $rhs_dims",
            )
        end
    end
    lhs = if lhs_dims != target_shape
        TrVal(
            T1,
            target_shape,
            CT.create_broadcast!(builder, lhs.handle, collect(target_shape)),
        )
    else
        lhs
    end

    rhs = if rhs_dims != target_shape
        TrVal(
            T2,
            target_shape,
            CT.create_broadcast!(builder, rhs.handle, collect(target_shape)),
        )
    else
        rhs
    end

    return lhs, rhs
end
function triton_broadcast(lhs::TrVal{TrBlock{T1,S1}}, rhs::TrVal{T2}) where {S1,T1,T2}
    lhs,
    TrVal(
        T2,
        dimtuple_to_vec(S1),
        CT.create_splat!(get_builder_ref(), rhs.handle, dimtuple_to_vec(S1)),
    )
end
function triton_broadcast(lhs::TrVal{T1}, rhs::TrVal{TrBlock{T2,S2}}) where {S2,T1,T2}
    TrVal(
        T1,
        dimtuple_to_vec(S2),
        CT.create_splat!(get_builder_ref(), lhs.handle, dimtuple_to_vec(S2)),
    ),
    rhs
end
function triton_broadcast(
    lhs::TrVal{T1},
    rhs::TrVal{T2},
) where {T1<:TrTypeable,T2<:TrTypeable}
    lhs, rhs
end

triton_broadcast(a, b, c) = begin
    a, b = triton_broadcast(a, b)
    b, c = triton_broadcast(b, c)
    a, b = triton_broadcast(a, b)
    a, b, c
end

@test @wbc begin
    t1 = TrVal(1.0)
    t2 = full((10, 20), 5)
    t1, t2 = triton_broadcast(t1, t2)
    size(t1) == size(t2)
end

program_id(axis) = TrVal(Int32, CT.create_get_program_id!(get_builder_ref(), axis - 1))
num_programs(axis) = TrVal(Int32, CT.create_get_num_programs!(get_builder_ref(), axis - 1))
@test @ok program_id(1)
@test @ok num_programs(1)


using Expronicon
using MLStyle

_split_arg(e) = @match e begin
    Expr(:(::), name, Tsym) => (name, Tsym)
    x => begin
        x.head
        throw("Binary op must take two tensors")
    end
end
macro binary_op_implicit_casting(fn)
    jlfn = JLFunction(fn)
    @assert _split_arg(jlfn.args[1])[2] == :TrVal && _split_arg(jlfn.args[2])[2] == :TrVal "Binary op must take two tensors"
    @assert length(jlfn.args) == 2 "Binary op must take two tensors"

    arg_names = map(x -> _split_arg(x)[1], jlfn.args)

    orig_args = copy(jlfn.args)
    jlfn.args[1] = Expr(:(::), arg_names[1], :IntoTrVal)
    jlfn.body = quote
        $(jlfn.name)(TrVal($(arg_names[1])), $(arg_names[2]))
    end
    left_fn = codegen_ast(jlfn)

    jlfn.args = orig_args
    jlfn.args[2] = Expr(:(::), _split_arg(jlfn.args[2])[1], :IntoTrVal)
    jlfn.body = quote
        $(jlfn.name)($(arg_names[1]), TrVal($(arg_names[2])))
    end
    right_fn = codegen_ast(jlfn)


    quote
        $(esc(fn))
        $(esc(left_fn))
        $(esc(right_fn))
    end
end

@binary_op_implicit_casting Base.:+(lhs::TrVal, rhs::TrVal) = begin
    lhs, rhs = triton_broadcast(lhs, rhs)
    # @assert types_shapes_match_uptopointer(lhs, rhs) "Types and shapes must match, got x: $lhs, y: $rhs"

    # offset + ptr
    # ptr + offset
    if is_pointer(trtype(rhs)) && !is_pointer(trtype(lhs))
        lhs, rhs = rhs, lhs
    end
    if is_pointer(trtype(lhs))
        return trval_like(lhs, CT.create_addptr!(get_builder_ref(), lhs.handle, rhs.handle))
    end

    # float + float
    if is_floating(trtype(lhs)) && is_floating(trtype(rhs))
        return trval_like(lhs, CT.create_fadd!(get_builder_ref(), lhs.handle, rhs.handle))
    end

    if is_integer(trtype(lhs)) && is_integer(trtype(rhs))
        return trval_like(lhs, CT.create_add!(get_builder_ref(), lhs.handle, rhs.handle))
    end

    throw("Can't add $lhs and $rhs")
end

@test @wbc begin
    TrVal(2.0) + TrVal(5.0)
    true
end
@test @wbc begin
    TrVal(2.0) + 1.0
    true
end
@test @wbc begin
    full((5,), 3) + TrVal(1)
    true
end

Base.zero(::Type{T}) where {U<:TrTypeable,T<:TrVal{U}} =
    TrVal(U, CT.get_null_value!(get_builder_ref(), construct_ir_type(U)))
Base.zero(x::TrVal) = zero(typeof(x))
@test @wbc begin
    zero(TrVal{Int32})
    true
end


Base.one(::Type{T}) where {U<:TrTypeable,T<:TrVal{U}} = TrVal(one(U))
Base.one(x::TrVal) = one(typeof(x))
@test @wbc begin
    one(TrVal{Float32})
    true
end


@binary_op_implicit_casting Base.:-(x::TrVal, y::TrVal) = begin
    x, y = triton_broadcast(x, y)
    if is_pointer(trtype(x))
        return trval_like(x, CT.create_addptr!(get_builder_ref(), x.handle, (-y).handle))
    end
    if is_floating(trtype(x)) && is_floating(trtype(y))
        return trval_like(x, CT.create_fsub!(get_builder_ref(), x.handle, y.handle))
    end
    if is_integer(trtype(x)) && is_integer(trtype(y))
        return trval_like(x, CT.create_sub!(get_builder_ref(), x.handle, y.handle))
    end
    throw("Can't subtract $x and $y")
end
Base.:-(x::TrVal) = begin
    is_pointer(trtype(x)) && throw("Can't negate a pointer")
    zero(typeof(x)) - x
end
@test @wbc begin
    -TrVal(1.0)
    true
end
@test @wbc begin
    cast(TrVal(1), Ptr{Int64}) - TrVal(Int32(2))
    true
end

@binary_op_implicit_casting Base.:*(x::TrVal, y::TrVal) = begin
    x, y = triton_broadcast(x, y)
    if is_floating(trtype(x)) && is_floating(trtype(y))
        return trval_like(x, CT.create_fmul!(get_builder_ref(), x.handle, y.handle))
    end
    if is_integer(trtype(x)) && is_integer(trtype(y))
        return trval_like(x, CT.create_mul!(get_builder_ref(), x.handle, y.handle))
    end
    throw("Can't multiply $x and $y")
end
@test @wbc begin
    TrVal(1.0) * TrVal(2.0)
    true
end

@binary_op_implicit_casting Base.:/(x::TrVal, y::TrVal) = begin
    x, y = triton_broadcast(x, y)
    if is_floating(trtype(x)) && is_integer(trtype(y))
        y = cast(y, trtype(x))
    elseif is_integer(trtype(x)) && is_floating(trtype(y))
        x = cast(x, trtype(y))
    elseif is_floating(trtype(x)) && is_floating(trtype(y))
        if fp_mantissa_width(trtype(x)) > fp_mantissa_width(trtype(y))
            y = cast(y, trtype(x))
        else
            x = cast(x, trtype(y))
        end
    else
        # TODO think about int/int
        throw("Can't divide $x and $y")
    end
    return trval_like(x, CT.create_fdiv!(get_builder_ref(), x.handle, y.handle))
end
@test @wbc (TrVal(1.0) / TrVal(2.0f0)) |> trtype == Float64

@binary_op_implicit_casting Base.div(x::TrVal, y::TrVal) = begin
    x, y = triton_broadcast(x, y)
    if is_integer(trtype(x)) && is_integer(trtype(y))
        if is_signed(trtype(x))
            return trval_like(x, CT.create_sdiv!(get_builder_ref(), x.handle, y.handle))
        else
            return trval_like(x, CT.create_udiv!(get_builder_ref(), x.handle, y.handle))
        end
    end
    throw("Can't divide $x and $y")
end
@test @ok TrVal(1) ÷ TrVal(2)

cdiv(x, y) = (x + y - one(x)) ÷ y
@test @ok cdiv(TrVal(5), TrVal(2))

@binary_op_implicit_casting Base.rem(x::TrVal, y::TrVal) = begin
    x, y = triton_broadcast(x, y)
    if is_integer(trtype(x)) && is_integer(trtype(y))
        @assert is_signed(trtype(x)) == is_signed(trtype(y)) "Types must be both signed or both unsigned"
        if is_signed(trtype(x))
            return trval_like(x, CT.create_srem!(get_builder_ref(), x.handle, y.handle))
        else
            return trval_like(x, CT.create_urem!(get_builder_ref(), x.handle, y.handle))
        end
    end
    # TODO think about float % float
    throw("Can't divide $x and $y")
end
@test @ok TrVal(5) % TrVal(2)

# base_eq(x::TrVal, y::TrVal) = x == y
# base_neq(x::TrVal, y::TrVal) = x != y


COMPARISON_OPS = [
    (:<, :create_fcmpOLT!, :create_icmpSLT!, :create_icmpULT!),
    (:<=, :create_fcmpOLE!, :create_icmpSLE!, :create_icmpULE!),
    (:>, :create_fcmpOGT!, :create_icmpSGT!, :create_icmpUGT!),
    (:>=, :create_fcmpOGE!, :create_icmpSGE!, :create_icmpUGE!),
    (:(==), :create_fcmpOEQ!, :create_icmpEQ!, :create_icmpEQ!),
    (:!=, :create_fcmpUNE!, :create_icmpNE!, :create_icmpNE!),
]
for (op_name, float_op, signed_op, unsigned_op) in COMPARISON_OPS
    eval(
        quote
            @binary_op_implicit_casting Base.$op_name(x::TrVal, y::TrVal) = begin
                x, y = triton_broadcast(x, y)
                # return_ty = change_scalar_type(trtype(x), TrBool)

                if is_floating(trtype(x)) && is_floating(trtype(y))
                    return trval_like(
                        x,
                        CT.$float_op(get_builder_ref(), x.handle, y.handle),
                        Bool,
                    )
                end
                if is_integer(trtype(x)) && is_integer(trtype(y))
                    if is_signed(trtype(x))
                        return trval_like(
                            x,
                            CT.$signed_op(get_builder_ref(), x.handle, y.handle),
                            Bool,
                        )
                    else
                        return trval_like(
                            x,
                            CT.$unsigned_op(get_builder_ref(), x.handle, y.handle),
                            Bool,
                        )
                    end
                end
                if is_pointer(trtype(x)) && is_pointer(trtype(y))
                    # TODO decide once and for all if pointers are signed or unsigned 
                    return trval_like(
                        x,
                        CT.$signed_op(get_builder_ref(), x.handle, y.handle),
                        Bool,
                    )
                end
                throw("Can't compare $x and $y")
            end

            @test @wbc ($op_name(TrVal(1.0), TrVal(2.0))) |> trtype == Bool
            @test @wbc ($op_name(TrVal(Int32(1)), TrVal(Int32(1)))) |> trtype == Bool
            @test @wbc ($op_name(TrVal(UInt32(1)), TrVal(UInt32(1)))) |> trtype == Bool
            @test @wbc ($op_name(full((5,), 5.0), full((5,), 4.0))) |> trtype ==
                       TrBlock{Bool,Tuple{5}}
        end,
    )
end

for (op_name, create_op) in [(:&, :create_and!), (:|, :create_or!), (:^, :create_xor!)]
    eval(
        quote
            @binary_op_implicit_casting Base.$op_name(x::TrVal, y::TrVal) = begin
                x, y = triton_broadcast(x, y)
                @assert is_integer(trtype(x)) && is_integer(trtype(y)) "Both operands must be integers, got x: $(trtype(x)) and y: $(trtype(y))"
                trval_like(x, CT.$create_op(get_builder_ref(), x.handle, y.handle))
            end

            @test @wbc ($op_name(TrVal(Int32(1)), TrVal(Int32(1)))) |> trtype == Int32
        end,
    )
end

expanddims(x::TrVal{TrBlock{T,S}}, axis::Int) where {S,T} = begin
    # @assert is_block(trtype(x))
    dims = dimtuple_to_vec(S)
    @assert axis >= 1 && axis <= length(dims) + 1 "Axis must be in range [1, length(dims)+1]"
    new_shape = similar(dims, length(dims) + 1)
    new_shape[1:(axis-1)] .= dims[1:axis-1]
    new_shape[axis] = 1
    new_shape[axis+1:end] .= dims[axis:end]
    new_type = TrBlock{T,vec_to_dimtuple(new_shape)}
    TrVal(new_type, CT.create_expand_dims!(get_builder_ref(), x.handle, axis - 1))
end
@test @wbc size(expanddims(full((2, 3), 1.0), 2)) == (2, 1, 3)


function broadcast_impl_shape(x::TrVal{TrBlock{T,S}}, shape) where {S,T}
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
    TrVal(T, new_shape, CT.create_broadcast!(get_builder_ref(), x.handle, new_shape))
end
function broadcast_impl_shape(x::TrVal{T}, shape) where {T<:TrTypeable}
    TrVal(T, shape, CT.create_splat!(get_builder_ref(), x.handle, collect(Int64, shape)))
end
broadcast_impl_shape(x, shape) = broadcast_impl_shape(TrVal(x), shape)


@test @wbc size(broadcast_impl_shape(TrVal(1.0), [2, 3])) == (2, 3)
@test @wbc begin
    res = broadcast_impl_shape(full((2, 1, 3), 1.0f0), [2, 5, 3])
    size(res) == (2, 5, 3) && scalar_type_of(trtype(res)) == Float32
end

Base.zeros(::Type{TrVal{T}}, dims::NTuple{N, <:Integer}) where {T, N} = broadcast_impl_shape(zero(T), dims)
Base.zeros(::Type{TrVal{T}}, dims::Vararg{Union{Integer, AbstractUnitRange}}) where {T} = broadcast_impl_shape(zero(T), collect(Int64, dims))
@test @wbc isa(zeros(TrVal{Float32}, (2, 3)), TrVal)
@test @wbc isa(zeros(TrVal{Float32}, 2, 3, 5), TrVal)

Base.ones(::Type{TrVal{T}}, dims::NTuple{N, <:Integer}) where {T, N} = broadcast_impl_shape(one(T), dims)
Base.ones(::Type{TrVal{T}}, dims::Vararg{Union{Integer, AbstractUnitRange}}) where {T} = broadcast_impl_shape(one(T), collect(Int64, dims))
@test @wbc isa(ones(TrVal{Float32}, (2, 3)), TrVal)
@test @wbc isa(ones(TrVal{Float32}, 2, 3, 5), TrVal)

_string_to_load_cache_modifier(x) = begin
    if x == ".ca"
        return CppTriton.CM_CA
    end
    if x == ".cg"
        return CppTriton.CM_CG
    end
    if x == ""
        return CppTriton.CM_NONE
    end
    throw("Unknown cache modifier $x")
end

_string_to_store_cache_modifier(x) = begin
    if x == ".wb"
        return CppTriton.CM_WB
    end
    if x == ".wt"
        return CppTriton.CM_WT
    end
    if x == ".cg"
        return CppTriton.CM_CG
    end
    if x == ".cs"
        return CppTriton.CM_CS
    end
    if x == ""
        return CppTriton.CM_NONE
    end
    throw("Unknown cache modifier $x")
end

_string_to_eviction_policy(x) = begin
    if x == "evict_last"
        return CppTriton.EP_EVICT_LAST
    end
    if x == "evict_first"
        return CppTriton.EP_EVICT_FIRST
    end
    if x == ""
        return CppTriton.EP_NORMAL
    end
    throw("Unknown eviction policy $x")
end

_string_to_padding_option(x) = begin
    if x == "zero"
        return CppTriton.PO_PAD_ZERO
    end
    if x == "nan"
        return CppTriton.PO_PAD_NAN
    end
    throw("Unknown padding option $x")
end

shapes_match(x::TrVal, y::TrVal) = size(x) == size(y)


# points_to_type(ptr::TrVal{TrBlock{Ptr{T}, S}}, dst::TrVal{TrBlock{T, S}}) where {S, T} = true
# points_to_type(ptr::TrVal{Ptr{T}}, dst::TrVal{T}) where {T <: TrTypeable} = true
points_to_type(ptr::TrVal, dst::TrVal) =
    is_pointer(trtype(ptr)) && points_to(trtype(ptr)) == trtype(dst)
@test @wbc !points_to_type(full((2, 3), 1.0), full((2, 3), 1.0))
@test @wbc points_to_type(cast(TrVal(5), Ptr{Int64}), TrVal(5))

_load_legacy(
    ptr::TrVal,
    mask::Union{TrVal,Nothing},
    other::Union{TrVal,Nothing},
    cache,
    eviction,
    is_volatile,
) = begin
    @assert is_pointer(trtype(ptr)) "ptr must be a pointer or a pointer block"
    @assert primitive_bandwidth(scalar_type_of(trtype(ptr))) != 1 "TODO ptr can't point to bools"
    @assert isnothing(mask) == isnothing(other) "mask and other must be either both nothing or both tensors"

    if !isnothing(mask)
        ptr, mask, other = triton_broadcast(ptr, mask, other)
    end

    @assert isnothing(mask) ||
            (scalar_type_of(trtype(mask)) == Bool && shapes_match(ptr, mask)) "mask must be a boolean tensor with the same shape as ptr, got $mask and $ptr"
    @assert isnothing(other) || points_to_type(ptr, other) "ptr must have the same type as other, got $ptr and $other"

    # result_ty = change_scalar_type(trtype(ptr), spoints_to(trtype(ptr)))

    if isnothing(mask)
        trval_like(
            ptr,
            CT.create_load!(get_builder_ref(), ptr.handle, cache, eviction, is_volatile),
            scalar_type_of(points_to(trtype(ptr))),
        )
    else
        trval_like(
            ptr,
            CT.create_masked_load!(
                get_builder_ref(),
                ptr.handle,
                mask.handle,
                other.handle,
                cache,
                eviction,
                is_volatile,
            ),
            scalar_type_of(points_to(trtype(ptr))),
        )
    end
end

# @wbc @show trtype(cast(full((2, 3), 5), Ptr{Int64}))


@test @wbc begin
    ptr = cast(TrVal(1), Ptr{Float32})
    mask = TrVal(true)
    other = TrVal(2.0f0)
    _load_legacy(ptr, mask, other, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    _load_legacy(ptr, nothing, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    true
end
@test @wbc begin
    ptr = cast(full((2, 3), 5), Ptr{Int64})
    mask = full((2, 3), true)
    other = full((2, 3), 2)
    _load_legacy(ptr, nothing, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    _load_legacy(ptr, mask, other, CppTriton.CM_NONE, CppTriton.EP_NORMAL, false)
    true
end


load(
    ptr::TrVal{<:TrBlock};
    mask::Union{IntoTrVal,Nothing} = nothing,
    other::Union{IntoTrVal,Nothing} = nothing,
    cache = "",
    eviction = "",
    is_volatile = false,
) = begin
    ptr = TrVal(ptr)
    if !isnothing(mask)
        mask = TrVal(mask)
    end
    if !isnothing(other)
        other = TrVal(other)
    end
    _load_legacy(
        ptr,
        mask,
        other,
        _string_to_load_cache_modifier(cache),
        _string_to_eviction_policy(eviction),
        is_volatile,
    )
end



_store_legacy(ptr::TrVal, val::TrVal, mask::Union{TrVal,Nothing}, cache, eviction) = begin
    @assert is_pointer(trtype(ptr)) "ptr must be a pointer or a pointer block"
    @assert primitive_bandwidth(scalar_type_of(trtype(ptr))) != 1 "TODO ptr can't point to bools"

    if !isnothing(mask) && is_block(trtype(mask))
        ptr, val, mask = triton_broadcast(ptr, val, mask)
    else
        ptr, val = triton_broadcast(ptr, val)
    end

    @assert points_to_type(ptr, val) "ptr must be ptr<T> where val is <T>, got ptr: $ptr and val: $val"
    @assert isnothing(mask) || (scalar_type_of(trtype(mask)) == Bool) "mask must be a boolean tensor, got $mask"
    @assert isnothing(mask) || shapes_match(ptr, mask) "mask must be a boolean tensor with the same shape as ptr, got $mask and $ptr"

    if isnothing(mask)
        CT.create_store!(get_builder_ref(), ptr.handle, val.handle, cache, eviction)
    else
        CT.create_masked_store!(
            get_builder_ref(),
            ptr.handle,
            val.handle,
            mask.handle,
            cache,
            eviction,
        )
    end
end

@test @wbc begin
    ptr = cast(TrVal(1), Ptr{Float32})
    val = TrVal(2.0f0)
    mask = TrVal(true)
    _store_legacy(ptr, val, mask, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    _store_legacy(ptr, val, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    true
end
@test @wbc begin
    ptr = cast(full((2, 3), 5), Ptr{Int64})
    val = full((2, 3), 2)
    mask = full((2, 3), true)
    _store_legacy(ptr, val, mask, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    _store_legacy(ptr, val, nothing, CppTriton.CM_NONE, CppTriton.EP_NORMAL)
    true
end

store!(
    ptr::TrVal{<:TrBlock},
    val::IntoTrVal;
    mask::Union{IntoTrVal,Nothing} = nothing,
    cache = "",
    eviction = "",
) = begin
    _store_legacy(
        TrVal(ptr),
        TrVal(val),
        isnothing(mask) ? nothing : TrVal(mask),
        _string_to_store_cache_modifier(cache),
        _string_to_eviction_policy(eviction),
    )
end



triton_return() = CT.ret!(get_builder_ref(), CT.CxxRef{CT.Value}[])
@test @wbc begin
    triton_return()
    true
end

device_print(prefix, args...) = begin
    handles = collect(
        map(x -> Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(x.handle)), args),
    )
    CT.create_print!(get_builder_ref(), prefix, handles)
end
@test @wbc begin
    device_print("hello", full((2, 3), 5), full((2, 3), 5))
    true
end


triton_yield(vs...) = begin
    if isempty(vs)
        CT.create_yield_op!(get_builder_ref(), CT.CxxRef{CT.Value}[])
    else
        CT.create_yield_op!(
            get_builder_ref(),
            collect(
                map(
                    x -> Base.unsafe_convert(
                        CT.CxxRef{CT.Value},
                        CT.CxxRef(TrVal(x).handle),
                    ),
                    vs,
                ),
            ),
        )
    end
end
@test @wbc begin
    triton_yield(full((2, 3), 5), full((2, 3), 5), 2.0)
    true
end

types_shapes_match(x, y) =
    shapes_match(x, y) && scalar_type_of(trtype(x)) == scalar_type_of(trtype(y))

triton_where(cond::IntoTrVal, x::IntoTrVal, y::IntoTrVal) = begin
    cond = TrVal(cond)
    x = TrVal(x)
    y = TrVal(y)
    cond = cast(cond, Bool)
    if is_block(trtype(cond))
        cond, x, y = triton_broadcast(cond, x, y)
    else
        x, y = triton_broadcast(x, y)
    end
    @assert types_shapes_match(x, y) "x and y must have the same type and shape, got $x and $y"
    trval_like(x, CT.create_select!(get_builder_ref(), cond.handle, x.handle, y.handle))
end
@test @wbc begin
    cond = full((2, 3), true)
    x = full((2, 3), 5)
    y = full((2, 3), 2)
    triton_where(cond, x, y)
    true
end
@test @wbc begin
    cond = TrVal(true)
    x = full((2, 3), 5)
    y = full((2, 3), 2)
    triton_where(cond, x, y)
    true
end


@binary_op_implicit_casting Base.min(x::TrVal, y::TrVal) = triton_where(x < y, x, y)
@binary_op_implicit_casting Base.max(x::TrVal, y::TrVal) = triton_where(x > y, x, y)
@test @wbc begin
    x = full((2, 3), 5)
    y = full((2, 3), 2)
    min(x, y)
    max(x, y)
    true
end


_dot_accum_type(::Type{Float32}, ::Type, output_type) = Float32
_dot_accum_type(::Type{Float16}, ::Type, output_type) = Float32
_dot_accum_type(::Type, ::Type, output_type) = output_type

dot(
    x::TrVal,
    y::TrVal;
    allow_tf32 = true,
    output_ty::Type{T} = Float32,
) where {T<:TrTypeableSimple} = begin
    @assert is_block(trtype(x)) && is_block(trtype(y)) "x and y must be block tensors, got $x and $y"
    @assert scalar_type_of(trtype(x)) == scalar_type_of(trtype(y)) "x and y must have the same type, got $x and $y"
    @assert length(size(x)) == 2 && length(size(y)) == 2 "x and y must be 2D tensors, got $x and $y"
    @assert size(x, 2) == size(y, 1) "x and y must have compatible shapes, got $x and $y"
    @assert size(x, 1) >= 16 && size(x, 2) >= 16 && size(y, 1) >= 16 && size(y, 2) >= 16 "x and y must be at least 16x16, got $x and $y"
    @assert is_floating(trtype(x)) && is_floating(trtype(y)) "TODO x and y must be floating point tensors, got $x and $y"
    accum_type =
        _dot_accum_type(scalar_type_of(trtype(x)), scalar_type_of(trtype(y)), output_ty)
    accum = zero(TrVal{accum_type})
    M = size(x, 1)
    N = size(y, 2)
    accum_splat = CT.create_splat!(get_builder_ref(), accum.handle, [M, N])
    ret_ty = TrBlock{accum_type,Tuple{M,N}}
    TrVal(
        ret_ty,
        CT.create_dot!(get_builder_ref(), x.handle, y.handle, accum_splat, allow_tf32),
    )
end
@test @wbc begin
    x = full((16, 32), 5.0)
    y = full((32, 64), 2.0)
    size(dot(x, y)) == (16, 64)
end

# dot_fma(x::TrVal, y::TrVal, accum::TrVal; allow_tf32 = true) = begin
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
#     # accum = zero(get_builder_ref(), accum_type)
#     # M = size(x, 1)
#     # N = size(y, 2)
#     # accum_splat = CT.create_splat!(get_builder_ref(), accum.handle, [M, N])
#     # ret_ty = BlockTrTypeable(accum_type, [M, N])
#     TrVal(get_builder_ref(), CT.create_dot!(get_builder_ref(), x.handle, y.handle, accum.handle, allow_tf32), accum.type)
# end




# MATH OPERATIONS

# TritonBlockOrSimple{S, T} = Union{TrBlock{S, T}, T}

# TODO should I make this more julian and require broadcasting on block types?

for (fn, node_create_fn) in [
    (:exp, :create_exp!),
    (:log, :create_log!),
    (:cos, :create_cos!),
    (:sin, :create_sin!),
    (:sqrt, :create_sqrt!),
]
    @eval begin
        function $fn(x::TrVal{T}) where {T<:Union{TrType{Float32},TrType{Float64}}}
            trval_like(x, CT.$node_create_fn(get_builder_ref(), x.handle))
        end
    end
end
@test @wbc begin
    x = full((16, 32), 5.0)
    size(sqrt(x)) == (16, 32)
end
@test @wbc begin
    x = TrVal(3.0)
    size(sqrt(x)) == ()
end

function abs(x::TrVal{T}) where {T}
    if is_floating(T)
        trval_like(x, CT.create_fabs!(get_builder_ref(), x.handle))
    elseif is_signed(T)
        trval_like(x, CT.create_iabs!(get_builder_ref(), x.handle))
    elseif is_integer(T)
        x
    else
        error("Unexpected type $T")
    end
end


Base.convert(::Type{TrVal}, x::IntoTrVal) = TrVal(x)

# map(x -> Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(x.handle)), args)
_cpp_vec_of_refs(ls) = CT.StdVector([Base.unsafe_convert(CT.CxxRef{CT.Value}, CT.CxxRef(x.handle)) for x in ls])

function block_ptr(base::TrVal; parent_shape, parent_strides, offsets, block_shape, order)
    # TODO this is probably bad style bc it causes type instability. What's a more elegant way to write this?
    parent_shape = cast.(collect(TrVal, parent_shape), Int32)
    parent_strides = cast.(collect(TrVal, parent_strides), Int32)
    offsets = cast.(collect(TrVal, offsets), Int64)

    @assert is_pointer(trtype(base)) "base must be a pointer, got $(trtype(base))"
    @assert !is_block(trtype(base)) "base must not be a block, got $(trtype(base))"

    # note: bool should be handled carefully, as int8
    @assert points_to(trtype(base)) != Bool "TODO pointers to bool not yet supported"

    try
        block_shape = collect(Int32, block_shape)
    catch
        throw(ArgumentError("block_shape must be a tuple of static integers"))
    end

    try
        order = collect(Int32, order)
    catch
        throw(ArgumentError("order must be a tuple of static integers"))
    end
    @assert sort(order) == sort(1:length(order)) "order must be a permutation of 1:length(order), got $order"

    @assert length(parent_shape) ==
            length(parent_strides) ==
            length(offsets) ==
            length(block_shape) ==
            length(order) "parent_shape, parent_strides, offsets, block_shape, and order must all have the same length"


    handle = CT.create_make_block_ptr!(
        get_builder_ref(),
        base.handle,
        _cpp_vec_of_refs(parent_shape),
        _cpp_vec_of_refs(parent_strides),
        _cpp_vec_of_refs(offsets),
        CT.StdVector(block_shape),
        CT.StdVector(order .- Int32(1)),
    )
    TrVal{TrBlockPtr{points_to(trtype(base)),vec_to_dimtuple(block_shape)}}(handle)
end
@test @ok begin
    block_ptr(
        cast(zero(TrVal{Int64}), Ptr{Float32}),
        parent_shape=(16, 32),
        parent_strides=(32, 1),
        offsets=(0, 0),
        block_shape=(4, 4),
        order=(2, 1),
    )
end

function advance(base::TrVal{<:TrBlockPtr}, offsets)
    return TrVal{trtype(base)}(
        CT.create_advance!(
            get_builder_ref(),
            base.handle,
            _cpp_vec_of_refs(collect(TrVal, offsets)),
        ),
    )
end

@test @ok begin
    bptr = block_ptr(
        cast(zero(TrVal{Int64}), Ptr{Float32}),
        parent_shape=(16, 32),
        parent_strides=(32, 1),
        offsets=(0, 0),
        block_shape=(4, 4),
        order=(2, 1),
    )
    size(advance(bptr, (1, 1))) == size(bptr)
end

_process_boundary_check(xs, len) = begin
    bs = collect(Int32, xs)
    @assert Set(bs) == Set(unique(bs)) "boundary_check must not contain duplicates, got $(xs)"
    @assert Set(unique(bs)) ⊆ Set(1:len) "boundary_check must be a subset of 1:ndims(ptr), got $(xs)"
    sort(bs) .- Int32(1)
end

function store(
    ptr::TrVal{TrBlockPtr{T, S}},
    val_p::IntoTrVal;
    boundary_check = 1:ndims(ptr),
    cache = "",
    eviction = "",
) where {T, S}
    val = TrVal(val_p)
    if is_scalar(trtype(val))
        val = broadcast_impl_shape(val, size(ptr))
    end
    @assert shapes_match(ptr, val) "shapes of the block pointer and the value must be compatible, got $(size(ptr)) and $(size(val))"
    @assert points_to_type(ptr, val) "pointed-to types must match, got $(points_to(trtype(ptr))) and $(trtype(val))"
    @assert trtype(val) != Bool "TODO pointers to bool not yet supported"

    # boundary_check = collect(Int32, boundary_check_arg)
    # @assert Set(unique(boundary_check)) ⊆ Set(1:ndims(ptr)) "boundary_check must be a subset of 1:ndims(ptr), got $(boundary_check_arg)"
    # @assert Set(boundary_check) == Set(unique(boundary_check)) "boundary_check must not contain duplicates, got $(boundary_check_arg)"

    CT.create_tensor_pointer_store!(
        get_builder_ref(),
        ptr.handle,
        val.handle,
        CT.StdVector(_process_boundary_check(boundary_check, ndims(ptr))),
        _string_to_store_cache_modifier(cache),
        _string_to_eviction_policy(eviction),
    )
end
@test @ok begin
    bptr = block_ptr(
        cast(zero(TrVal{Int64}), Ptr{Float32});
        parent_shape=(16, 32),
        parent_strides=(32, 1),
        offsets=(0, 0),
        block_shape=(4, 4),
        order=(2, 1),
    )

    store(bptr, 3.0f0)
    store(bptr, ones(TrVal{Float32}, 4, 4))
end

function load(
    ptr::TrVal{TrBlockPtr{T, S}};
    boundary_check = 1:ndims(ptr),
    padding_option = "zero",
    cache = "",
    eviction = "",
    is_volatile = false
) where {T, S}
    @assert T != Bool "TODO pointers to bool not yet supported"
    padding_parsed = _string_to_padding_option(padding_option)
    if !is_floating(T) && padding_parsed == CppTriton.PO_PAD_NAN
        throw(ArgumentError("padding_option = \"nan\" is only supported for floating point types"))
    end
    
    handle = CT.create_tensor_pointer_load!(
        get_builder_ref(),
        ptr.handle,
        CT.StdVector(_process_boundary_check(boundary_check, ndims(ptr))),
        padding_parsed,
        _string_to_store_cache_modifier(cache),
        _string_to_eviction_policy(eviction),
        is_volatile
    )
    TrVal(TrBlock{T, S}, handle)
end
@test @ok begin
    bptr = block_ptr(
        cast(zero(TrVal{Int64}), Ptr{Float32}),
        parent_shape=(16, 32),
        parent_strides=(32, 1),
        offsets=(0, 0),
        block_shape=(4, 4),
        order=(2, 1),
    )

    load(bptr; boundary_check=1:2)
end



# def advance(base: tl.tensor, offsets, builder: ir.builder) -> tl.tensor:
#     # Convert dynamic offsets to IR values
#     offsets = _convert_to_ir_values(builder, offsets, require_i64=False)

#     # Advanced block pointer type is the same as before
#     return tl.tensor(builder.create_advance(base.handle, offsets), base.type)





# function find_common_broadcast_size(vec_of_dims)
#     lengths = unique(length.(vec_of_dims))

# end

# EXTERNAL CALLS

# String(:asdf)

function external_call(lib_name, lib_path, fn_symbol, ret_type, is_pure, args...)
    builder = get_builder_ref()
    arg_type_objects = [construct_ir_type(builder, trtype(arg)) for arg in args]

    # TODO implicitly broadcast first

    TrVal(
        CT.create_extern_elementwise!(
            lib_name,
            lib_path,
            String(fn_symbol),
            args,
            construct_ir_type(builder, ret_type),
            is_pure,
        ),
        ret_type,
    )
end



