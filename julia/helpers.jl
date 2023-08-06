using MLStyle
using Test
using StructEquality
using InteractiveUtils: subtypes


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

# macro test_throws_m(filter, expr)
#     quote
#         @test_throws $filter let ctx = CppTriton.MLIRContext()
#             CppTriton.load_triton!(ctx)
#             builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
#             $expr
#         end
#     end
# end


abstract type TritonType end
is_scalar(x::TritonType) = false
is_block(x::TritonType) = false
is_pointer(x::TritonType) = false

@data ScalarTritonType <: TritonType begin
    Tvoid
    Tint1
    Tint8
    Tuint8
    Tint16
    Tuint16
    Tint32
    Tuint32
    Tint64
    Tuint64
    Tfp8e5
    Tfp8e4
    Tfp8e4b15
    Tfp16
    Tbf16
    Tfp32
    Tfp64
end

is_floating(x::ScalarTritonType) = @match x begin
    Tfp8e5 => true
    Tfp8e4 => true
    Tfp8e4b15 => true
    Tfp16 => true
    Tbf16 => true
    Tfp32 => true
    Tfp64 => true
    _ => false
end

is_integer(x::ScalarTritonType) = @match x begin
    Tint1 => true
    Tint8 => true
    Tuint8 => true
    Tint16 => true
    Tuint16 => true
    Tint32 => true
    Tuint32 => true
    Tint64 => true
    Tuint64 => true
    _ => false
end

construct_ir_type(builder, t::TritonType)::CT.MLIRTypeAllocated = @match t begin
    Tvoid => CppTriton.get_void_ty(builder)
    Tint1 => CppTriton.get_int1_ty(builder)
    Tint8 => CppTriton.get_int8_ty(builder)
    Tuint8 => CppTriton.get_int8_ty(builder)
    Tint16 => CppTriton.get_int16_ty(builder)
    Tuint16 => CppTriton.get_int16_ty(builder)
    Tint32 => CppTriton.get_int32_ty(builder)
    Tuint32 => CppTriton.get_int32_ty(builder)
    Tint64 => CppTriton.get_int64_ty(builder)
    Tuint64 => CppTriton.get_int64_ty(builder)
    Tfp8e5 => CppTriton.get_fp8e5_ty(builder)
    Tfp8e4 => CppTriton.get_fp8e4_ty(builder)
    Tfp8e4b15 => CppTriton.get_fp8e4b15_ty(builder)
    Tfp16 => CppTriton.get_half_ty(builder)
    Tbf16 => CppTriton.get_bf16_ty(builder)
    Tfp32 => CppTriton.get_float_ty(builder)
    Tfp64 => CppTriton.get_double_ty(builder)
end
scalar_type(x::ScalarTritonType) = x
base_scalar_type(x::ScalarTritonType) = x
is_scalar(x::ScalarTritonType) = true

@test @wcok construct_ir_type(builder, Tvoid)

@struct_hash_equal struct PointerTritonType <: TritonType
    scalar::ScalarTritonType
    #address_space::Int
end
@as_record PointerTritonType
construct_ir_type(builder, t::PointerTritonType)::CT.MLIRTypeAllocated =
    # hardcode address space (third param) as 1
    CppTriton.get_ptr_ty(builder, construct_ir_type(builder, t.scalar), 1)
scalar_type(x::PointerTritonType) = x
base_scalar_type(x::PointerTritonType) = x.scalar
is_pointer(x::PointerTritonType) = true
is_floating(x::PointerTritonType) = false
is_integer(x::PointerTritonType) = false


TRITON_MAX_TENSOR_NUMEL = 131072
@struct_hash_equal struct BlockTritonType <: TritonType
    scalar::Union{ScalarTritonType, PointerTritonType}
    dims::Vector{Int64}
    numel
end
@as_record BlockTritonType
function BlockTritonType(scalar, dims)
    numel = prod(dims)
    @assert numel <= TRITON_MAX_TENSOR_NUMEL "Tensor size exceeds Triton's limit"
    BlockTritonType(scalar, dims, numel)
end

construct_ir_type(builder, t::BlockTritonType)::CT.MLIRTypeAllocated =
    CppTriton.get_block_ty(builder, construct_ir_type(builder, t.scalar), collect(t.dims))
scalar_type(x::BlockTritonType) = x.scalar
base_scalar_type(x::BlockTritonType) = base_scalar_type(x.scalar)
is_block(x::BlockTritonType) = true
is_floating(x::BlockTritonType) = is_floating(x.scalar)
is_integer(x::BlockTritonType) = is_integer(x.scalar)
is_pointer(x::BlockTritonType) = is_pointer(x.scalar)


# more type helpers

is_fp8(x::TritonType) = is_fp8(base_scalar_type(x))
is_fp8(x::ScalarTritonType) = @match x begin
    Tfp8e5 => true
    Tfp8e4 => true
    Tfp8e4b15 => true
    _ => false
end

primitive_bandwidth(x::TritonType) = primitive_bandwidth(base_scalar_type(x))
primitive_bandwidth(x::ScalarTritonType) = @match x begin
    Tint1 => 1
    Tint8 => 8
    Tuint8 => 8
    Tint16 => 16
    Tuint16 => 16
    Tint32 => 32
    Tuint32 => 32
    Tint64 => 64
    Tuint64 => 64
    Tfp8e5 => 8
    Tfp8e4 => 8
    Tfp8e4b15 => 8
    Tfp16 => 16
    Tbf16 => 16
    Tfp32 => 32
    Tfp64 => 64
    _ => error("Not a primitive type")
end

fp_mantissa_width(x::TritonType) = fp_mantissa_width(base_scalar_type(x))
fp_mantissa_width(x::ScalarTritonType) = @match x begin
    Tfp8e5 => 2
    Tfp8e4 => 3
    Tfp8e4b15 => 3
    Tfp16 => 7
    Tbf16 => 10
    Tfp32 => 23
    Tfp64 => 53
    _ => error("Not a floating point type")
end

is_integer(x::TritonType) = is_integer(base_scalar_type(x))
is_integer(x::ScalarTritonType) = @match x begin
    Tint1 => true
    Tint8 => true
    Tuint8 => true
    Tint16 => true
    Tuint16 => true
    Tint32 => true
    Tuint32 => true
    Tint64 => true
    Tuint64 => true
    _ => false
end

is_signed(x::TritonType) = is_signed(base_scalar_type(x))
is_signed(x::ScalarTritonType) = @match x begin
    Tint1 => true
    Tint8 => true
    Tint16 => true
    Tint32 => true
    Tint64 => true
    Tuint8 => false
    Tuint16 => false
    Tuint32 => false
    Tuint64 => false
    _ => error("Not an integer type")
end

is_standard_floating(x::TritonType) = is_standard_floating(base_scalar_type(x))
is_standard_floating(x::ScalarTritonType) = @match x begin
    Tfp16 => true
    Tbf16 => true
    Tfp32 => true
    Tfp64 => true
    _ => false
end

is_bool(x::TritonType) = is_bool(base_scalar_type(x))
is_bool(x::ScalarTritonType) = x == Tint1

numel(x::TritonType) = 1
numel(x::BlockTritonType) = x.numel

Base.size(x::TritonType) = Int64[]
Base.size(x::BlockTritonType) = x.dims
Base.size(x::BlockTritonType, dim) = x.dims[dim]



# TODO
# t1 = PointerTritonType(Tint8)
# bt = construct_ir_type(builder, t1)
# CT.repr(CT.CxxRef(bt))

# CT.get_type(Tensor(builder, 5.0).handle) |> x -> CT.repr(CT.CxxRef(x))

# Conversion from MLIR types to Julia types. Unfortunately we can't do this losslessly because MLIR doesn't handle unsigned integers
# (should we maybe just drop unsigned integers?)

const TypeMLIRToJuliaNameLookup = let
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    Dict([CT.repr(CT.CxxRef(construct_ir_type(builder, x()))) => x for x in subtypes(ScalarTritonType)])
end

_parse_type_from_ptrscalar(x::AbstractString) = begin
    if startswith(x, "!tt.ptr<")
        inner = x[9:end-1]
        PointerTritonType(TypeMLIRToJuliaNameLookup[inner]())
    else
        TypeMLIRToJuliaNameLookup[x]()
    end
end

_parse_type_from_repr(x::AbstractString) = begin
    if startswith(x, "tensor<")
        notensor = x[8:end-1]
        parts = split(String(notensor), 'x')
        if length(parts) == 1
            # a 0-dim tensor
            BlockTritonType(_parse_type_from_ptrscalar(parts[1]), Int64[])
        else
            dims = parse.(Int64, parts[1:end-1])
            BlockTritonType(_parse_type_from_ptrscalar(parts[end]), dims)
        end
    else
        _parse_type_from_ptrscalar(x)
    end
end

##

# BlockTritonType(PointerTritonType(Tbf16), Int64[], 1) == BlockTritonType(PointerTritonType(Tbf16), Int64[], 1) 

# res = construct_ir_type(builder, BlockTritonType(Tint64, Int64[1, 2, 3]))

# CT.repr(CT.CxxRef(res)) |> _parse_type_from_repr

let do_test(builder, T) = begin
        x = T()
        repr = CT.repr(CT.CxxRef(construct_ir_type(builder, x))) |> String
        back_to_julia = _parse_type_from_repr(repr)

        if !contains(Base.repr(back_to_julia), "uint")
            @assert x == back_to_julia "Type mismatch: $x != $back_to_julia"
        end
    end
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    for T in subtypes(ScalarTritonType)
        do_test(builder, T)
        do_test(builder, () -> PointerTritonType(T()))
        do_test(builder, () -> BlockTritonType(PointerTritonType(T()), Int64[]))
        do_test(builder, () -> BlockTritonType(PointerTritonType(T()), Int64[1, 2, 3]))
        # @show T
        # @show repr(T) == "Tvoid", Base.repr(T)
        T() == Tvoid || do_test(builder, () -> BlockTritonType(T(), Int64[1, 2, 3]))
    end
end

    
change_scalar_type(x::ScalarTritonType, y::ScalarTritonType) = y
change_scalar_type(x::PointerTritonType, y::ScalarTritonType) = PointerTritonType(y)
change_scalar_type(x::BlockTritonType, y::ScalarTritonType) = BlockTritonType(y, x.dims)

