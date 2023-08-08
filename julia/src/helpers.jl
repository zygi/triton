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


abstract type TrTypeable end
is_scalar(x::TrTypeable) = false
is_block(x::TrTypeable) = false
is_pointer(x::TrTypeable) = false

@data ScalarTrTypeable <: TrTypeable begin
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

is_floating(x::ScalarTrTypeable) = @match x begin
    Tfp8e5 => true
    Tfp8e4 => true
    Tfp8e4b15 => true
    Tfp16 => true
    Tbf16 => true
    Tfp32 => true
    Tfp64 => true
    _ => false
end

is_integer(x::ScalarTrTypeable) = @match x begin
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

construct_ir_type(builder, t::TrTypeable)::CT.MLIRTypeAllocated = @match t begin
    Tvoid => CppTriton.get_void_ty!(builder)
    Tint1 => CppTriton.get_int1_ty!(builder)
    Tint8 => CppTriton.get_int8_ty!(builder)
    Tuint8 => CppTriton.get_int8_ty!(builder)
    Tint16 => CppTriton.get_int16_ty!(builder)
    Tuint16 => CppTriton.get_int16_ty!(builder)
    Tint32 => CppTriton.get_int32_ty!(builder)
    Tuint32 => CppTriton.get_int32_ty!(builder)
    Tint64 => CppTriton.get_int64_ty!(builder)
    Tuint64 => CppTriton.get_int64_ty!(builder)
    Tfp8e5 => CppTriton.get_fp8e5_ty!(builder)
    Tfp8e4 => CppTriton.get_fp8e4_ty!(builder)
    Tfp8e4b15 => CppTriton.get_fp8e4b15_ty!(builder)
    Tfp16 => CppTriton.get_half_ty!(builder)
    Tbf16 => CppTriton.get_bf16_ty!(builder)
    Tfp32 => CppTriton.get_float_ty!(builder)
    Tfp64 => CppTriton.get_double_ty!(builder)
end
scalar_type(x::ScalarTrTypeable) = x
base_scalar_type(x::ScalarTrTypeable) = x
is_scalar(x::ScalarTrTypeable) = true

@test @wcok construct_ir_type(builder, Tvoid)

@struct_hash_equal struct PointerTrTypeable <: TrTypeable
    scalar::ScalarTrTypeable
    #address_space::Int
end
@as_record PointerTrTypeable
construct_ir_type(builder, t::PointerTrTypeable)::CT.MLIRTypeAllocated =
    # hardcode address space (third param) as 1
    CppTriton.get_ptr_ty!(builder, construct_ir_type(builder, t.scalar), 1)
scalar_type(x::PointerTrTypeable) = x
base_scalar_type(x::PointerTrTypeable) = x.scalar
is_pointer(x::PointerTrTypeable) = true
is_floating(x::PointerTrTypeable) = false
is_integer(x::PointerTrTypeable) = false


TRITON_MAX_TENSOR_NUMEL = 131072
@struct_hash_equal struct BlockTrTypeable <: TrTypeable
    scalar::Union{ScalarTrTypeable, PointerTrTypeable}
    dims::Vector{Int64}
    numel
end
@as_record BlockTrTypeable
function BlockTrTypeable(scalar, dims)
    numel = prod(dims)
    @assert numel <= TRITON_MAX_TENSOR_NUMEL "Tensor size exceeds Triton's limit"
    BlockTrTypeable(scalar, dims, numel)
end

construct_ir_type(builder, t::BlockTrTypeable)::CT.MLIRTypeAllocated =
    CppTriton.get_block_ty!(builder, construct_ir_type(builder, t.scalar), collect(t.dims))
scalar_type(x::BlockTrTypeable) = x.scalar
base_scalar_type(x::BlockTrTypeable) = base_scalar_type(x.scalar)
is_block(x::BlockTrTypeable) = true
is_floating(x::BlockTrTypeable) = is_floating(x.scalar)
is_integer(x::BlockTrTypeable) = is_integer(x.scalar)
is_pointer(x::BlockTrTypeable) = is_pointer(x.scalar)


# more type helpers

is_fp8(x::TrTypeable) = is_fp8(base_scalar_type(x))
is_fp8(x::ScalarTrTypeable) = @match x begin
    Tfp8e5 => true
    Tfp8e4 => true
    Tfp8e4b15 => true
    _ => false
end

primitive_bandwidth(x::TrTypeable) = primitive_bandwidth(base_scalar_type(x))
primitive_bandwidth(x::ScalarTrTypeable) = @match x begin
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

fp_mantissa_width(x::TrTypeable) = fp_mantissa_width(base_scalar_type(x))
fp_mantissa_width(x::ScalarTrTypeable) = @match x begin
    Tfp8e5 => 2
    Tfp8e4 => 3
    Tfp8e4b15 => 3
    Tfp16 => 7
    Tbf16 => 10
    Tfp32 => 23
    Tfp64 => 53
    _ => error("Not a floating point type")
end

is_integer(x::TrTypeable) = is_integer(base_scalar_type(x))
is_integer(x::ScalarTrTypeable) = @match x begin
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

is_signed(x::TrTypeable) = is_signed(base_scalar_type(x))
is_signed(x::ScalarTrTypeable) = @match x begin
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

is_standard_floating(x::TrTypeable) = is_standard_floating(base_scalar_type(x))
is_standard_floating(x::ScalarTrTypeable) = @match x begin
    Tfp16 => true
    Tbf16 => true
    Tfp32 => true
    Tfp64 => true
    _ => false
end

is_bool(x::TrTypeable) = is_bool(base_scalar_type(x))
is_bool(x::ScalarTrTypeable) = x == Tint1

numel(x::TrTypeable) = 1
numel(x::BlockTrTypeable) = x.numel

Base.size(x::TrTypeable) = Int64[]
Base.size(x::BlockTrTypeable) = x.dims
Base.size(x::BlockTrTypeable, dim) = x.dims[dim]



# TODO
# t1 = PointerTrTypeable(Tint8)
# bt = construct_ir_type(builder, t1)
# CT.repr(CT.CxxRef(bt))

# CT.get_type(Tensor(builder, 5.0).handle) |> x -> CT.repr(CT.CxxRef(x))

# Conversion from MLIR types to Julia types. Unfortunately we can't do this losslessly because MLIR doesn't handle unsigned integers
# (should we maybe just drop unsigned integers?)

const TypeMLIRToJuliaNameLookup = let
    ctx = CppTriton.MLIRContext()
    CppTriton.load_triton!(ctx)
    builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))
    Dict([CT.repr(CT.CxxRef(construct_ir_type(builder, x()))) => x for x in subtypes(ScalarTrTypeable)])
end

_parse_type_from_ptrscalar(x::AbstractString) = begin
    if startswith(x, "!tt.ptr<")
        inner = x[9:end-1]
        PointerTrTypeable(TypeMLIRToJuliaNameLookup[inner]())
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
            BlockTrTypeable(_parse_type_from_ptrscalar(parts[1]), Int64[])
        else
            dims = parse.(Int64, parts[1:end-1])
            BlockTrTypeable(_parse_type_from_ptrscalar(parts[end]), dims)
        end
    else
        _parse_type_from_ptrscalar(x)
    end
end

##

# BlockTrTypeable(PointerTrTypeable(Tbf16), Int64[], 1) == BlockTrTypeable(PointerTrTypeable(Tbf16), Int64[], 1) 

# res = construct_ir_type(builder, BlockTrTypeable(Tint64, Int64[1, 2, 3]))

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
    for T in subtypes(ScalarTrTypeable)
        do_test(builder, T)
        do_test(builder, () -> PointerTrTypeable(T()))
        do_test(builder, () -> BlockTrTypeable(PointerTrTypeable(T()), Int64[]))
        do_test(builder, () -> BlockTrTypeable(PointerTrTypeable(T()), Int64[1, 2, 3]))
        # @show T
        # @show repr(T) == "Tvoid", Base.repr(T)
        T() == Tvoid || do_test(builder, () -> BlockTrTypeable(T(), Int64[1, 2, 3]))
    end
end

    
change_scalar_type(x::ScalarTrTypeable, y::ScalarTrTypeable) = y
change_scalar_type(x::PointerTrTypeable, y::ScalarTrTypeable) = PointerTrTypeable(y)
change_scalar_type(x::BlockTrTypeable, y::ScalarTrTypeable) = BlockTrTypeable(y, x.dims)

