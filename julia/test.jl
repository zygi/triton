module CppTriton
  using CxxWrap
  @wrapmodule(joinpath("build","libtriton_julia"))

  function __init__()
    @initcxx
  end
end

using MLStyle
using IRTools
using Cassette
using Test

ctx = CppTriton.MLIRContext()

CppTriton.load_triton!(ctx)

# CppTriton.CxxWrap.CxxPtr(ctx)

builder = CppTriton.TritonOpBuilder(CppTriton.CxxWrap.CxxPtr(ctx))

mod = CppTriton.create_module(builder)

# def to_ir(self, builder: ir.builder) -> ir.type:
# if self.name == 'void':
#     return builder.get_void_ty()
# elif self.name == 'int1':
#     return builder.get_int1_ty()
# elif self.name in ('int8', 'uint8'):
#     return builder.get_int8_ty()
# elif self.name in ('int16', 'uint16'):
#     return builder.get_int16_ty()
# elif self.name in ('int32', 'uint32'):
#     return builder.get_int32_ty()
# elif self.name in ('int64', 'uint64'):
#     return builder.get_int64_ty()
# elif self.name == 'fp8e5':
#     return builder.get_fp8e5_ty()
# elif self.name == 'fp8e4':
#     return builder.get_fp8e4_ty()
# elif self.name == 'fp8e4b15':
#     return builder.get_fp8e4b15_ty()
# elif self.name == 'fp16':
#     return builder.get_half_ty()
# elif self.name == 'bf16':
#     return builder.get_bf16_ty()
# elif self.name == 'fp32':
#     return builder.get_float_ty()
# elif self.name == 'fp64':
#     return builder.get_double_ty()
# raise ValueError(f'fail to convert {self} to ir type')

is_scalar(x) = false
is_block(x) = false
is_pointer(x) = false

abstract type TritonType end
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

is_int(x::ScalarTritonType) = @match x begin
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

construct_ir_type(builder, t::TritonType) = @match t begin
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
is_scalar(x::ScalarTritonType) = true




TRITON_MAX_TENSOR_NUMEL = 131072
struct BlockTritonType
    scalar::ScalarTritonType
    dims::Tuple
    numel
end
function BlockTritonType(scalar::ScalarTritonType, dims)
    numel = 1
    for d in dims
        numel *= d
    end
    @assert numel <= TRITON_MAX_TENSOR_NUMEL "Tensor size exceeds Triton's limit"
    BlockTritonType(scalar, dims, numel)
end

construct_ir_type(builder, t::BlockTritonType) =
    CppTriton.get_block_ty(builder, construct_ir_type(builder, t.scalar), collect(t.dims))
scalar_type(x::BlockTritonType) = x.scalar
is_block(x::BlockTritonType) = true
is_floating(x::BlockTritonType) = is_floating(x.scalar)
is_int(x::BlockTritonType) = is_int(x.scalar)




struct PointerTritonType
    scalar::ScalarTritonType
    #address_space::Int
end
construct_ir_type(builder, t::PointerTritonType) =
    CppTriton.get_ptr_ty(builder, construct_ir_type(builder, t.scalar))
scalar_type(x::PointerTritonType) = x.scalar
is_pointer(x::PointerTritonType) = true
is_floating(x::PointerTritonType) = is_floating(x.scalar)
is_int(x::PointerTritonType) = is_int(x.scalar)


builder

# abstract type Tensor end
struct Tensor{T <: TritonType}
    builder
    handle
    type::T
end
Tensor(builder, b::Bool) = Tensor(builder, CppTriton.get_int1(builder, b), Tint1)
Tensor(builder, x::T) where {T <: Int} = begin
    if -2^31 <= x < 2^31
        return Tensor(builder, CppTriton.get_int32(builder, x), Tint32)
    elseif 2^31 <= x < 2^32
        return Tensor(builder, CppTriton.get_int32(builder, x), Tuint32)
    elseif -2^63 <= x < 2^63
        return Tensor(builder, CppTriton.get_int64(builder, x), Tint64)
    elseif 2^63 <= x < 2^64
        return Tensor(builder, CppTriton.get_int64(builder, x), Tuint64)
    else
        error("Nonrepresentable integer $x.")
    end
end
Tensor(builder, x::Float32) = Tensor(builder, CppTriton.get_fp32(builder, x), Tfp32)
Tensor(builder, x::Float64) = Tensor(builder, CppTriton.get_fp64(builder, x), Tfp64)
Tensor(builder, handle) = Tensor(builder, handle, TritonType(CppTriton.get_type(handle)))

is_floating(x::Tensor) = is_floating(x.type)
is_int(x::Tensor) = is_int(x.type)

@test @wc Tensor(builder, true) !== C_NULL

# leave type checking to mlir for now

Base.:-(x::Tensor) = begin
    @assert !is_pointer(x.type) "Cannot negate a pointer"
    zeroval = Tensor(x.builder, CppTriton.get_null_value(x.builder, construct_ir_type(x.builder, x.type)), x.type)
    zeroval - x
end

Base.:+(x::Tensor, y::Tensor) = @match (is_pointer(x.type), is_pointer(y.type)) begin
    (true, _) => Tensor(x.builder, CppTriton.create_addptr(x.builder, x.handle, y.handle), scalar_type(x.type))
    (false, true) => Tensor(x.builder, CppTriton.create_addptr(x.builder, y.handle, x.handle), scalar_type(y.type))
    _ => begin
        if is_floating(x) && is_floating(y)
            Tensor(x.builder, CppTriton.create_fadd(x.builder, x.handle, y.handle), scalar_type(x.type))
        elseif is_int(x) && is_int(y)
            Tensor(x.builder, CppTriton.create_add(x.builder, x.handle, y.handle), scalar_type(x.type))
        else
            error("Cannot add $x and $y")
        end
    end
end

Base.:-(x::Tensor, y::Tensor) = @match (is_pointer(x.type), is_pointer(y.type)) begin
    (true, _) => Tensor(x.builder, CppTriton.create_subptr(x.builder, x.handle, y.handle), scalar_type(x.type))
    (false, true) => Tensor(x.builder, CppTriton.create_addptr(x.builder, (-y).handle, x.handle), scalar_type(y.type))
    _ => begin
        if is_floating(x) && is_floating(y)
            Tensor(x.builder, CppTriton.create_fsub(builder, x.handle, y.handle), scalar_type(x.type))
        elseif is_int(x) && is_int(y)
            Tensor(x.builder, CppTriton.create_sub(builder, x.handle, y.handle), scalar_type(x.type))
        else
            error("Cannot subtract $x and $y")
        end
    end
end

Base.:*(x::Tensor, y::Tensor) = begin
    if is_floating(x) && is_floating(y)
        Tensor(x.builder, CppTriton.create_fmul(builder, x.handle, y.handle), scalar_type(x.type))
    elseif is_int(x) && is_int(y)
        Tensor(x.builder, CppTriton.create_mul(builder, x.handle, y.handle), scalar_type(x.type))
    else
        error("Cannot multiply $x and $y")
    end
end

typecast(x::Tensor, dst_t::TritonType) = begin
    src_t = x.type
    @match src_t begin
        BlockTritonType(t, dims) => begin dst_t = BlockTritonType(dst_t, dims) end
    end

    (src_t == dst_t) && return x

    

end
    Tensor(x.builder, CppTriton.create_cast(x.builder, x.handle, construct_ir_type(x.builder, t)), t)




a = rand(Float16, 3, 4)
b = rand(Float64, 3, 4)
a + b

# CppTriton.get_type(t1.handle)
# t1.type

t1 = Tensor(builder, 1)
t2 = Tensor(builder, 2)
t3 = Tensor(builder, 3.0)


t1 + t2
-t1
t1 - t2

t1 * t2

t3 * t2


##


# CppTriton.get_int1_ty
##

# construct_ir_type(void)

:(::Val{c}).args

@match :(::Val{c}) begin
    Expr(:(::), Expr(:curly, :Val, x)) => x
end

:(add(a, b, ::Val{c}) where c)


CppTriton.get_int1(builder, false)

get_tensor(builder, x::Bool) = CppTriton.get_int1(builder, x)
get_tensor(builder, x::Int32

macro test(ex)
    process_signature(s) = @match s begin
        Expr(:where, sign, generic_params...) => begin
            # @show sign.head
            @assert sign.head == :call
            
            fn_name = sign.args[1]

            get_val_arg_symbol(e) = @match e begin
                Expr(:(::), Expr(:curly, :Val, x)) => x
                _ => nothing
            end
            fn_args = sign.args[2:end]
            var_args = [x for x in fn_args if get_val_arg_symbol(x) === nothing]
            val_args = [get_val_arg_symbol(x) for x in fn_args if get_val_arg_symbol(x) !== nothing]

            # @show val_args
            # @show generic_params
            @assert sort(val_args) == sort(generic_params) "All type parameters must be ::Val{} arguments"

            return var_args, generic_params
        end
    end 

    process_block_entry(e) =
        @match e begin
            ::LineNumberNode => nothing
            ::Symbol => nothing
            Expr(:+=, lhs, rhs) => begin
                @show lhs, rhs
            end
            line => @show line.head
        end

    process_function(e) = @match e begin
        Expr(:block, args...) => throw("err1")
        Expr(:function, sig, Expr(:block, block_entries...)) => begin
            # @show sig.head
            args, template_args = process_signature(sig)
            template_set = Set(template_args)
            process_block_entry.(block_entries)
        end 

        # a::LineNumberNode => nothing
        a => begin println(a); a end
    end
    process_function(ex)
    # @show ex.head, ex.args[1]
    nothing
end

@test @wc function add(a, b, ::Val{c}) where {c}
    a += 1
    a = a + 1
    for i in 1:c
        a += b
    end
    a
    # a + b + c
end

test_fn.args[2]
# test_fn |> typeof



process_expr(e) = @match e begin
    Expr(:bleock, args...) => process_expr.(args) 
    # a::LineNumberNode => nothing
    a => begin println(a); a end
end

process_expr(test_fn)

# fieldnames(typeof(test_fn))


function add2(a, b, ::Val{c}) where c
    for i in 1:5
        a += b + c
    end
    a
    # a + b + c
end





# tl_program_id(; axis) :: Int = begin 
#     0
# end

# tl_load(ptr; mask) = 0
# tl_store(ptr, dest; mask) = nothing

function add_kernel(x_ptr, y_ptr, output_ptr, n_elements, ::Val{BLOCK_SIZE}) where BLOCK_SIZE
    pid = tl_program_id(0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl_arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl_load_mask(x_ptr + offsets, mask)
    y = tl_load_mask(y_ptr + offsets, mask)
    output = x + y
    # Write x + y back to DRAM.
    tl_store_mask(output_ptr + offsets, output, mask)
end



using SymbolicUtils
@syms a::NTuple{3, Int} tl_load(ptrs) tl_load_mask(ptrs, mask) tl_store(ptrs, values)  tl_store_mask(ptrs, values, mask)
@syms tl_program_id(axis::Int)::Int tl_arange(start, endd)::Int

# SymbolicUtils.symtype(a)

SymbolicUtils.promote_symtype(::Type{isless}, ::Type{Number}, ::Type{Number}) = Bool
# SymbolicUtils.promote_symtype(isless, ::Type{Number}, ::Type{Number}) = Bool


add_kernel(a, a, a, a, Val(10))

## asdf


@code_typed add_kernel(0, 100, 200, 10, Val(5))


typed = @code_typed add2(1, 2, Val(5))



# IRTools.IR(@code_typed add2(1, 2, Val(5))

typed |> collect

# get temp file
tmp = tempname()

Cassette.@context Ctx;

Cassette.prehook(::Ctx{Val{T}}, f, arg::T, rest...) where {T} = println(f, (arg, rest...))

Cassette.overdub(Ctx(metadata=Val(Int)), () -> add2(1, 2, Val(5)))

typeof(sin)



Cassette.@context SinToCosCtx

# Override the default recursive `overdub` implementation for `sin(x)`.
# Note that there's no tricks here; this is just a normal Julia method
# overload using the normal multiple dispatch semantics.
Cassette.overdub(::SinToCosCtx, ::typeof(sin), x) = cos(x)

x = rand(10)
y = Cassette.overdub(SinToCosCtx(), sum, i -> cos(i) + sin(i), x)
@test @wc y == sum(i -> 2 * cos(i), x)