test_kernel(; in_ptr, out_ptr, n, extra_increment) = begin
    pid = program_id(1)
    my_ptr = in_ptr + pid

    accum = zero(TrVal{Int32})
    accum2 = zero(TrVal{Int32})
    (final_accum, accum2) =
        triton_for!(Int32(0), Int32(5), Int32(1), accum, accum2) do i, accum, accum2
            in = load(my_ptr; mask = TrVal(true), other = TrVal(0.0f0))
            return (accum + i + TrVal(extra_increment), accum2) #+ TrVal(bd, EXTRA_INCREMENT) #+ cast(in, Tint64)
        end

    store(out_ptr + pid, cast(final_accum, Float32))
    triton_return()
end

arg_types = OrderedDict([
    :in_ptr => Ptr{Float32},
    :out_ptr => Ptr{Float32},
    :n => Int32,
])
template_vals = OrderedDict([:extra_increment => Int32(1)])

@testset "Simple kernel compiles and runs" begin
    test_a = CUDA.ones(Float32, 64)
    test_out = CUDA.zeros(Float32, 64)
    kernel = compile_triton_kernel(
        test_kernel,
        arg_types,
        template_vals,
        (_, _) -> prod(size(test_a)),
    )
    kernel(test_a, test_out, 64)
    @test test_out ≈ 15 .* test_a
end
