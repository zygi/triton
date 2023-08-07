test_kernel(; in_ptr::Tensor, out_ptr::Tensor, n::Tensor, extra_increment::Int32) = begin
    pid = program_id(1)
    my_ptr = in_ptr + pid
    
    accum = zero(TrInt32)
    accum2 = zero(TrInt32)
    (final_accum, accum2) = triton_for!(Int32(0), Int32(5), Int32(1), accum, accum2) do i, accum, accum2
        in = load(my_ptr; mask=Tensor(true), other=Tensor(0.0f0))
        return (accum + i + Tensor(extra_increment), accum2) #+ Tensor(bd, EXTRA_INCREMENT) #+ cast(in, Tint64)
    end

    store(out_ptr + pid, cast(final_accum, TrFloat32))
    triton_return()
end

arg_types = OrderedDict([:in_ptr => TritonPointerType{TrFloat32}, :out_ptr => TritonPointerType{TrFloat32}, :n => TrInt32])
template_vals = OrderedDict([:extra_increment => Int32(1)])

@test begin
    test_a = CUDA.ones(Float32, 64)
    test_out = CUDA.zeros(Float32, 64)
    kernel = compile_triton_kernel(test_kernel, arg_types, template_vals, (_, _) -> prod(size(test_a)))
    kernel(test_a, test_out, 64)
    test_out ≈ 15 .* test_a
end
