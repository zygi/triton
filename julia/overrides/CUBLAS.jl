# Override cublas to do FP16 matmuls using FP32 compute type unless math_precision() is set to a lower value

CUDA.CUBLAS.gemmExComputeType(TA, TB, TC, m, k, n) = begin
    if TA !== TB
        return nothing
    end
    sig = (TA, TC)
    # gemmEx requires sm_50 or higher
    cap = capability(device())
    if cap < v"5"
        return nothing
    end

    # source: CUBLAS Features and Technical Specifications
    if Float16 in sig && cap < v"5.3"
        return nothing
    end

    math_mode = CUDA.math_mode()
    reduced_precision = CUDA.math_precision()

    if sig === (Float16, Float16)
        # NOTE: Float16=Float16*Float16 can also happen in 32-bit compute
        if reduced_precision ===  :TensorFloat32
            return math_mode==CUDA.PEDANTIC_MATH ? CUDA.CUBLAS.CUBLAS_COMPUTE_32F_PEDANTIC : CUDA.CUBLAS.CUBLAS_COMPUTE_32F
        else
            return math_mode==CUDA.PEDANTIC_MATH ? CUDA.CUBLAS.CUBLAS_COMPUTE_16F_PEDANTIC : CUDA.CUBLAS.CUBLAS_COMPUTE_16F
        end
    end

    if sig === (Int8, Int32)
        # starting with CUDA 11.2, this is unsupported (NVIDIA bug #3221266)
        # TODO: might be fixed in a later version?
        version() >= v"11.3.1" && return nothing

        # Int32=Int8*Int8 requires m,n,k to be multiples of 4
        # https://forums.developer.nvidia.com/t/cublasgemmex-cant-use-cuda-r-8i-compute-type-on-gtx1080/58100/2
        if m%4 == 0 && n%4 == 0 && k%4 == 0
            return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I
        end
    end

    if math_mode == CUDA.FAST_MATH
        if sig === (Float32, Float32) ||
           sig === (Complex{Float32}, Complex{Float32})
            if reduced_precision === :Float16
                return CUBLAS_COMPUTE_32F_FAST_16F
            elseif reduced_precision === :BFloat16
                return CUBLAS_COMPUTE_32F_FAST_16BF
            elseif reduced_precision === :TensorFloat32
                return CUBLAS_COMPUTE_32F_FAST_TF32
            else
                throw(ArgumentError("Unknown reduced precision type $reduced_precision"))
            end
        end
    end

    if sig === (Float16,  Float16) ||
       sig === (Int8,     Float32) ||
       sig === (Float16,  Float32) ||
       sig === (Float32,  Float32) ||
       sig === (Complex{Int8},    Complex{Float32}) ||
       sig === (Complex{Float32}, Complex{Float32})
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F
    end

    if sig === (Float64, Float64) ||
       sig === (Complex{Float64}, Complex{Float64})
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F
    end

    # BFloat16 support was added in CUDA 11
    if version() >= v"11"
        if sig === (BFloat16, BFloat16) ||
           sig === (BFloat16, Float32)
            return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F
        end
    end

    return nothing
end