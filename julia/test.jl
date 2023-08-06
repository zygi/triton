

function test(a)
    b = zero(eltype(a))
    for i in 1:a
        b += i
    end
    return b
end

@code_lowered test(10)