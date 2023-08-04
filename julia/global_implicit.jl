const _global_T = CT.TritonOpBuilder
const _ref = Ref{Union{<:_global_T, Nothing}}(nothing)
get_builder_ref() = begin
    @assert !isnothing(_ref[]) "The global implicit $(_global_T) not initialized. Consider calling with `with_scoped`."
    _ref[]
end
set_builder_ref(val) = begin _ref[] = val end

with_scoped(fn, x::U) where {U <: _global_T} = begin
    old = _ref[]
    set_builder_ref(x)
    try
        fn()
    finally
        set_builder_ref(old)
    end
end

macro declare_alias_using_scoped(fn)
    @assert fn isa Symbol
    quote
        function $(fn)(args...)
            $(fn)(get_builder_ref(), args...)
        end
    end
end
