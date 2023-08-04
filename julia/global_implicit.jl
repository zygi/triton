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


macro with_scoped_builder(fn)
    jlfn = JLFunction(fn)
    @assert length(jlfn.args) >= 1 "Function must have at least one argument"
    @assert jlfn.args[1] == :builder "First argument must be builder"

    jlfn.args = jlfn.args[2:end]
    jlfn.body = quote
        builder = get_builder_ref()
        $(jlfn.body)
    end

    quote
        $(esc(fn))

        $(esc(codegen_ast(jlfn)))
    end
end
