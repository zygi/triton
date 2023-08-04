# If you know a better way to do this, lmk

# module CppTritonLoadingTracker
# if !@isdefined(_is_cpptriton_loaded)
#     const _is_cpptriton_loaded = Ref{Bool}(false)
# end
# # const _is_cpptriton_loaded = Ref{Bool}(false)

# # macro if_not_loaded(expr)
# #     if _is_cpptriton_loaded[]
# #         esc(:(nothing))
# #     else
# #         _is_cpptriton_loaded[] = true
# #         esc(expr)
# #     end
# # end

# # @macroexpand()

# # export _is_cpptriton_loaded
# end

module CppTriton
using CxxWrap
# using ..CppTritonLoadingTracker: _is_cpptriton_loaded #@if_not_loaded

# const _is_cpptriton_loaded = Ref{Bool}(false)
# ast = (@macroexpand(@wrapmodule(joinpath("build", "libtriton_julia"))))

# @show _is_cpptriton_loaded
# if !_is_cpptriton_loaded[]
#     _is_cpptriton_loaded[] = true
#     eval(ast)
# end
# @show _is_cpptriton_loaded

@wrapmodule(joinpath("build", "libtriton_julia"))
function __init__()
    # @show "WAWAWA"
    @initcxx
    # @show "WAWAWA2"
end
end