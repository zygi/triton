using Test
# Super dumb test to make sure I'm not mistyping any c++ names

check_file(file) = begin
    file_contents = read(file, String)
    # find all occurrences of regex
    regex = r"CT\.([a-zA-Z0-9_!]+)\("
    all = getindex.(eachmatch(regex, file_contents), 1)


    regex2 = r"CppTriton\.([a-zA-Z0-9_!]+)\("
    all = [all; getindex.(eachmatch(regex2, file_contents), 1)]

    ms = names(CT, all=true, imported=true) |> Set

    filter_exceptions = filter(all) do x; x != "StdVector" && x != "CxxRef" end

    for method in filter_exceptions
        name = Symbol(method)
        @assert name in ms "Method $name not found in CT"
    end 
end

# BASE_FOLDER = dirname(dirname(pathof(Triton)))

# TODO fix path
# @test begin check_file("julia/helpers.jl"); true end
# @test begin check_file("julia/tensor_ops.jl"); true end
@test begin check_file("julia/typed_types.jl"); true end
@test begin check_file("julia/ops.jl"); true end


#methodswith(RegexMatch)