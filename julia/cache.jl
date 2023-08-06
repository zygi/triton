using Test
using Serialization

abstract type AbstractCacheManager{T} <: AbstractDict{String, T} end
struct FileSystemCacheManager{T} <: AbstractCacheManager{T}
    prefix::String
end
FileSystemCacheManager(prefix::String, ::Type{T}) where T = begin
    isdir(prefix) || mkdir(prefix)
    FileSystemCacheManager{T}(prefix)
end

Base.haskey(m::FileSystemCacheManager, k::String) = isfile(joinpath(m.prefix, k))
Base.get(m::FileSystemCacheManager{T}, k::String, default) where T = begin
    if haskey(m, k)
        Serialization.deserialize(joinpath(m.prefix, k))::T
    else
        default
    end
end
Base.setindex!(m::FileSystemCacheManager{T}, v::T, k::String) where T = begin
    Serialization.serialize(joinpath(m.prefix, k), v)
    @bp
end
Base.delete!(m::FileSystemCacheManager, k::String) = rm(joinpath(m.prefix, k))
Base.iterate(m::FileSystemCacheManager) = begin
    files = readdir(m.prefix; sort=true)
    if isempty(files)
        nothing
    else
        (files[1] => m[files[1]], files[2:end])
    end
end
Base.iterate(m::FileSystemCacheManager, s::Vector{String}) = if isempty(s) nothing else (s[1] => m[s[1]], s[2:end]) end
Base.keys(m::FileSystemCacheManager) = readdir(m.prefix)
Base.length(m::FileSystemCacheManager) = length(keys(m))
# resist the urge to wipe the cache. This will only lead to accidental disaster.
# wipe!(m::FileSystemCacheManager) = rm(m.prefix; recursive=true) 

@testset begin
    isdir("/tmp/tritontest") && rm("/tmp/tritontest"; recursive=true)
    m = FileSystemCacheManager("/tmp/tritontest", Float64)
    m["foo"] = 1.0
    m["bar"] = 2.0
    @test m["foo"] == 1.0
    @test m["bar"] == 2.0
    @test !haskey(m, "baz")
    @test haskey(m, "foo")
    @test haskey(m, "bar")
    delete!(m, "foo")
    @test !haskey(m, "foo")
    @test_throws "" m["foo"]
    rm("/tmp/tritontest"; recursive=true)
end

struct Cache{K, V, M<:AbstractCacheManager{V}} <: AbstractDict{K, V}
    manager::M
    mapping_fn::Function
end
Cache(::Type{K}, manager::M, mapping_fn::Function) where {K, V, M<:AbstractCacheManager{V}} = begin
    # rts = Base.return_types(mapping_fn, (V,))
    @assert length(Base.return_types(mapping_fn, (K,))) == 1 "mapping_fn must take a single argument and return a single value"
    Cache{K, V, M}(manager, mapping_fn)
end
Base.haskey(c::Cache, k) = haskey(c.manager, c.mapping_fn(k))
Base.get(c::Cache, k, default) = get(c.manager, c.mapping_fn(k), default)
Base.setindex!(c::Cache, v, k) = setindex!(c.manager, v, c.mapping_fn(k))
Base.delete!(c::Cache, k) = delete!(c.manager, c.mapping_fn(k))
Base.length(c::Cache) = length(c.manager)
Base.values(c::Cache) = (v for (k, v) in c.manager)
Base.iterate(c::Cache, args...) = iterate(c.manager, args...)
