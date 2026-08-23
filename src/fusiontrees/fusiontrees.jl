
struct FusionTree{N}
    uncoupled::NTuple{N,FermionParity}
    coupled::FermionParity
    isdual::NTuple{N,Bool}
    function FusionTree{N}(uncoupled::NTuple{N,FermionParity}, coupled::FermionParity,
                           isdual::NTuple{N,Bool}=ntuple(_ -> false, N)) where {N}
        return new{N}(uncoupled, coupled, isdual)
    end
end

function FusionTree(uncoupled::NTuple{N,FermionParity},
                    coupled::FermionParity=unit(FermionParity),
                    isdual::NTuple{N,Bool}=ntuple(_ -> false, N)) where {N}
    return FusionTree{N}(map(s -> convert(FermionParity, s), uncoupled),
                         convert(FermionParity, coupled), isdual)
end

# Properties
sectortype(::Type{<:FusionTree}) = FermionParity
Base.length(::Type{<:FusionTree{N}}) where {N} = N

sectortype(f::FusionTree) = sectortype(typeof(f))
Base.length(f::FusionTree) = length(typeof(f))

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree, h::UInt)
    h = hash(f.coupled, hash(f.uncoupled, h))
    h = hash(f.isdual, h)
    return h
end
function Base.:(==)(f₁::FusionTree{N}, f₂::FusionTree{N}) where {N}
    f₁.coupled == f₂.coupled || return false
    f₁.isdual == f₂.isdual || return false
    @inbounds for i in 1:N
        f₁.uncoupled[i] == f₂.uncoupled[i] || return false
    end
    return true
end
Base.:(==)(f₁::FusionTree, f₂::FusionTree) = false

# Facilitate getting correct fusion tree types
function fusiontreetype(N::Int)
    return FusionTree{N}
end



# Manipulate fusion trees
include("manipulations.jl")

# Fusion tree iterators
include("iterator.jl")
