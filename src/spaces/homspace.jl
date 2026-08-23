

struct HomSpace{P1<:ProductSpace,P2<:ProductSpace}
    codomain::P1
    domain::P2
end
codomain(W::HomSpace) = W.codomain
domain(W::HomSpace) = W.domain

dual(W::HomSpace) = HomSpace(dual(W.domain), dual(W.codomain))
function Base.adjoint(W::HomSpace)
    return HomSpace(W.domain, W.codomain)
end

Base.hash(W::HomSpace, h::UInt) = hash(domain(W), hash(codomain(W), h))
function Base.:(==)(W₁::HomSpace, W₂::HomSpace)
    return (W₁.codomain == W₂.codomain) && (W₁.domain == W₂.domain)
end

spacetype(W::HomSpace) = GradedSpace
sectortype(W::HomSpace) = FermionParity
spacetype(::Type{<:HomSpace}) = GradedSpace
sectortype(::Type{<:HomSpace}) = FermionParity

numout(W::HomSpace) = length(codomain(W))
numin(W::HomSpace) = length(domain(W))
numind(W::HomSpace) = numin(W) + numout(W)

const TensorSpace = Union{GradedSpace,ProductSpace}
const TensorMapSpace{N₁,N₂} = HomSpace{ProductSpace{N₁},ProductSpace{N₂}}

function Base.getindex(W::TensorMapSpace{N₁,N₂}, i) where {N₁,N₂}
    return i <= N₁ ? codomain(W)[i] : dual(domain(W)[i - N₁])
end

function ←(codom::ProductSpace, dom::ProductSpace)
    return HomSpace(codom, dom)
end
function ←(codom::GradedSpace, dom::GradedSpace)
    return HomSpace(ProductSpace(codom), ProductSpace(dom))
end
function ←(codom::Union{GradedSpace,ProductSpace}, dom::Union{GradedSpace,ProductSpace})
    return ←(promote(codom, dom)...)
end
→(dom::Union{GradedSpace,ProductSpace}, codom::Union{GradedSpace,ProductSpace}) = ←(codom, dom)


function blocksectors(W::HomSpace)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    if N₂ <= N₁
        return sort!(filter!(c -> hasblock(codom, c), blocksectors(dom)))
    else
        return sort!(filter!(c -> hasblock(dom, c), blocksectors(codom)))
    end
end

hasblock(W::HomSpace, c::FermionParity) = hasblock(codomain(W), c) && hasblock(domain(W), c)

function dim(W::HomSpace)
    d = 0
    for c in blocksectors(W)
        d += blockdim(codomain(W), c) * blockdim(domain(W), c)
    end
    return d
end

# Operations on HomSpaces
# -----------------------

function permute(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {N₁,N₂}
    p = (p₁..., p₂...)
    TupleTools.isperm(p) && length(p) == numind(W) ||
        throw(ArgumentError("$((p₁, p₂)) is not a valid permutation for $(W)"))
    return select(W, (p₁, p₂))
end

function select(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {N₁,N₂}
    cod = ProductSpace{N₁}(map(n -> W[n], p₁))
    dom = ProductSpace{N₂}(map(n -> dual(W[n]), p₂))
    return cod ← dom
end

function compose(W::HomSpace, V::HomSpace)
    domain(W) == codomain(V) || throw(SpaceMismatch("$(domain(W)) ≠ $(codomain(V))"))
    return HomSpace(codomain(W), domain(V))
end

# Block and fusion tree ranges: structure information for building tensors
#--------------------------------------------------------------------------
struct FusionBlockStructure{N,F₁,F₂}
    totaldim::Int
    blockstructure::SectorDict{FermionParity,Tuple{Tuple{Int,Int},UnitRange{Int}}}
    fusiontreelist::Vector{Tuple{F₁,F₂}}
    fusiontreestructure::Vector{Tuple{NTuple{N,Int},NTuple{N,Int},Int}}
    fusiontreeindices::FusionTreeDict{Tuple{F₁,F₂},Int}
end

# 对 Z₂ 对称性该结构很小，直接计算，不使用 cache
function fusionblockstructure(W::HomSpace)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    F₁ = fusiontreetype(N₁)
    F₂ = fusiontreetype(N₂)

    # output structure
    blockstructure = SectorDict{FermionParity,Tuple{Tuple{Int,Int},UnitRange{Int}}}() # size, range
    fusiontreelist = Vector{Tuple{F₁,F₂}}()
    fusiontreestructure = Vector{Tuple{NTuple{N₁ + N₂,Int},NTuple{N₁ + N₂,Int},Int}}() # size, strides, offset

    # temporary data structures
    splittingtrees = Vector{F₁}()
    splittingstructure = Vector{Tuple{Int,Int}}()

    # main computational routine
    blockoffset = 0
    for c in blocksectors(W)
        empty!(splittingtrees)
        empty!(splittingstructure)

        offset₁ = 0
        for f₁ in fusiontrees(codom, c)
            push!(splittingtrees, f₁)
            d₁ = dim(codom, f₁.uncoupled)
            push!(splittingstructure, (offset₁, d₁))
            offset₁ += d₁
        end
        blockdim₁ = offset₁
        strides = (1, blockdim₁)

        offset₂ = 0
        for f₂ in fusiontrees(dom, c)
            s₂ = f₂.uncoupled
            d₂ = dim(dom, s₂)
            for (f₁, (offset₁, d₁)) in zip(splittingtrees, splittingstructure)
                push!(fusiontreelist, (f₁, f₂))
                totaloffset = blockoffset + offset₂ * blockdim₁ + offset₁
                subsz = (dims(codom, f₁.uncoupled)..., dims(dom, f₂.uncoupled)...)
                @assert !any(isequal(0), subsz)
                substr = _subblock_strides(subsz, (d₁, d₂), strides)
                push!(fusiontreestructure, (subsz, substr, totaloffset))
            end
            offset₂ += d₂
        end
        blockdim₂ = offset₂
        blocksize = (blockdim₁, blockdim₂)
        blocklength = blockdim₁ * blockdim₂
        blockrange = (blockoffset + 1):(blockoffset + blocklength)
        blockoffset = last(blockrange)
        blockstructure[c] = (blocksize, blockrange)
    end

    fusiontreeindices = sizehint!(FusionTreeDict{Tuple{F₁,F₂},Int}(),
                                  length(fusiontreelist))
    for (i, f₁₂) in enumerate(fusiontreelist)
        fusiontreeindices[f₁₂] = i
    end
    totaldim = blockoffset
    structure = FusionBlockStructure(totaldim, blockstructure,
                                     fusiontreelist, fusiontreestructure,
                                     fusiontreeindices)
    return structure
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    return Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
end

# Diagonal ranges
#----------------
# TODO: is this something we want to cache?
function diagonalblockstructure(W::HomSpace)
    ((numin(W) == numout(W) == 1) && domain(W) == codomain(W)) ||
        throw(SpaceMismatch("Diagonal only support on V←V with a single space V"))
    structure = SectorDict{FermionParity,UnitRange{Int}}() # range
    offset = 0
    dom = domain(W)[1]
    for c in blocksectors(W)
        d = dim(dom, c)
        structure[c] = offset .+ (1:d)
        offset += d
    end
    return structure
end
