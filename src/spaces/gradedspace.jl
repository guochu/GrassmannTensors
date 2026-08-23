

# 唯一确定的（分次）向量空间：dims 固定为 NTuple{2,Int}，下标 1/2 分别对应
# even/odd 两个 FermionParity sector。
struct GradedSpace
    dims::NTuple{2,Int}
    dual::Bool
end

GradedSpace(dims::NTuple{2,Int}; dual::Bool=false) = GradedSpace(dims, dual)

function GradedSpace(dims::Tuple{Vararg{Pair}}; dual::Bool=false)
    d = (0, 0)
    for (c, dc) in dims
        dc < 0 && throw(ArgumentError("Sector has negative dimension $dc"))
        k = convert(FermionParity, c)
        i = findindex(k)
        d = TupleTools.setindex(d, dc, i)
    end
    return GradedSpace(d, dual)
end
function GradedSpace(dim1::Pair, rdims::Vararg{Pair}; dual::Bool=false)
    return GradedSpace((dim1, rdims...); dual=dual)
end
function GradedSpace(dims::AbstractDict; dual::Bool=false)
    return GradedSpace(collect(dims)...; dual=dual)
end
GradedSpace(g::Base.Generator; dual::Bool=false) = GradedSpace(g...; dual=dual)

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

function dim(V::GradedSpace)
    return reduce(+, dim(V, c) for c in sectors(V); init=0)
end
dim(V::GradedSpace, c::FermionParity) = V.dims[findindex(c)]

function sectors(V::GradedSpace)
    return [FermionParity(i) for i in 0:1 if V.dims[i + 1] != 0]
end

hassector(V::GradedSpace, s::FermionParity) = dim(V, s) != 0

dual(V::GradedSpace) = conj(V)
Base.conj(V::GradedSpace) = GradedSpace(V.dims, !V.dual)
isdual(V::GradedSpace) = V.dual

# equality / comparison
function Base.:(==)(V₁::GradedSpace, V₂::GradedSpace)
    return (V₁.dims == V₂.dims) && V₁.dual == V₂.dual
end

Base.oneunit(::Type{GradedSpace}) = GradedSpace(one(FermionParity) => 1)
Base.zero(::Type{GradedSpace}) = GradedSpace(one(FermionParity) => 0)
Base.oneunit(V::GradedSpace) = oneunit(typeof(V))
Base.zero(V::GradedSpace) = zero(typeof(V))

function ⊕(V₁::GradedSpace, V₂::GradedSpace)
    dual1 = isdual(V₁)
    dual1 == isdual(V₂) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{FermionParity,Int}()
    for c in union(sectors(V₁), sectors(V₂))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V₁, c) + dim(V₂, c)
    end
    return typeof(V₁)(dims; dual=dual1)
end
⊕(V₁::GradedSpace, V₂::GradedSpace, Vs::GradedSpace...) = ⊕(⊕(V₁, V₂), Vs...)

function fuse(V₁::GradedSpace, V₂::GradedSpace)
    dims = SectorDict{FermionParity,Int}()
    for a in sectors(V₁), b in sectors(V₂)
        c = a ⊗ b # unique fusion outcome for FermionParity
        dims[c] = get(dims, c, 0) + dim(V₁, a) * dim(V₂, b)
    end
    return typeof(V₁)(dims)
end
function fuse(V₁::GradedSpace, V₂::GradedSpace, V₃::GradedSpace...)
    return fuse(fuse(V₁, V₂), V₃...)
end
fuse(V::GradedSpace) = V

function infimum(V₁::GradedSpace, V₂::GradedSpace)
    if V₁.dual == V₂.dual
        return GradedSpace(c => min(dim(V₁, c), dim(V₂, c))
                           for c in union(sectors(V₁), sectors(V₂)); dual=V₁.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V₁::GradedSpace, V₂::GradedSpace)
    if V₁.dual == V₂.dual
        return GradedSpace(c => max(dim(V₁, c), dim(V₂, c))
                           for c in union(sectors(V₁), sectors(V₂)); dual=V₁.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end


const FermionicSpace = GradedSpace
