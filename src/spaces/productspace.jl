

struct ProductSpace{N}
    spaces::NTuple{N,GradedSpace}
    ProductSpace{N}(spaces::NTuple{N,GradedSpace}) where {N} = new{N}(spaces)
end
function ProductSpace{N}(spaces::Vararg{GradedSpace,N}) where {N}
    return ProductSpace{N}(spaces)
end

function ProductSpace(spaces::Tuple{GradedSpace,Vararg{GradedSpace}})
    return ProductSpace{length(spaces)}(spaces)
end
function ProductSpace(space1::GradedSpace, rspaces::Vararg{GradedSpace})
    return ProductSpace((space1, rspaces...))
end

ProductSpace(P::ProductSpace) = P

# Corresponding methods
#-----------------------
dims(P::ProductSpace) = map(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n])
dim(P::ProductSpace) = prod(dims(P))

dual(P::ProductSpace{0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

sectors(P::ProductSpace) = product(map(sectors, P.spaces)...)

function hassector(V::ProductSpace{N}, s::NTuple{N,FermionParity}) where {N}
    return reduce(&, map(hassector, V.spaces, s); init=true)
end

function dims(P::ProductSpace{N}, sector::NTuple{N,FermionParity}) where {N}
    return map(dim, P.spaces, sector)
end

function dim(P::ProductSpace{N}, sector::NTuple{N,FermionParity}) where {N}
    return reduce(*, dims(P, sector); init=1)
end

function blocksectors(P::ProductSpace{N}) where {N}
    bs = Vector{FermionParity}()
    if N == 0
        push!(bs, one(FermionParity))
    elseif N == 1
        for s in sectors(P)
            push!(bs, first(s))
        end
    else
        for s in sectors(P)
            c = ⊗(s...)
            if !(c in bs)
                push!(bs, c)
            end
        end
    end
    return sort!(bs)
end


function fusiontrees(P::ProductSpace{N}, blocksector::FermionParity) where {N}
    uncoupled = map(sectors, P.spaces)
    isdualflags = map(isdual, P.spaces)
    return fusiontrees(uncoupled, blocksector, isdualflags)
end

hasblock(P::ProductSpace, c::FermionParity) = !isempty(fusiontrees(P, c))

function blockdim(P::ProductSpace, c::FermionParity)
    d = 0
    for f in fusiontrees(P, c)
        d += dim(P, f.uncoupled)
    end
    return d
end

function Base.:(==)(P1::ProductSpace{N}, P2::ProductSpace{N}) where {N}
    return (P1.spaces == P2.spaces)
end
Base.:(==)(P1::ProductSpace, P2::ProductSpace) = false

Base.hash(P::ProductSpace{N}, h::UInt) where {N} = hash(P.spaces, hash(N, h))

# Default construction from product of spaces
#---------------------------------------------
⊗(V::GradedSpace, Vrest::GradedSpace...) = ProductSpace(V, Vrest...)
⊗(P::ProductSpace) = P
function ⊗(P1::ProductSpace{N₁}, P2::ProductSpace{N₂}) where {N₁,N₂}
    return ProductSpace{N₁ + N₂}((P1.spaces..., P2.spaces...))
end
function ⊗(P::ProductSpace{N}, V::GradedSpace) where {N}
    return ProductSpace{N + 1}((P.spaces..., V))
end


# unit element with respect to the monoidal structure of taking tensor products
Base.one(::Type{GradedSpace}) = ProductSpace{0}(())
Base.one(::Type{<:ProductSpace}) = ProductSpace{0}(())
Base.one(V::GradedSpace) = one(typeof(V)) # 与 TensorKit 一致：one(V) 是空积空间 ProductSpace{0}
Base.one(P::ProductSpace) = one(typeof(P))

# spacetype for composite spaces（用于空间之间的比较）
spacetype(::Type{<:ProductSpace}) = ProductSpace
spacetype(P::ProductSpace) = ProductSpace

Base.:^(V::GradedSpace, N::Int) = ProductSpace{N}(ntuple(n -> V, N))
Base.:^(V::ProductSpace, N::Int) = ⊗(ntuple(n -> V, N)...)
function Base.literal_pow(::typeof(^), V::GradedSpace, p::Val{N}) where {N}
    return ProductSpace{N}(ntuple(n -> V, p))
end

fuse(P::ProductSpace{0}) = oneunit(GradedSpace)
fuse(P::ProductSpace) = fuse(P.spaces...)

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]

Base.iterate(P::ProductSpace, args...) = Base.iterate(P.spaces, args...)
Base.indexed_iterate(P::ProductSpace, args...) = Base.indexed_iterate(P.spaces, args...)

Base.eltype(::Type{<:ProductSpace}) = GradedSpace
Base.eltype(P::ProductSpace) = eltype(typeof(P))

Base.IteratorEltype(::Type{<:ProductSpace}) = Base.HasEltype()
Base.IteratorSize(::Type{<:ProductSpace}) = Base.HasLength()

Base.reverse(P::ProductSpace) = ProductSpace(reverse(P.spaces))




# Promotion and conversion
# ------------------------
function Base.promote_rule(::Type{ProductSpace{N}}, ::Type{GradedSpace}) where {N}
    return ProductSpace
end
function Base.promote_rule(::Type{GradedSpace}, ::Type{ProductSpace{N}}) where {N}
    return ProductSpace
end


# GradedSpace to ProductSpace
Base.convert(::Type{<:ProductSpace}, P::ProductSpace) = P
Base.convert(::Type{<:ProductSpace}, V::GradedSpace) = ⊗(V)
