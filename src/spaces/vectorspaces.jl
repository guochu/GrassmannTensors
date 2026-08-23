

# VECTOR SPACES:
#==============================================================================#
# 本包只使用一种确定的向量空间：GradedSpace（Z₂ 费米子分次空间，dims 固定为
# NTuple{2,Int}，分别对应 even/odd 两个 sector），以及由其组成的
# ProductSpace / HomSpace。不再有 VectorSpace / ElementarySpace / CompositeSpace
# 等抽象类型层次。

# include order matters: gradedspace must come before productspace/homspace
include("gradedspace.jl")
include("productspace.jl")
include("homspace.jl")

# 在类型定义之后补充 GradedSpace / ProductSpace 的便捷方法
sectortype(::Type{GradedSpace}) = FermionParity
sectortype(V::GradedSpace) = FermionParity
spacetype(::Type{GradedSpace}) = GradedSpace
spacetype(V::GradedSpace) = GradedSpace

reduceddim(V::GradedSpace) = sum(Base.Fix1(dim, V), sectors(V); init=0)

# make GradedSpace instances behave similar to ProductSpace instances
blocksectors(V::GradedSpace) = collect(sectors(V))
blockdim(V::GradedSpace, c::FermionParity) = dim(V, c)

# adjoint 便捷方法
Base.adjoint(V::GradedSpace) = dual(V)
Base.adjoint(P::ProductSpace) = dual(P)



# Partial order for vector spaces
#---------------------------------
function isisomorphic(V₁::Union{GradedSpace,ProductSpace}, V₂::Union{GradedSpace,ProductSpace})
    spacetype(V₁) == spacetype(V₂) || return false
    for c in union(blocksectors(V₁), blocksectors(V₂))
        if blockdim(V₁, c) != blockdim(V₂, c)
            return false
        end
    end
    return true
end

function ismonomorphic(V₁::Union{GradedSpace,ProductSpace}, V₂::Union{GradedSpace,ProductSpace})
    spacetype(V₁) == spacetype(V₂) || return false
    for c in blocksectors(V₁)
        if blockdim(V₁, c) > blockdim(V₂, c)
            return false
        end
    end
    return true
end

function isepimorphic(V₁::Union{GradedSpace,ProductSpace}, V₂::Union{GradedSpace,ProductSpace})
    spacetype(V₁) == spacetype(V₂) || return false
    for c in blocksectors(V₂)
        if blockdim(V₁, c) < blockdim(V₂, c)
            return false
        end
    end
    return true
end

# unicode alternatives
const ≅ = isisomorphic
const ≾ = ismonomorphic
const ≿ = isepimorphic

≺(V₁::Union{GradedSpace,ProductSpace}, V₂::Union{GradedSpace,ProductSpace}) = V₁ ≾ V₂ && !(V₁ ≿ V₂)
≻(V₁::Union{GradedSpace,ProductSpace}, V₂::Union{GradedSpace,ProductSpace}) = V₁ ≿ V₂ && !(V₁ ≾ V₂)




infimum(V₁::GradedSpace, V₂::GradedSpace, V₃::GradedSpace...) = infimum(infimum(V₁, V₂), V₃...)

function supremum(V₁::GradedSpace, V₂::GradedSpace, V₃::GradedSpace...)
    return supremum(supremum(V₁, V₂), V₃...)
end
