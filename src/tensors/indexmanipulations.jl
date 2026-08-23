

@propagate_inbounds function Base.permute!(tdst::AbstractTensorMap,
                                           tsrc::AbstractTensorMap,
                                           p::Index2Tuple)
    return add_permute!(tdst, tsrc, p, One(), Zero())
end


function permute(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple{N₁,N₂};
                 copy::Bool=false) where {N₁,N₂}
    space′ = permute(space(t), (p₁, p₂))
    # share data if possible
    if !copy && p₁ === codomainind(t) && p₂ === domainind(t)
        return t
    end
    # general case
    @inbounds begin
        return permute!(similar(t, space′), t, (p₁, p₂))
    end
end
function permute(t::TensorMap, (p₁, p₂)::Index2Tuple{N₁,N₂}; copy::Bool=false) where {N₁,N₂}
    space′ = permute(space(t), (p₁, p₂))
    # share data if possible
    if !copy
        if p₁ === codomainind(t) && p₂ === domainind(t)
            return t
        elseif has_shared_permute(t, (p₁, p₂))
            return TensorMap(t.data, space′)
        end
    end
    # general case
    @inbounds begin
        return permute!(similar(t, space′), t, (p₁, p₂))
    end
end
function permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple; copy::Bool=false)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return adjoint(permute(adjoint(t), (p₁′, p₂′); copy))
end
function permute(t::AbstractTensorMap, p::IndexTuple; copy::Bool=false)
    return permute(t, (p, ()); copy)
end
permute(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false) = permute(t, (p1, p2); copy=copy)


function has_shared_permute(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    return (p₁ === codomainind(t) && p₂ === domainind(t))
end
function has_shared_permute(t::TensorMap, (p₁, p₂)::Index2Tuple)
    if p₁ === codomainind(t) && p₂ === domainind(t)
        return true
    else
        return false
    end
end
function has_shared_permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return has_shared_permute(t', (p₁′, p₂′))
end


# Braid
#-------
@propagate_inbounds function braid!(tdst::AbstractTensorMap,
                                    tsrc::AbstractTensorMap,
                                    p::Index2Tuple,
                                    levels::IndexTuple)
    return add_braid!(tdst, tsrc, p, levels, One(), Zero())
end

function braid(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple, levels::IndexTuple;
               copy::Bool=false)
    @assert length(levels) == numind(t)
    # FermionParity 为 SymmetricBraiding：编织退化为一个置换
    return permute(t, (p₁, p₂); copy=copy)
end

# Transpose
#------------
_transpose_indices(t::AbstractTensorMap) = (reverse(domainind(t)), reverse(codomainind(t)))

function LinearAlgebra.transpose!(tdst::AbstractTensorMap,
                                  tsrc::AbstractTensorMap,
                                  (p₁, p₂)::Index2Tuple=_transpose_indices(tsrc))
    return add_transpose!(tdst, tsrc, (p₁, p₂), One(), Zero())
end

function LinearAlgebra.transpose(t::AbstractTensorMap,
                                 (p₁, p₂)::Index2Tuple=_transpose_indices(t);
                                 copy::Bool=false)
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    space′ = permute(space(t), (p₁, p₂))
    @inbounds begin
        return transpose!(similar(t, space′), t, (p₁, p₂))
    end
end

function LinearAlgebra.transpose(t::AdjointTensorMap,
                                 (p₁, p₂)::Index2Tuple=_transpose_indices(t);
                                 copy::Bool=false)
    p₁′ = map(n -> adjointtensorindex(t, n), p₂)
    p₂′ = map(n -> adjointtensorindex(t, n), p₁)
    return adjoint(transpose(adjoint(t), (p₁′, p₂′); copy=copy))
end

# Repartition
#-------------
function repartition!(tdst::AbstractTensorMap{S}, tsrc::AbstractTensorMap{S}) where {S}
    numind(tsrc) == numind(tdst) ||
        throw(ArgumentError("tsrc and tdst should have an equal amount of indices"))
    all_inds = (codomainind(tsrc)..., reverse(domainind(tsrc))...)
    p₁ = ntuple(i -> all_inds[i], numout(tdst))
    p₂ = reverse(ntuple(i -> all_inds[i + numout(tdst)], numin(tdst)))
    return transpose!(tdst, tsrc, (p₁, p₂))
end

@constprop :aggressive function repartition(t::AbstractTensorMap, N₁::Int,
                                            N₂::Int=numind(t) - N₁;
                                            copy::Bool=false)
    N₁ + N₂ == numind(t) ||
        throw(ArgumentError("Invalid repartition: $(numind(t)) to ($N₁, $N₂)"))
    all_inds = (codomainind(t)..., reverse(domainind(t))...)
    p₁ = ntuple(i -> all_inds[i], N₁)
    p₂ = reverse(ntuple(i -> all_inds[i + N₁], N₂))
    return transpose(t, (p₁, p₂); copy)
end

# Twist
#-------
function twist!(t::AbstractTensorMap, is; inv::Bool=false)
    if !all(in(allind(t)), is)
        msg = "Can't twist indices $is of a tensor with only $(numind(t)) indices."
        throw(ArgumentError(msg))
    end
    isempty(is) && return t
    N₁ = numout(t)
    for (f₁, f₂) in fusiontrees(t)
        θ = prod(i -> i <= N₁ ? twist(f₁.uncoupled[i]) : twist(f₂.uncoupled[i - N₁]), is)
        inv && (θ = θ')
        rmul!(t[f₁, f₂], θ)
    end
    return t
end

twist(t::AbstractTensorMap, i; inv::Bool=false) = twist!(copy(t), i; inv)


#-------------------------------------
# Full implementations based on `add`
#-------------------------------------

@propagate_inbounds function add_permute!(tdst::AbstractTensorMap,
                                          tsrc::AbstractTensorMap,
                                          p::Index2Tuple,
                                          α::Number,
                                          β::Number,
                                          backend::AbstractBackend...)
    transformer = treepermuter(tdst, tsrc, p)
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

@propagate_inbounds function add_braid!(tdst::AbstractTensorMap,
                                        tsrc::AbstractTensorMap,
                                        p::Index2Tuple,
                                        levels::IndexTuple,
                                        α::Number,
                                        β::Number,
                                        backend::AbstractBackend...)
    length(levels) == numind(tsrc) ||
        throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(tsrc)) ← $(domain(tsrc))"))
    # FermionParity has SymmetricBraiding: braiding reduces to a permutation
    return add_permute!(tdst, tsrc, p, α, β, backend...)
end

@propagate_inbounds function add_transpose!(tdst::AbstractTensorMap,
                                            tsrc::AbstractTensorMap,
                                            p::Index2Tuple,
                                            α::Number,
                                            β::Number,
                                            backend::AbstractBackend...)
    transformer = treetransposer(tdst, tsrc, p)
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

function add_transform!(tdst::AbstractTensorMap,
                        tsrc::AbstractTensorMap,
                        (p₁, p₂)::Index2Tuple,
                        transformer,
                        α::Number,
                        β::Number,
                        backend::AbstractBackend...)
    @boundscheck begin
        permute(space(tsrc), (p₁, p₂)) == space(tdst) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end

    if p₁ === codomainind(tsrc) && p₂ === domainind(tsrc)
        add!(tdst, tsrc, α, β)
    else
        add_transform_kernel!(tdst, tsrc, (p₁, p₂), transformer, α, β, backend...)
    end

    return tdst
end

function add_transform_kernel!(tdst::TensorMap,
                               tsrc::TensorMap,
                               (p₁, p₂)::Index2Tuple,
                               transformer::AbelianTreeTransformer,
                               α::Number,
                               β::Number,
                               backend::AbstractBackend...)
    structure_dst = transformer.structure_dst.fusiontreestructure
    structure_src = transformer.structure_src.fusiontreestructure

    # TODO: this could be multithreaded
    for (row, col, val) in zip(transformer.rows, transformer.cols, transformer.vals)
        sz_dst, str_dst, offset_dst = structure_dst[col]
        subblock_dst = StridedView(tdst.data, sz_dst, str_dst, offset_dst)

        sz_src, str_src, offset_src = structure_src[row]
        subblock_src = StridedView(tsrc.data, sz_src, str_src, offset_src)

        TO.tensoradd!(subblock_dst, subblock_src, (p₁, p₂), false, α * val, β, backend...)
    end

    return nothing
end

function add_transform_kernel!(tdst::AbstractTensorMap,
                               tsrc::AbstractTensorMap,
                               (p₁, p₂)::Index2Tuple,
                               fusiontreetransform::Function,
                               α::Number,
                               β::Number,
                               backend::AbstractBackend...)
    I = sectortype(spacetype(tdst))
    _add_abelian_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
    return nothing
end


function _add_abelian_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    # if Threads.nthreads() > 1
    #     Threads.@sync for (f₁, f₂) in fusiontrees(tsrc)
    #         Threads.@spawn _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
    #                                            f₁, f₂, α, β, backend...)
    #     end
    # else
        for (f₁, f₂) in fusiontrees(tsrc)
            _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
                                f₁, f₂, α, β, backend...)
        end
    # end
    return nothing
end

function _add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β, backend...)
    (f₁′, f₂′), coeff = first(fusiontreetransform(f₁, f₂))
    @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, β,
                            backend...)
    return nothing
end

