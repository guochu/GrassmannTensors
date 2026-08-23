# BraidingTensor:
# special (2,2) tensor that implements a standard braiding operation
#====================================================================#
"""
    struct BraidingTensor{T} <: AbstractTensorMap{T,2,2}
    BraidingTensor(V1::GradedSpace, V2::GradedSpace, adjoint::Bool=false)

Specific subtype of [`AbstractTensorMap`](@ref) for representing the braiding tensor that
braids the first input over the second input; its inverse can be obtained as the adjoint.

It holds that `domain(BraidingTensor(V1, V2)) == V1 ⊗ V2` and
`codomain(BraidingTensor(V1, V2)) == V2 ⊗ V1`.
"""
struct BraidingTensor{T} <: AbstractTensorMap{T,2,2}
    V1::GradedSpace
    V2::GradedSpace
    adjoint::Bool
    function BraidingTensor{T}(V1::GradedSpace, V2::GradedSpace,
                               adjoint::Bool=false) where {T}
        # for FermionParity the exchange symmetry is automatic: Nsymbol(a,b,c) == Nsymbol(b,a,c)
        # (Z₂ fusion is commutative), so no consistency check is required here
        return new{T}(V1, V2, adjoint)
        # partial construction: only construct rowr and colr when needed
    end
end
function BraidingTensor(V1::GradedSpace, V2::GradedSpace, adjoint::Bool=false)
    # FermionParity 为费米型编织，默认使用复数标量
    return BraidingTensor{ComplexF64}(V1, V2, adjoint)
end
function BraidingTensor(V::HomSpace, adjoint::Bool=false)
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor(V[2], V[1], adjoint)
end
function BraidingTensor{T}(V::HomSpace, adjoint::Bool=false) where {T}
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor{T}(V[2], V[1], adjoint)
end
function Base.adjoint(b::BraidingTensor{T}) where {T}
    return BraidingTensor{T}(b.V1, b.V2, !b.adjoint)
end

space(b::BraidingTensor) = b.adjoint ? b.V1 ⊗ b.V2 ← b.V2 ⊗ b.V1 : b.V2 ⊗ b.V1 ← b.V1 ⊗ b.V2

# TODO: this will probably give issues with GPUs, so we should try to avoid
# calling this method alltogether
storagetype(::Type{BraidingTensor{T}}) where {T} = Vector{T}

# artin_braid(f, 1) 在 2 腿树上的特化：交换前两条腿。
# Z₂ 费米子下符号恒为 Rsymbol(a, b, c)（c = f.coupled = a ⊗ b），
# 逆编织（inv）仅取共轭，而 Rsymbol 为实值 ±1，故正逆无差别。
@inline function braidfirst(f::FusionTree{2})
    f′ = FusionTree((f.uncoupled[2], f.uncoupled[1]), f.coupled,
                    (f.isdual[2], f.isdual[1]))
    return f′, Rsymbol(f.uncoupled[1], f.uncoupled[2], f.coupled)
end

@inline function Base.getindex(b::BraidingTensor, f₁::FusionTree{2},
                               f₂::FusionTree{2})
    c = f₁.coupled
    V1, V2 = domain(b)
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        ((f₁.uncoupled[1] ∈ sectors(V2)) && (f₂.uncoupled[1] ∈ sectors(V1))) ||
            throw(SectorMismatch())
        ((f₁.uncoupled[2] ∈ sectors(V1)) && (f₂.uncoupled[2] ∈ sectors(V2))) ||
            throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(codomain(b), f₁.uncoupled)..., dims(domain(b), f₂.uncoupled)...)
        n1 = d[1] * d[2]
        n2 = d[3] * d[4]
        data = sreshape(StridedView(Matrix{eltype(b)}(undef, n1, n2)), d)
        fill!(data, zero(eltype(b)))
        if f₁.uncoupled == reverse(f₂.uncoupled)
            # braidfirst 交换 2 条腿，Z₂ 下符号 = Rsymbol（逆编织符号相同）
            r = Rsymbol(f₂.uncoupled[1], f₂.uncoupled[2], f₂.coupled)
            @inbounds for i in axes(data, 1), j in axes(data, 2)
                data[i, j, j, i] = r
            end
        end
        return data
    end
end

# efficient copy constructor
Base.copy(b::BraidingTensor) = b

TensorMap(b::BraidingTensor) = copy!(similar(b), b)
Base.convert(::Type{TensorMap}, b::BraidingTensor) = TensorMap(b)

function block(b::BraidingTensor, s::FermionParity)
    sectortype(b) == typeof(s) || throw(SectorMismatch())

    # TODO: probably always square?
    m = blockdim(codomain(b), s)
    n = blockdim(domain(b), s)
    data = Matrix{eltype(b)}(undef, (m, n))

    length(data) == 0 && return data # s ∉ blocksectors(b)

    data = fill!(data, zero(eltype(b)))

    structure = fusionblockstructure(b)
    base_offset = first(structure.blockstructure[s][2]) - 1

    for ((f1, f2), (sz, str, off)) in
        zip(structure.fusiontreelist, structure.fusiontreestructure)
        if (f1.uncoupled != reverse(f2.uncoupled)) || !(f1.coupled == f2.coupled == s)
            continue
        end

        # braidfirst 交换 2 条腿，Z₂ 下符号 = Rsymbol（逆编织符号相同）
        r = Rsymbol(f2.uncoupled[1], f2.uncoupled[2], f2.coupled)

        # change offset to account for single block
        subblock = StridedView(data, sz, str, off - base_offset)
        @inbounds for i in axes(subblock, 1), j in axes(subblock, 2)
            subblock[i, j, j, i] = r
        end
    end

    return data
end

# Index manipulations
# -------------------
has_shared_permute(t::BraidingTensor, ::Index2Tuple) = false
function add_transform!(tdst::AbstractTensorMap,
                        tsrc::BraidingTensor,
                        (p₁, p₂)::Index2Tuple,
                        fusiontreetransform,
                        α::Number,
                        β::Number,
                        backend::AbstractBackend...)
    return add_transform!(tdst, TensorMap(tsrc), (p₁, p₂), fusiontreetransform, α, β,
                          backend...)
end

# VectorInterface
# ---------------
# TODO

# TensorOperations
# ----------------
# TODO: implement specialized methods

function TO.tensoradd!(C::AbstractTensorMap,
                       A::BraidingTensor, pA::Index2Tuple, conjA,
                       α::Number, β::Number, backend=TO.DefaultBackend(),
                       allocator=TO.DefaultAllocator())
    return TO.tensoradd!(C, TensorMap(A), pA, conjA, α, β, backend, allocator)
end

# Planar operations
# -----------------
# TODO: implement specialized methods

function planaradd!(C::AbstractTensorMap,
                    A::BraidingTensor, p::Index2Tuple,
                    α::Number, β::Number,
                    backend, allocator)
    return planaradd!(C, TensorMap(A), p, α, β, backend, allocator)
end

function planarcontract!(C::AbstractTensorMap,
                         A::BraidingTensor,
                         (oindA, cindA)::Index2Tuple,
                         B::AbstractTensorMap,
                         (cindB, oindB)::Index2Tuple,
                         (p1, p2)::Index2Tuple,
                         α::Number, β::Number,
                         backend, allocator)
    # special case only defined for contracting 2 indices
    length(oindA) == length(cindA) == 2 ||
        return planarcontract!(C, TensorMap(A), (oindA, cindA), B, (cindB, oindB), (p1, p2),
                               α, β, backend, allocator)

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
                                                 oindB, cindB, p1, p2)

    if space(B, cindB[1]) != space(A, cindA[1])' ||
       space(B, cindB[2]) != space(A, cindA[2])'
        throw(SpaceMismatch("$(space(C)) ≠ permute($(space(A))[$oindA, $cindA] * $(space(B))[$cindB, $oindB], ($p1, $p2)"))
    end

    # FermionParity 是费米型编织（非 Bosonic），直接走编织收缩路径
    scale!(C, β)

    for (f₁, f₂) in fusiontrees(B)
        local newtrees
        for ((f₁′, f₂′), coeff′) in _treetranspose(f₁, f₂, cindB, oindB)
            # braidfirst(f₁′)：交换 f₁′ 前两条腿；Z₂ 下符号 = Rsymbol（逆编织符号相同）
            f₁′′, coeff′′ = braidfirst(f₁′)
            f12 = (f₁′′, f₂′)
            coeff = coeff′ * coeff′′
            if @isdefined newtrees
                newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
            else
                newtrees = Dict(f12 => coeff)
            end
        end
        for ((f₁′, f₂′), coeff) in newtrees
            TO.tensoradd!(C[f₁′, f₂′], B[f₁, f₂], (reverse(cindB), oindB), false, α * coeff,
                          One(), backend, allocator)
        end
    end
    return C
end
function planarcontract!(C::AbstractTensorMap,
                         A::AbstractTensorMap,
                         (oindA, cindA)::Index2Tuple,
                         B::BraidingTensor,
                         (cindB, oindB)::Index2Tuple,
                         (p1, p2)::Index2Tuple,
                         α::Number, β::Number,
                         backend, allocator)
    # special case only defined for contracting 2 indices
    length(oindB) == length(cindB) == 2 ||
        return planarcontract!(C, A, (oindA, cindA), TensorMap(B), (cindB, oindB), (p1, p2),
                               α, β, backend, allocator)

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
                                                 oindB, cindB, p1, p2)

    if space(B, cindB[1]) != space(A, cindA[1])' ||
       space(B, cindB[2]) != space(A, cindA[2])'
        throw(SpaceMismatch("$(space(C)) ≠ permute($(space(A))[$oindA, $cindA] * $(space(B))[$cindB, $oindB], ($p1, $p2)"))
    end

    # FermionParity 是费米型编织（非 Bosonic），直接走编织收缩路径
    scale!(C, β)

    for (f₁, f₂) in fusiontrees(A)
        local newtrees
        for ((f₁′, f₂′), coeff′) in _treetranspose(f₁, f₂, oindA, cindA)
            # braidfirst(f₂′)：交换 f₂′ 前两条腿；Z₂ 下符号 = Rsymbol（逆编织符号相同）
            f₂′′, coeff′′ = braidfirst(f₂′)
            f12 = (f₁′, f₂′′)
            coeff = coeff′ * conj(coeff′′)
            if @isdefined newtrees
                newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
            else
                newtrees = Dict(f12 => coeff)
            end
        end
        for ((f₁′, f₂′), coeff) in newtrees
            TO.tensoradd!(C[f₁′, f₂′], A[f₁, f₂], (oindA, reverse(cindA)), false, α * coeff,
                          One(), backend, allocator)
        end
    end
    return C
end

# ambiguity fix:
function planarcontract!(C::AbstractTensorMap, A::BraidingTensor, pA::Index2Tuple,
                         B::BraidingTensor, pB::Index2Tuple, pAB::Index2Tuple,
                         α::Number, β::Number, backend,
                         allocator)
    return planarcontract!(C, TensorMap(A), pA, TensorMap(B), pB, pAB, α, β, backend,
                           allocator)
end

function planartrace!(C::AbstractTensorMap,
                      A::BraidingTensor,
                      p::Index2Tuple, q::Index2Tuple,
                      α::Number, β::Number,
                      backend,
                      allocator)
    return planartrace!(C, TensorMap(A), p, q, α, β, backend, allocator)
end
