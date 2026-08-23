"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

# struct TrivialTreeTransformer <: TreeTransformer end

struct AbelianTreeTransformer{T,N,F1,F2,F3,F4} <: TreeTransformer
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}
    structure_dst::FusionBlockStructure{N,F1,F2}
    structure_src::FusionBlockStructure{N,F3,F4}
end

function treetransformertype(Vdst, Vsrc)
    N = numind(Vdst)
    F1 = fusiontreetype(numout(Vdst))
    F2 = fusiontreetype(numin(Vdst))
    F3 = fusiontreetype(numout(Vsrc))
    F4 = fusiontreetype(numin(Vsrc))

    return AbelianTreeTransformer{Int,N,F1,F2,F3,F4}
end

function TreeTransformer(transform::Function, Vsrc::HomSpace, Vdst::HomSpace)
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    rows = Int[]
    cols = Int[]
    vals = Int[]

    for (row, (f1, f2)) in enumerate(structure_src.fusiontreelist)
        for ((f3, f4), coeff) in transform(f1, f2)
            col = structure_dst.fusiontreeindices[(f3, f4)]
            push!(rows, row)
            push!(cols, col)
            push!(vals, coeff)
        end
    end

    return AbelianTreeTransformer(rows, cols, vals, structure_dst, structure_src)
end

# transpose 的树级变换：平面置换（cyclic），无 crossing 符号。
# 对 FermionParity，fold/bend 类 duality 操作的系数恒为 1，因此可直接
# 复用 permute 的树重排逻辑并把符号置为 1。
function _treetranspose(f₁::FusionTree, f₂::FusionTree,
                        p1::IndexTuple, p2::IndexTuple)
    ((f₁′, f₂′), coeff) = only(permute(f₁, f₂, p1, p2))
    return SingletonDict((f₁′, f₂′) => one(coeff))
end

for (transform, transformer) in ((:permute, :permuter), (:transpose, :transposer))
    treetransformcache = Symbol("tree", transformer, "cache")
    usetreetransformcache = Symbol("usetree", transformer, "cache")
    treetransformer = Symbol("tree", transformer)
    _get_treetransformer = Symbol("_get_", treetransformer)
    _treetransformer = Symbol("_", treetransformer)
    treeop = transform === :transpose ? :_treetranspose : :permute

    @eval begin
        const $treetransformcache = Dict{Any,Any}()
        const $usetreetransformcache = Ref{Bool}(true)

        function $treetransformer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
            return fusiontreetransform(f1, f2) = $treeop(f1, f2, p...)
        end
        function $treetransformer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
            if $usetreetransformcache[]
                key = (space(tdst), space(tsrc), p)
                A = treetransformertype(space(tdst), space(tsrc))
                return $_get_treetransformer(A, key)
            else
                return $_treetransformer((space(tdst), space(tsrc), p))
            end
        end
        @noinline function $_get_treetransformer(A, key)
            d::A = get!($treetransformcache, key) do
                return $_treetransformer(key)
            end
            return d
        end
        function $_treetransformer((Vdst, Vsrc, p))
            fusiontreetransform(f1, f2) = $treeop(f1, f2, p...)
            return TreeTransformer(fusiontreetransform, Vsrc, Vdst)
        end
    end
end
