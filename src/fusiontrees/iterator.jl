
function couple(uncoupled::NTuple{N,FermionParity}) where {N}
    ⊗(uncoupled...)
end
function couple(uncoupled::NTuple{0})
    one(FermionParity)
end

function fusiontrees(uncoupled::NTuple{N,FermionParity}, coupled::FermionParity) where {N}
    if couple(uncoupled) == coupled
        return (FusionTree{N}(uncoupled, coupled), )
    else
        return ()
    end
end
function fusiontrees(uncoupled::NTuple{N,FermionParity}, coupled::FermionParity,
                     isdualflags::NTuple{N,Bool}) where {N}
    if couple(uncoupled) == coupled
        return (FusionTree{N}(uncoupled, coupled, isdualflags), )
    else
        return ()
    end
end
function fusiontrees(uncoupled::NTuple{0,FermionParity}, coupled::FermionParity)
    if one(FermionParity) == coupled
        return (FusionTree{0}(uncoupled, coupled), )
    else
        return ()
    end
end
function fusiontrees(uncoupleds::Tuple, coupled::FermionParity)
    # 注意：不能用 `(gen...,)` 收集全部组合再 TupleTools.flatten——腿数 N 较大时
    # （如 12 腿，2^N 个组合）嵌套元组过长，flatten 的类型推断会递归爆栈。
    # 直接在 Iterators.product 上过滤，收集成 Vector，无递归。
    N = length(uncoupleds)
    trees = FusionTree{N}[]
    for uncoupled in Iterators.product(uncoupleds...)
        couple(uncoupled) == coupled && push!(trees, FusionTree(uncoupled, coupled))
    end
    return trees
end
function fusiontrees(uncoupleds::Tuple, coupled::FermionParity, isdualflags::Tuple)
    N = length(uncoupleds)
    trees = FusionTree{N}[]
    for uncoupled in Iterators.product(uncoupleds...)
        couple(uncoupled) == coupled && push!(trees, FusionTree(uncoupled, coupled, isdualflags))
    end
    return trees
end
