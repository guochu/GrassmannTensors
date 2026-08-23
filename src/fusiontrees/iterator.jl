
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
    trees = ((fusiontrees(uncoupled, coupled) for uncoupled in Iterators.product(uncoupleds...))...,)
    return TupleTools.flatten(trees)
end
function fusiontrees(uncoupleds::Tuple, coupled::FermionParity, isdualflags::Tuple)
    trees = ((fusiontrees(uncoupled, coupled, isdualflags)
              for uncoupled in Iterators.product(uncoupleds...))...,)
    return TupleTools.flatten(trees)
end
