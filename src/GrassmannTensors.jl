
module GrassmannTensors

# Exports
#---------
# Types:
export FermionParity, fℤ₂

export GradedSpace, FermionicSpace
export ProductSpace
export FusionTree
export HomSpace, TensorSpace, TensorMapSpace
export AbstractTensorMap, AbstractTensor, TensorMap, Tensor
export DiagonalTensorMap, BraidingTensor
export TruncationScheme
export SpaceMismatch, SectorMismatch, IndexError # error types

# general vector space methods
export space, field, dual, dim, reduceddim, dims, fuse, flip, isdual, oplus,
       insertleftunit, insertrightunit, removeunit

# partial order for vector spaces
export infimum, supremum, isisomorphic, ismonomorphic, isepimorphic

# methods for sectors and properties thereof
export sectortype, sectors, hassector, Rsymbol, otimes, twist
export fusiontrees, braid, permute, transpose

# some unicode
export ⊕, ⊗, ℂ, ℝ, ℤ, ←, →, ≾, ≿, ≅, ≺, ≻

# tensor maps
export domain, codomain, numind, numout, numin, domainind, codomainind, allind
export spacetype, sectortype, storagetype, scalartype, tensormaptype
export blocksectors, blockdim, block, blocks

# random methods for constructor
export rand, rand!, randn, randn!

# special purpose constructors
export zero, one, one!, id, id!, isomorphism, isomorphism!, isometry, isometry!, unitary

# reexport most of VectorInterface and some more tensor algebra
export zerovector, zerovector!, zerovector!!, scale, scale!, scale!!, add, add!, add!!
export inner, dot, norm, normalize, normalize!, tr

# factorizations
export mul!, lmul!, rmul!, adjoint!, pinv, axpy!, axpby!
export leftorth, rightorth, leftorth!, rightorth!, tsvd!, tsvd, isposdef, isposdef!, ishermitian
export permute, permute!, transpose, transpose!, twist, twist!, braid, braid!, repartition
export catdomain, catcodomain

export OrthogonalFactorizationAlgorithm, QR, QRpos, LQ, LQpos, SVD, SDD, Polar

# tensor operations
export @tensor, @tensoropt, @ncon, ncon, @planar, @plansor
export scalar, add!, contract!

# hadamard product
export @hadamard, hadamardproduct, hadamardproduct!, hadamardproduct_structure,
       hadamardproduct_type, tensoralloc_hadamard, checkhadamardproduct

# truncation schemes
export notrunc, truncerr, truncdim, truncspace, truncbelow, truncdimcutoff

# Imports
#---------
using TupleTools
using TupleTools: StaticLength

using Strided

using VectorInterface

using TensorOperations: TensorOperations, @tensor, @tensoropt, @ncon, ncon
using TensorOperations: IndexTuple, Index2Tuple, linearize, AbstractBackend
const TO = TensorOperations

# HadamardProduct 接口（为 AbstractTensorMap 扩展方法，需要 import）
import HadamardProduct
using HadamardProduct: @hadamard, hadamardproduct, tensoralloc_hadamard


using Base: @boundscheck, @propagate_inbounds, @constprop,
            OneTo, tail, front,
            tuple_type_head, tuple_type_tail, tuple_type_cons,
            SizeUnknown, HasLength, HasShape, IsInfinite, EltypeUnknown, HasEltype
using Base.Meta: isexpr
using Base.Iterators: product, filter

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: norm, dot, normalize, normalize!, tr,
                     axpy!, axpby!, lmul!, rmul!, mul!, ldiv!, rdiv!,
                     adjoint, adjoint!, transpose, transpose!,
                     lu, pinv, sylvester,
                     eigen, eigen!, svd, svd!,
                     isposdef, isposdef!, ishermitian, rank, cond,
                     Diagonal, Hermitian
using LinearAlgebra: LAPACK, triu!, BlasFloat, BlasReal, BlasComplex, checksquare


using Random: Random, rand!, randn!



include("sectors.jl")

# Auxiliary files
#-----------------
include("auxiliary/auxiliary.jl")
include("auxiliary/dicts.jl")
include("auxiliary/linalg.jl")

#--------------------------------------------------------------------
# experiment with different dictionaries
const SectorDict{K,V} = SortedVectorDict{K,V}
const FusionTreeDict{K,V} = Dict{K,V}
#--------------------------------------------------------------------

# Exception types:
#------------------
abstract type TensorException <: Exception end

# Exception type for all errors related to sector mismatch
struct SectorMismatch{S<:Union{Nothing,AbstractString}} <: TensorException
    message::S
end
SectorMismatch() = SectorMismatch{Nothing}(nothing)
Base.show(io::IO, ::SectorMismatch{Nothing}) = print(io, "SectorMismatch()")
Base.show(io::IO, e::SectorMismatch) = print(io, "SectorMismatch(\"", e.message, "\")")

# Exception type for all errors related to vector space mismatch
struct SpaceMismatch{S<:Union{Nothing,AbstractString}} <: TensorException
    message::S
end
SpaceMismatch() = SpaceMismatch{Nothing}(nothing)
Base.show(io::IO, ::SpaceMismatch{Nothing}) = print(io, "SpaceMismatch()")
Base.show(io::IO, e::SpaceMismatch) = print(io, "SpaceMismatch(\"", e.message, "\")")

# Exception type for all errors related to invalid tensor index specification.
struct IndexError{S<:Union{Nothing,AbstractString}} <: TensorException
    message::S
end
IndexError() = IndexError{Nothing}(nothing)
Base.show(io::IO, ::IndexError{Nothing}) = print(io, "IndexError()")
Base.show(io::IO, e::IndexError) = print(io, "IndexError(", e.message, ")")

# Constructing and manipulating fusion trees and iterators thereof
#------------------------------------------------------------------
include("fusiontrees/fusiontrees.jl")

# Definitions and methods for vector spaces
#-------------------------------------------
include("spaces/vectorspaces.jl")

# Definitions and methods for tensors
#-------------------------------------
# general definitions
include("tensors/abstracttensor.jl")
include("tensors/blockiterator.jl")
include("tensors/tensor.jl")
include("tensors/adjoint.jl")
include("tensors/linalg.jl")
include("tensors/vectorinterface.jl")
include("tensors/tensoroperations.jl")
include("tensors/treetransformers.jl")
include("tensors/indexmanipulations.jl")
include("tensors/hadamard.jl")
include("tensors/diagonal.jl")
include("tensors/truncation.jl")
include("tensors/factorizations.jl")
include("tensors/braidingtensor.jl")

# planar network formalism (needs to come after the tensors)
include("planar/analyzers.jl")
include("planar/preprocessors.jl")
include("planar/postprocessors.jl")
include("planar/planaroperations.jl")
include("planar/macros.jl")

include("shows.jl")

# trun off multi-thread
function __init__()
    Strided.disable_threads()
    set_num_blas_threads(1)
end

end
