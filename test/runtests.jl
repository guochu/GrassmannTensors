push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)))

using Test
using Random
using LinearAlgebra
using TensorOperations

using GrassmannTensors
using GrassmannTensors: FermionParity, fℤ₂, Rsymbol, twist, block, blockdim,
                        fusiontrees, HomSpace

Random.seed!(1234)

# ---- spaces used throughout the tests ----
# even/odd sector dims: (0 => d0, 1 => d1)
VFerm = (FermionicSpace(0 => 1, 1 => 1),
        FermionicSpace(0 => 1, 1 => 2)',
        FermionicSpace(0 => 3, 1 => 2)',
        FermionicSpace(0 => 2, 1 => 3),
        FermionicSpace(0 => 2, 1 => 5))

@testset "GrassmannTensors" begin
    include("sectors.jl")
    include("spaces.jl")
    include("tensors.jl")
    include("diagonal.jl")
    include("fermions.jl")
    include("fusiontrees.jl")
    include("braidingtensor.jl")
    include("planar.jl")
    include("hadamard.jl")
end
