@testset "Fields and vector spaces" begin
    for (V1, V2, V3, V4, V5) in (VFerm,)
        @testset "HomSpace" begin
            W = HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
            @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
            @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
            @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
            @test spacetype(W) == typeof(V1)
            @test sectortype(W) == sectortype(V1)
            @test W[1] == V1
            @test W[2] == V2
            @test W[3] == V3'
            @test W[4] == V4'
            @test W[5] == V5'
            @test W == permute(W, ((1, 2), (3, 4, 5)))
            @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
            @test (V1 ⊗ V2 ← V1 ⊗ V2) == GrassmannTensors.compose(W, W')
        end
        @testset "dim and blocks" begin
            # a compatible HomSpace V1⊗V2 ← V1'⊗V2'
            W = V1 ⊗ V2 ← V1' ⊗ V2'
            @test dim(W) == sum(blockdim(codomain(W), c) * blockdim(domain(W), c)
                                for c in blocksectors(W))
            @test dim(W) == dim(W')
            P = V1 ⊗ V2
            @test dim(P) == dim(V1) * dim(V2)
            for c in blocksectors(P)
                @test blockdim(P, c) > 0
                @test blockdim(P, c) == blockdim(P', c)
            end
            # fusion tree structure of a product space
            for f in fusiontrees(V1 ⊗ V2 ⊗ V3, one(FermionParity))
                @test length(f.uncoupled) == 3
                @test f.coupled == one(FermionParity)
            end
        end
    end
end
