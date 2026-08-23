diagspacelist = (FermionicSpace(0 => 2, 1 => 3),)

@testset "DiagonalTensor with domain $V" for V in diagspacelist
    @testset "Basic properties and algebra" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @test scalartype(t) == T
            @test codomain(t) == ProductSpace(V)
            @test domain(t) == ProductSpace(V)
            @test space(t) == (V ← V)
            @test space(t') == (V ← V)
            @test dim(t) == dim(space(t))
            # blocks
            bs = blocks(t)
            (c, b1), state = iterate(bs)
            @test c == first(blocksectors(V ← V))
            b2 = block(t, first(blocksectors(t)))
            @test b1 == b2
            # basic linear algebra
            @test isa(norm(t), real(T))
            @test norm(t)^2 ≈ dot(t, t)
            α = rand(T)
            @test norm(α * t) ≈ abs(α) * norm(t)
            @test norm(t + t, 2) ≈ 2 * norm(t, 2)
            @test norm(t) ≈ norm(t')

            @test t == TensorMap(t)
            @test norm(t + TensorMap(t)) ≈ 2 * norm(t)

            @test norm(one!(copy(t))) ≈ sqrt(dim(V))
            @test one!(copy(t)) == id(V)

            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            α = rand(T)
            β = rand(T)
            @test dot(t1, t2) ≈ conj(dot(t2, t1))
            @test dot(t2, t1) ≈ conj(dot(t2', t1'))
            @test dot(t2, α * t1 + β * t2) ≈ α * dot(t2, t1) + β * dot(t2, t2)
        end
    end
    @testset "Trace and Multiplication" begin
        t1 = DiagonalTensorMap(rand(Float64, reduceddim(V)), V)
        t2 = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
        @test tr(TensorMap(t1)) == tr(t1)
        @test tr(TensorMap(t2)) == tr(t2)
        @test TensorMap(t1 * t2) ≈ TensorMap(t1) * TensorMap(t2)

        u = randn(Float64, V ⊗ V' ⊗ V, V)
        @test u * t1 ≈ u * TensorMap(t1)
        @test t1 * u' ≈ TensorMap(t1) * u'
    end
end
