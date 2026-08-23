for (V1, V2, V3, V4, V5) in (VFerm,)
    @testset "Tensors" begin
        @testset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64)
                t = zeros(T, W)
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                # blocks
                bs = blocks(t)
                (c, b1), state = iterate(bs)
                @test c == first(blocksectors(W))
                b2 = block(t, first(blocksectors(t)))
                @test b1 == b2
                @test size(b1) == (blockdim(W, c), 1)
            end
        end

        @testset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                t = rand(T, W)
                d = convert(Dict, t)
                @test t == convert(TensorMap, d)
            end
        end

        @testset "Basic linear algebra" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = rand(T, W)
                @test scalartype(t) == T
                @test space(t) == W
                @test space(t') == W'
                @test dim(t) == dim(space(t))
                @test codomain(t) == codomain(W)
                @test domain(t) == domain(W)
                # linear algebra
                @test isa(norm(t), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
                @test norm(t) ≈ norm(t')

                t2 = rand!(similar(t))
                β = rand(T)
                @test dot(β * t2, α * t) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')

                i1 = isomorphism(T, V1 ⊗ V2, V2 ⊗ V1)
                i2 = isomorphism(Vector{T}, V2 ⊗ V1, V1 ⊗ V2)
                @test i1 * i2 == id(T, V1 ⊗ V2)
                @test i2 * i1 == id(Vector{T}, V2 ⊗ V1)

                w = isometry(T, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)), V1)
                @test dim(w) == 2 * dim(V1 ← V1)
                @test w' * w == id(Vector{T}, V1)
                @test w * w' == (w * w')^2
            end
        end

        @testset "Permutations: inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = rand(ComplexF64, W)
            t′ = randn!(similar(t))
            for (p1, p2) in (((1, 2, 3), (4, 5)),
                             ((5, 4), (3, 2, 1)),
                             ((1,), (2, 3, 4, 5)),
                             ((2, 3), (1, 4, 5)),
                             ((), (1, 2, 3, 4, 5)))
                t2 = permute(t, (p1, p2))
                @test norm(t2) ≈ norm(t)
                t2′ = permute(t′, (p1, p2))
                @test dot(t2′, t2) ≈ dot(t′, t)
            end
        end

        @testset "Full trace: self-consistency" begin
            t = rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
            t2 = permute(t, ((1, 2), (4, 3)))
            s = tr(t2)
            @test conj(s) ≈ tr(t2')
            # NOTE: for fermions tr(t2) (plain blockwise trace, as in TensorKit)
            # need not equal the full @tensor contraction, which carries twist
            # factors; we only check @tensor self-consistency here.
            @tensor s2 = t[a, b, b, a]
            @tensor t3[a, b] := t[a, c, c, b]
            @tensor s3 = t3[a, a]
            @test s2 ≈ s3
            # and tr is consistent with the plain block trace
            @test s ≈ sum(dim(c) * tr(b) for (c, b) in blocks(t2))
        end

        @testset "Partial trace: self-consistency" begin
            t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            @tensor t2[a, b] := t[c, d, b, d, c, a]
            @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
            @tensor t5[a, b] := t4[a, b, c, c]
            @test t2 ≈ t5
        end

        @testset "Trace and contraction" begin
            t1 = rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
            t2 = rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
            @tensor t3[1, 2, 3, 4, 5, 6] := t1[1, 2, 3] * t2[4, 5, 6]
            @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
            @tensor tb[a, b] := t3[x, y, a, y, b, x]
            @test ta ≈ tb
        end

        @testset "diag/diagm" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            t = randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end

        @testset "Factorization" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                for t in (rand(T, W), rand(T, W)')
                    Q, R = leftorth(t, ((3, 4, 2), (1, 5)); alg=GrassmannTensors.QR())
                    @test Q' * Q ≈ one(Q' * Q)
                    @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))

                    L, Q = rightorth(t, ((3, 4), (2, 1, 5)); alg=GrassmannTensors.LQ())
                    @test Q * Q' ≈ one(Q * Q')
                    @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))

                    U, S, V = tsvd(t, ((3, 4, 2), (1, 5)); alg=GrassmannTensors.SVD())
                    @test U' * U ≈ one(U' * U)
                    @test V * V' ≈ one(V * V')
                    @test U * S * V ≈ permute(t, ((3, 4, 2), (1, 5)))
                end
            end
        end
    end
end
