# Fermionic sign conventions specific to GrassmannTensors (FermionParity)
e = FermionParity(false)
o = FermionParity(true)
Vo = FermionicSpace(1 => 2)           # odd-only space (dim 2 in the odd sector)
Ve = FermionicSpace(0 => 3)           # even-only space

@testset "Fermionic signs" begin
    @testset "twist on tensor data" begin
        t = randn(ComplexF64, Vo ← Vo)
        tt = twist(t, 1)
        @test space(tt) == space(t)
        @test block(tt, o) ≈ -block(t, o)          # odd block flips sign
        # even-only space: twist is the identity
        te = randn(ComplexF64, Ve ← Ve)
        @test block(twist(te, 1), e) ≈ block(te, e)
    end

    @testset "permute sign: exchange two odd legs" begin
        A = randn(ComplexF64, Vo ⊗ Vo ← one(Vo))
        A2 = permute(A, (2, 1))
        blk = block(A, e)
        blk2 = block(A2, e)
        idx(i, j) = (i - 1) * 2 + j
        for i in 1:2, j in 1:2
            @test blk2[idx(j, i), 1] ≈ -blk[idx(i, j), 1]
        end
    end

    @testset "braid == permute" begin
        A = randn(ComplexF64, Vo ⊗ Vo ← one(Vo))
        Ab = braid(A, ((2, 1), ()), (2, 1))
        @test block(Ab, e) ≈ block(permute(A, (2, 1)), e)
    end

    @testset "(1,1)∘(1,1) contraction: no twist on dual domain leg" begin
        ao = randn(ComplexF64, Vo ← Vo)
        bo = randn(ComplexF64, Vo ← Vo)
        manual = block(ao, o) * block(bo, o)
        @test block(ao * bo, o) ≈ manual
        # cross-check via @tensor (result is a (2,0) Tensor fusing to sector e)
        Cref = @tensor Cref[a, b] := ao[a, c] * bo[c, b]
        @test block(Cref, e) ≈ reshape(manual, 4, 1)
    end

    @testset "tr of odd (1,1) tensor" begin
        M = randn(ComplexF64, Vo ← Vo)
        @test tr(M) ≈ tr(block(M, o))
    end

    @testset "ncon vs @tensor" begin
        Ao = randn(ComplexF64, Vo ⊗ Vo ← one(Vo))
        Bo = randn(ComplexF64, one(Vo) ← Vo ⊗ Vo)
        Cn = ncon((Ao, Bo), ([-1, 1], [1, -2]))
        Cr = @tensor Cr[-1, -2] := Ao[-1, 1] * Bo[1, -2]
        @test Cn ≈ Cr
    end
end

@testset "BraidingTensor" begin
    V = FermionicSpace(2 => 3, 1 => 2)
    for inv in (false, true)
        b = BraidingTensor(V, V, inv)
        @test b isa BraidingTensor
        @test space(b) == (V ⊗ V ← V ⊗ V)
        @test space(b') == (V ⊗ V ← V ⊗ V)
        # block size: codomain blockdim for sector e = 3*3 (e⊗e) + 2*2 (o⊗o) = 13
        @test size(block(b, e)) == (13, 13)
        # elementary braid on two odd legs: (o,o) couples to e, Rsymbol = -1
        # (for Z₂ the inverse braid equals the braid since ±1 is its own inverse)
        f = only(fusiontrees((o, o), e))
        d = b[f, f]
        @test size(d) == (2, 2, 2, 2)
        @test all(d[i, j, j, i] == -1 for i in 1:2, j in 1:2)
        # off-diagonal data is zero
        @test all(d[i, j, k, l] == 0
                  for i in 1:2, j in 1:2, k in 1:2, l in 1:2 if !(k == j && l == i))
        # inverse braid gives the same data
        b′ = BraidingTensor(V, V, true)
        @test all(b′[f, f][i, j, j, i] == -1 for i in 1:2, j in 1:2)
    end
    # BraidingTensor converts to an ordinary TensorMap of the same space
    V = FermionicSpace(0 => 1, 1 => 1)
    b = BraidingTensor(V, V, false)
    @test space(TensorMap(b)) == space(b)
end

@testset "@planar" begin
    V = FermionicSpace(0 => 1, 1 => 1)
    # a simple planar network with an internal index
    X = randn(ComplexF64, V ⊗ V ← V)
    Y = randn(ComplexF64, V ← V ⊗ V)
    P = @planar P[-1 -2; -3 -4] := X[-1 -2; 1] * Y[1; -3 -4]
    @test space(P) == ((V ⊗ V) ← (V ⊗ V))
    # planar (no braiding) must agree with @tensor on the same network
    Pr = @tensor Pr[-1 -2; -3 -4] := X[-1 -2; 1] * Y[1; -3 -4]
    @test P ≈ Pr
end
