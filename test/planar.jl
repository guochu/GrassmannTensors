# Planar 运算与网络测试：参照 TensorKit test/tensors/planar.jl。
# 注意：FermionParity 为费米型编织，planar（transpose 路径、无 twist）与
# tensor（permute 路径、带 twist）只在无奇偶交叉/无 twist 闭环时一致。
# 因此与 tensoradd!/tensortrace!/tensorcontract! 的对照测试使用偶数分次空间
# （等价于平凡对称性）；混合分次空间的平面性质在 braidingtensor.jl 中验证。
using GrassmannTensors: planaradd!, planartrace!, planarcontract!
using TensorOperations: tensoradd!, tensortrace!, tensorcontract!

@testset "planar methods" begin
    V2 = FermionicSpace(0 => 2); V3 = FermionicSpace(0 => 3)
    V4 = FermionicSpace(0 => 4); V5 = FermionicSpace(0 => 5); V6 = FermionicSpace(0 => 6)

    @testset "planaradd" begin
        A = randn(ComplexF64, V2 ⊗ V3 ← V6 ⊗ V5 ⊗ V4)
        C = randn(ComplexF64, V5' ⊗ V6' ← V4 ⊗ V3' ⊗ V2')
        p = ((4, 3), (5, 2, 1))
        C1 = copy(C)
        planaradd!(C1, A, p, true, true)
        C2 = copy(C)
        tensoradd!(C2, A, p, false, true, true)
        @test C1 ≈ C2
    end

    @testset "planartrace" begin
        A = randn(ComplexF64, V2 ⊗ V3 ← V2 ⊗ V5 ⊗ V4)
        C = randn(ComplexF64, V5' ⊗ V3 ← V4)
        p = ((4, 2), (5,))
        q = ((1,), (3,))
        C1 = copy(C)
        planartrace!(C1, A, p, q, true, true)
        C2 = copy(C)
        tensortrace!(C2, A, p, q, false, true, true)
        @test C1 ≈ C2
    end

    @testset "planarcontract" begin
        A = randn(ComplexF64, V2 ⊗ V3 ← V2 ⊗ V5 ⊗ V4)
        B = randn(ComplexF64, V2 ⊗ V4 ← V4 ⊗ V3)
        C = randn(ComplexF64, V5' ⊗ V2' ⊗ V2 ← V2' ⊗ V4)
        pA = ((1, 3, 4), (5, 2))
        pB = ((2, 4), (1, 3))
        pAB = ((3, 2, 1), (4, 5))
        C1 = copy(C)
        planarcontract!(C1, A, pA, B, pB, pAB, true, true)
        C2 = copy(C)
        tensorcontract!(C2, A, pA, false, B, pB, false, pAB, true, true)
        @test C1 ≈ C2
    end
end

@testset "@planar" begin
    @testset "contractcheck" begin
        V = FermionicSpace(0 => 1, 1 => 1)
        A = randn(ComplexF64, V ⊗ V ← V)
        B = randn(ComplexF64, V ⊗ V ← V')
        @tensor C1[i j; k l] := A[i j; m] * B[k l; m]
        @tensor contractcheck = true C2[i j; k l] := A[i j; m] * B[k l; m]
        @test C1 ≈ C2
        B2 = randn(ComplexF64, V ⊗ V ← V) # m 的 dual 不对
        @test_throws SpaceMismatch begin
            @tensor contractcheck = true C3[i j; k l] := A[i j; m] * B2[k l; m]
        end

        A = randn(ComplexF64, V ← V ⊗ V)
        B = randn(ComplexF64, V ⊗ V ← V)
        @planar C1[i; j] := A[i; k l] * τ[k l; m n] * B[m n; j]
        @planar contractcheck = true C2[i; j] := A[i; k l] * τ[k l; m n] * B[m n; j]
        @test C1 ≈ C2
        @test_throws SpaceMismatch begin
            @planar contractcheck = true C3[i; j] := A[i; k l] * τ[k l; m n] * B[n j; m]
        end
    end

    @testset "MPS networks" begin
        # ∂AC：无 twist 闭环，混合分次空间下 planar 与 tensor 一致
        Vp = FermionicSpace(0 => 1, 1 => 1)
        Vmps = FermionicSpace(0 => 3, 1 => 2)
        Vmpo = FermionicSpace(0 => 2, 1 => 1)
        x = randn(ComplexF64, Vmps ⊗ Vp ← Vmps)
        O = randn(ComplexF64, Vmpo ⊗ Vp ← Vp ⊗ Vmpo)
        GL = randn(ComplexF64, Vmps ⊗ Vmpo' ← Vmps)
        GR = randn(ComplexF64, Vmps ⊗ Vmpo ← Vmps)
        y = @tensor y[-1 -2; -3] := GL[-1 2; 1] * x[1 3; 4] * O[2 -2; 3 5] * GR[4 5; -3]
        y′ = @planar y′[-1 -2; -3] := GL[-1 2; 1] * x[1 3; 4] * O[2 -2; 3 5] * GR[4 5; -3]
        @test y ≈ y′

        # 转移矩阵：conj(x) 形成闭环，用偶数分次空间（twist 为单位元）
        Vpe = FermionicSpace(0 => 2)
        Vmpse = FermionicSpace(0 => 3)
        xe = randn(ComplexF64, Vmpse ⊗ Vpe ← Vmpse)
        ve = randn(ComplexF64, Vmpse ← Vmpse)
        ρ = @tensor ρ[-1; -2] := xe[-1 2; 1] * conj(xe[-2 2; 3]) * ve[1; 3]
        ρ′ = @planar ρ′[-1; -2] := xe[-1 2; 1] * conj(xe[-2 2; 3]) * ve[1; 3]
        @test ρ ≈ ρ′

        # 含 τ 的转移矩阵：τ 显式编码编织后，混合分次空间下 @plansor 与 @tensor 一致
        ρ2 = @tensor ρ2[-1 -2; -3] := GL[1 -2; 3] * x[3 2; -3] * conj(x[1 2; -1])
        ρ3 = @plansor ρ3[-1 -2; -3] := GL[1 2; 4] * x[4 5; -3] * τ[2 3; 5 -2] * conj(x[1 3; -1])
        @test ρ2 ≈ ρ3
        ρ2′ = @planar ρ2′[-1 -2; -3] := GL[1 2; 4] * x[4 5; -3] * τ[2 3; 5 -2] * conj(x[1 3; -1])
        @test ρ2′ ≈ ρ3
    end

    @testset "Issue 93" begin
        # 同一标量网络的所有等价写法（偶数分次空间，编织符号全为 +1）
        V1 = FermionicSpace(0 => 2)
        V2 = FermionicSpace(0 => 3)
        t1 = randn(ComplexF64, V1 ← V2)
        t2 = randn(ComplexF64, V2 ← V1)
        tr1 = @planar opt = true t1[a; b] * t2[b; a] / 2
        tr2 = @planar opt = true t1[d; a] * t2[b; c] * 1 / 2 * τ[c b; a d]
        tr3 = @planar opt = true t1[d; a] * t2[b; c] * τ[a c; d b] / 2
        tr4 = @planar opt = true t1[f; a] * 1 / 2 * t2[c; d] * τ[d b; c e] * τ[e b; a f]
        tr5 = @planar opt = true t1[f; a] * t2[c; d] / 2 * τ[d b; c e] * τ[a e; f b]
        tr6 = @planar opt = true t1[f; a] * t2[c; d] * τ[c d; e b] / 2 * τ[e b; a f]
        tr7 = @planar opt = true t1[f; a] * t2[c; d] * (τ[c d; e b] * τ[a e; f b] / 2)
        @test tr1 ≈ tr2 ≈ tr3 ≈ tr4 ≈ tr5 ≈ tr6 ≈ tr7

        tr1 = @plansor opt = true t1[a; b] * t2[b; a] / 2
        tr2 = @plansor opt = true t1[d; a] * t2[b; c] * 1 / 2 * τ[c b; a d]
        tr3 = @plansor opt = true t1[d; a] * t2[b; c] * τ[a c; d b] / 2
        tr4 = @plansor opt = true t1[f; a] * 1 / 2 * t2[c; d] * τ[d b; c e] * τ[e b; a f]
        tr5 = @plansor opt = true t1[f; a] * t2[c; d] / 2 * τ[d b; c e] * τ[a e; f b]
        tr6 = @plansor opt = true t1[f; a] * t2[c; d] * τ[c d; e b] / 2 * τ[e b; a f]
        tr7 = @plansor opt = true t1[f; a] * t2[c; d] * (τ[c d; e b] * τ[a e; f b] / 2)
        @test tr1 ≈ tr2 ≈ tr3 ≈ tr4 ≈ tr5 ≈ tr6 ≈ tr7
    end

    @testset "Issue 262" begin
        # 加法的顺序无关性
        V = FermionicSpace(0 => 1, 1 => 1)
        A = randn(ComplexF64, V ← V)
        B = randn(ComplexF64, V ← V')
        C = randn(ComplexF64, V' ← V)
        D1 = @planar D1[i; j] := A[i; j] + B[i; k] * C[k; j]
        D2 = @planar D2[i; j] := B[i; k] * C[k; j] + A[i; j]
        @test D1 ≈ D2
    end
end
