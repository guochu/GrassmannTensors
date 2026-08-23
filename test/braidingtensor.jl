# BraidingTensor 平面性质测试：参照 TensorKit test/tensors/braidingtensor.jl。
# FermionParity 为费米型编织：planar 轮换（转置整条腿循环）不引入编织符号，
# 用 `transpose`（仅 F-symbol）作为参照；`τ` 网络的收缩一致性用 `braid` 参照。
@testset "BraidingTensor planar" begin
    for V in (FermionicSpace(0 => 1, 1 => 1),
              FermionicSpace(2 => 3, 1 => 2))
        T = ComplexF64
        t = randn(T, V ⊗ V' ⊗ V' ⊗ V ← V ⊗ V')
        @testset "BraidingTensor planar contractions with $V" begin
            @testset "planaradd! with BraidingTensor" begin
                b = BraidingTensor(V, V')
                bb = TensorMap(b)
                # planar 腿循环 (cod1, cod2, dom2, dom1) 的四个轮换；
                # 用 transpose（F-symbols only）作参照，permute 会带编织符号
                @planar t1[-1 -2; -3 -4] := b[-1 -2; -3 -4]
                @test t1 ≈ bb
                @planar t2[-1 -2; -3 -4] := b[-3 -1; -4 -2]
                @test t2 ≈ transpose(bb, ((2, 4), (1, 3)))
                @planar t3[-1 -2; -3 -4] := b[-4 -3; -2 -1]
                @test t3 ≈ transpose(bb, ((4, 3), (2, 1)))
                @planar t4[-1 -2; -3 -4] := b[-2 -4; -1 -3]
                @test t4 ≈ transpose(bb, ((3, 1), (4, 2)))
                # adjoint BraidingTensor
                ba = b'
                @planar t5[-1 -2; -3 -4] := ba[-1 -2; -3 -4]
                @test t5 ≈ TensorMap(ba)
            end

            @testset "τ as left factor" begin
                # 最前两腿编织：τ 等价于 braid（FermionParity 下 braid == permute）
                ττ = TensorMap(BraidingTensor(V, V'))
                @planar t1[-1 -2 -3 -4; -5 -6] := τ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
                @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
                @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -2 -1] * t[1 2 -3 -4; -5 -6]
                @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-2 2; -1 1] * t[1 2 -3 -4; -5 -6]
                @test t1 ≈ braid(t, ((2, 1, 3, 4), (5, 6)), (1, 2, 3, 4, 5, 6))
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4

                # 最后两腿编织
                ττ = TensorMap(BraidingTensor(V', V))
                @planar t1[-1 -2 -3 -4; -5 -6] := τ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
                @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
                @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -4 -3] * t[-1 -2 1 2; -5 -6]
                @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-4 2; -3 1] * t[-1 -2 1 2; -5 -6]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4
            end

            @testset "τ as right factor" begin
                # 域腿全部为 τ 的输入
                ττ = TensorMap(BraidingTensor(V', V))
                @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[1 2; -5 -6]
                @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ[1 2; -5 -6]
                @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[-6 -5; 2 1]
                @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[2 -6; 1 -5]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4

                # τ'（adjoint BraidingTensor）
                ττ = TensorMap(BraidingTensor(V, V'))
                @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[1 2; -5 -6]
                @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ'[1 2; -5 -6]
                @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[-6 -5; 2 1]
                @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[2 -6; 1 -5]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4
            end

            @testset "τ with fully contracted output" begin
                # 标量输出（(0,2) 张量）
                ττ = TensorMap(BraidingTensor(V', V))
                @planar t1[(); (-1, -2)] := τ[2 1; 3 4] * t[1 2 3 4; -1 -2]
                @planar t2[(); (-1, -2)] := ττ[2 1; 3 4] * t[1 2 3 4; -1 -2]
                @planar t3[(); (-1, -2)] := τ[4 3; 1 2] * t[1 2 3 4; -1 -2]
                @planar t4[(); (-1, -2)] := τ'[1 4; 2 3] * t[1 2 3 4; -1 -2]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4

                # rank-1 输出
                ττ = TensorMap(BraidingTensor(V, V))
                @planar t1[-1; -2] := τ[2 1; 3 4] * t[-1 1 2 3; -2 4]
                @planar t2[-1; -2] := ττ[2 1; 3 4] * t[-1 1 2 3; -2 4]
                @planar t3[-1; -2] := τ[4 3; 1 2] * t[-1 1 2 3; -2 4]
                @planar t4[-1; -2] := τ'[1 4; 2 3] * t[-1 1 2 3; -2 4]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4
            end

            @testset "τ with one open codomain leg" begin
                ττ = TensorMap(BraidingTensor(V, V'))
                @planar t1[-1 -2; -3 -4] := τ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
                @planar t2[-1 -2; -3 -4] := ττ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
                @planar t3[-1 -2; -3 -4] := τ[2 1; 3 -1] * t[1 2 3 -2; -3 -4]
                @planar t4[-1 -2; -3 -4] := τ'[3 2; -1 1] * t[1 2 3 -2; -3 -4]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4
            end

            @testset "τ as right factor with open domain leg" begin
                ττ = TensorMap(BraidingTensor(V', V))
                @planar t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[1 2; -4 3]
                @planar t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ[1 2; -4 3]
                @planar t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[3 -4; 2 1]
                @planar t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[2 3; 1 -4]
                @test t1 ≈ t2
                @test t1 ≈ t3
                @test t1 ≈ t4
            end

            @testset "BraidingTensor × BraidingTensor" begin
                # b1 的 domain == b2 的 codomain == V⊗V'，straight-through planar 收缩
                b1 = BraidingTensor(V, V')
                b2 = BraidingTensor(V', V)
                bb1 = TensorMap(b1)
                bb2 = TensorMap(b2)
                @planar t1[-1 -2; -3 -4] := b1[-1 -2; 1 2] * b2[1 2; -3 -4]
                @planar t2[-1 -2; -3 -4] := bb1[-1 -2; 1 2] * bb2[1 2; -3 -4]
                @test t1 ≈ t2
            end
        end
    end
end

@testset "BraidingTensor space checks" begin
    V = FermionicSpace(2 => 3, 1 => 2)
    W = V ⊗ V' ← V' ⊗ V
    b = BraidingTensor(W)
    @test space(b) == W
    @test codomain(b) == codomain(W)
    @test domain(b) == domain(W)
    @test adjoint(b) isa BraidingTensor
    # 非平面空间无法定义编织
    W2 = reverse(codomain(W)) ← domain(W)
    @test_throws SpaceMismatch BraidingTensor(W2)
end
