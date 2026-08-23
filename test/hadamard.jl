# hadamard.jl
# -----------
# Hadamard product（HadamardProduct.jl 接口）测试。对应文献（SciPost Phys.
# Core 7, 063 (2024) 附录 B.2-B.8）：两个 Grassmann Tensor 先做 outer product，
# 再把待融合指标移到一起，按 Eq. (B.7) 融合成对指标：
#
#     ξ^a · ξ^b： 00 → 1（sector even），10/01 → ξ（sector odd，两块相加），
#                 11 → 0（ξ²=0，直接丢弃）。
#
# 测试内容：
#   - Test 1-8：手动逐块参考验证主实现（纯 codomain 共享 / domain 共享 / 纯外积 /
#     空间不匹配抛错 / conjA 与显式 adjoint 等价 / expert 模式 α、β / @hadamard 宏 /
#     多共享腿 + 外积混合）；
#   - Test 9：独立参考实现 `hadamardproduct_ref`（块级外积 + 三重标量循环融合，
#     与主实现的 Strided 融合广播是两条独立执行路径）交叉验证主实现；
#   - Test 10：共享空间各 sector 维度不等（GT 定义下输入不一致）必须抛 SpaceMismatch。
import HadamardProduct as HD

@testset "hadamard product" begin
    Random.seed!(42)
    # 等维空间：PDF 的 GV 框架中每个 sector 的指数空间各 1 维；等维保证 (0,1)/(1,0)
    # 组合的逐点相乘维度对齐。
    V = FermionicSpace(0 => 2, 1 => 2)
    oneF = one(FermionParity)
    even = FermionParity(false)
    odd = FermionParity(true)

    # Z₂ 下张量 T 的块（codomain sectors..., domain sectors...）存在的条件：
    # ⊗(所有 uncoupled sectors) == one（codomain 与 domain 的耦合 sector 相等）
    blkvalid(cs, ds) = isone(reduce(⊗, (cs..., ds...); init=oneF))

    @testset "pure codomain shared A[i,j]*B[j,k]" begin
        A = rand(ComplexF64, V ⊗ V, one(V))
        B = rand(ComplexF64, V ⊗ V, one(V))
        C = hadamardproduct(A, (:i, :j), B, (:j, :k))
        @test space(C) == (V ⊗ V ⊗ V ← one(V))
        for si in sectors(V), sc in sectors(V), sk in sectors(V)
            blkvalid((si, sc, sk), ()) || continue # C 块不存在
            sC = (si, sc, sk)
            ref = zeros(ComplexF64, dim(V, si), dim(V, sc), dim(V, sk))
            if !sc.isodd
                if blkvalid((si, even), ()) && blkvalid((even, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] = A[(si, even)][i, m] * B[(even, sk)][m, k]
                    end
                end
            else
                if blkvalid((si, even), ()) && blkvalid((odd, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] += A[(si, even)][i, m] * B[(odd, sk)][m, k]
                    end
                end
                if blkvalid((si, odd), ()) && blkvalid((even, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] += A[(si, odd)][i, m] * B[(even, sk)][m, k]
                    end
                end
            end
            @test C[sC] ≈ ref
        end
    end

    @testset "domain shared leg A[i,k]*B[j,k]" begin
        A2 = rand(ComplexF64, V, V)   # (i) ← (k)
        B2 = rand(ComplexF64, V, V)   # (j) ← (k)
        C2 = hadamardproduct(A2, (:i, :k), B2, (:j, :k))
        @test space(C2) == (V ⊗ V ← V)
        for si in sectors(V), sj in sectors(V), sc in sectors(V)
            blkvalid((si, sj), (sc,)) || continue # C2 块：codomain=(i,j), domain=(k)
            sC = (si, sj, sc)
            ref = zeros(ComplexF64, dim(V, si), dim(V, sj), dim(V, sc))
            if !sc.isodd
                if blkvalid((si,), (even,)) && blkvalid((sj,), (even,))
                    for i in 1:dim(V, si), j in 1:dim(V, sj), m in 1:dim(V, sc)
                        ref[i, j, m] = A2[(si, even)][i, m] * B2[(sj, even)][j, m]
                    end
                end
            else
                if blkvalid((si,), (even,)) && blkvalid((sj,), (odd,))
                    for i in 1:dim(V, si), j in 1:dim(V, sj), m in 1:dim(V, sc)
                        ref[i, j, m] += A2[(si, even)][i, m] * B2[(sj, odd)][j, m]
                    end
                end
                if blkvalid((si,), (odd,)) && blkvalid((sj,), (even,))
                    for i in 1:dim(V, si), j in 1:dim(V, sj), m in 1:dim(V, sc)
                        ref[i, j, m] += A2[(si, odd)][i, m] * B2[(sj, even)][j, m]
                    end
                end
            end
            @test C2[sC] ≈ ref
        end
    end

    @testset "outer product A[i,j]*B[k,l]" begin
        A3 = rand(ComplexF64, V ⊗ V, one(V))
        B3 = rand(ComplexF64, V ⊗ V, one(V))
        C3 = hadamardproduct(A3, (:i, :j), B3, (:k, :l))
        @test space(C3) == (V ⊗ V ⊗ V ⊗ V ← one(V))
        for si in sectors(V), sj in sectors(V), sk in sectors(V), sl in sectors(V)
            blkvalid((si, sj, sk, sl), ()) || continue
            sC = (si, sj, sk, sl); sA = (si, sj); sB = (sk, sl)
            if blkvalid(sA, ()) && blkvalid(sB, ())
                for i in 1:size(C3[sC], 1), j in 1:size(C3[sC], 2),
                    k in 1:size(C3[sC], 3), l in 1:size(C3[sC], 4)
                    @test C3[sC][i, j, k, l] ≈ A3[sA][i, j] * B3[sB][k, l]
                end
            else
                @test iszero(C3[sC])
            end
        end
    end

    @testset "shared leg type mismatch throws" begin
        A = rand(ComplexF64, V ⊗ V, one(V))   # (i, j) ← ()，j 是 codomain 腿
        B4 = rand(ComplexF64, V, V)           # (k) ← (j)，j 是 domain 腿
        @test_throws SpaceMismatch hadamardproduct(A, (:i, :j), B4, (:k, :j))
    end

    @testset "space vs its dual space mismatch throws" begin
        # A 的共享腿 :k 是 codomain 空间 V；B 的共享腿 :k 是 domain 腿，其 primal
        # 空间为 dual(V)。space(A, :k) = V 与 space(B, :k) = dual(dual(V)) = V 值恰好
        # 相等，但一条是空间、另一条是它的对偶空间（腿类型也不同），必须报错。
        A9 = rand(ComplexF64, V ⊗ V, one(V))   # (i, k) ← ()，k 是 codomain V
        B9 = rand(ComplexF64, V, dual(V))      # (m) ← (k)，k 是 domain，primal 空间 dual(V)
        @test_throws SpaceMismatch hadamardproduct(A9, (:i, :k), B9, (:m, :k))
    end

    @testset "different shared spaces throws" begin
        # A 的共享腿 :k 是 codomain 空间 V（0=>2, 1=>2），B 的共享腿 :k 是 codomain
        # 空间 W（0=>3, 1=>3）：腿类型相同但空间不同（维度不相等），指标"不完全
        # 相同"，必须报 SpaceMismatch。
        W = FermionicSpace(0 => 3, 1 => 3)
        A10 = rand(ComplexF64, V ⊗ V, one(V))   # (i, k) ← ()，k: codomain V
        B10 = rand(ComplexF64, W ⊗ V, one(V))   # (k, j) ← ()，k: codomain W
        @test_throws SpaceMismatch hadamardproduct(A10, (:i, :k), B10, (:k, :j))
    end

    @testset "conjA == explicit adjoint" begin
        # A5: (i,j) ← (l)，共享腿 l 为 domain；取 conj 后 l 在 adjoint(A5) 中变为
        # codomain，因此 B5 的共享腿 l 必须同为 codomain 才一致（即与显式 adjoint
        # 路径逐位等价）。
        A5 = rand(ComplexF64, V ⊗ V, V)      # (i,j) ← (l)
        B5 = rand(ComplexF64, V ⊗ V, one(V)) # (o, l) 均为 codomain
        pA5 = ((1, 2), (3,))    # outer=(i,j)，shared=(l)
        pB5 = ((2,), (1,))      # shared=(l 位置 2)，outer=(o 位置 1)
        pAB5 = ((1, 2, 3, 4), ())  # natural = (i, j, l, o)，C 保持该序
        C_adj = hadamardproduct(adjoint(A5), ((2, 3), (1,)), false, B5, pB5, false, pAB5)
        C_conj = hadamardproduct(A5, pA5, true, B5, pB5, false, pAB5)
        @test space(C_adj) == space(C_conj)
        @test C_adj == C_conj
    end

    @testset "expert mode α=2, β=3" begin
        A = rand(ComplexF64, V ⊗ V, one(V))
        B = rand(ComplexF64, V ⊗ V, one(V))
        C6 = rand(ComplexF64, V ⊗ V ⊗ V, one(V))
        old6 = Dict{Tuple{Any,Any,Any},Any}()
        for si in sectors(V), sc in sectors(V), sk in sectors(V)
            blkvalid((si, sc, sk), ()) || continue
            old6[(si, sc, sk)] = copy(C6[(si, sc, sk)])
        end
        HD.hadamardproduct!(C6, A, ((1,), (2,)), false, B, ((1,), (2,)), false,
                            ((1, 2, 3), ()), 2.0, 3.0)
        for si in sectors(V), sc in sectors(V), sk in sectors(V)
            blkvalid((si, sc, sk), ()) || continue
            sC = (si, sc, sk)
            ref = zeros(ComplexF64, dim(V, si), dim(V, sc), dim(V, sk))
            if !sc.isodd
                if blkvalid((si, even), ()) && blkvalid((even, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] = A[(si, even)][i, m] * B[(even, sk)][m, k]
                    end
                end
            else
                if blkvalid((si, even), ()) && blkvalid((odd, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] += A[(si, even)][i, m] * B[(odd, sk)][m, k]
                    end
                end
                if blkvalid((si, odd), ()) && blkvalid((even, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] += A[(si, odd)][i, m] * B[(even, sk)][m, k]
                    end
                end
            end
            @test C6[sC] ≈ 2.0 .* ref .+ 3.0 .* old6[sC]
        end
    end

    @testset "@hadamard macro" begin
        A7 = rand(Float64, V ⊗ V, one(V))
        B7 = rand(Float64, V ⊗ V, one(V))
        @hadamard C7[i, j, k] := A7[i, j] * B7[j, k]
        @test space(C7) == (V ⊗ V ⊗ V ← one(V))
        for si in sectors(V), sc in sectors(V), sk in sectors(V)
            blkvalid((si, sc, sk), ()) || continue
            sC = (si, sc, sk)
            ref = zeros(Float64, dim(V, si), dim(V, sc), dim(V, sk))
            if !sc.isodd
                if blkvalid((si, even), ()) && blkvalid((even, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] = A7[(si, even)][i, m] * B7[(even, sk)][m, k]
                    end
                end
            else
                if blkvalid((si, even), ()) && blkvalid((odd, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] += A7[(si, even)][i, m] * B7[(odd, sk)][m, k]
                    end
                end
                if blkvalid((si, odd), ()) && blkvalid((even, sk), ())
                    for i in 1:dim(V, si), m in 1:dim(V, sc), k in 1:dim(V, sk)
                        ref[i, m, k] += A7[(si, odd)][i, m] * B7[(even, sk)][m, k]
                    end
                end
            end
            @test C7[sC] ≈ ref
        end
    end

    @testset "multi shared + outer (domain shared)" begin
        # A8: (i,j) ← (k,l)；B8: (m) ← (k,l)。共享 (k,l) 同为 domain；C 的腿 =
        # (i,j,m) cod, (k,l) dom
        A8 = rand(ComplexF64, V ⊗ V, V ⊗ V)
        B8 = rand(ComplexF64, V, V ⊗ V)
        C8 = hadamardproduct(A8, (:i, :j, :k, :l), B8, (:m, :k, :l))
        @test space(C8) == (V ⊗ V ⊗ V ← V ⊗ V)
        # 组合枚举：k 腿 (a₁,b₁) × l 腿 (a₂,b₂)
        abopts(c) = c.isodd ? ((even, odd), (odd, even)) : ((even, even),)
        for si in sectors(V), sj in sectors(V), sm in sectors(V),
            sc₁ in sectors(V), sc₂ in sectors(V)
            blkvalid((si, sj, sm), (sc₁, sc₂)) || continue
            sC = (si, sj, sm, sc₁, sc₂)
            ref = zeros(ComplexF64, dim(V, si), dim(V, sj), dim(V, sm),
                        dim(V, sc₁), dim(V, sc₂))
            for (a₁, b₁) in abopts(sc₁), (a₂, b₂) in abopts(sc₂)
                # A8 块 [si, sj, a₁, a₂]（codomain=(i,j), domain=(k,l)）
                blkvalid((si, sj), (a₁, a₂)) || continue
                blkvalid((sm,), (b₁, b₂)) || continue
                for i in 1:dim(V, si), j in 1:dim(V, sj), m in 1:dim(V, sm),
                    m₁ in 1:dim(V, sc₁), m₂ in 1:dim(V, sc₂)
                    ref[i, j, m, m₁, m₂] += A8[(si, sj, a₁, a₂)][i, j, m₁, m₂] *
                                            B8[(sm, b₁, b₂)][m, m₁, m₂]
                end
            end
            @test C8[sC] ≈ ref
        end
    end

    # ==========================================================================
    # Test 9: 参考实现（笨方法：块级外积 + 融合）交叉验证主实现
    # ==========================================================================
    # 参考实现步骤（与主实现的 Strided 融合广播完全独立的执行路径）：
    #   1. 对 C 的每个融合树块，按 B.7 枚举共享腿 sector 组合 (aₖ, bₖ)；
    #   2. A/B 块用 permutedims（真实拷贝）重排到 natural 序 (oA..., sh..., oB...)；
    #   3. 融合 = 三重标量循环 C[i, m, k] += A[i, m] * B[m, k]（共享下标 m 对齐）。
    # 复用的仅是最低限度的数学辅助（腿类型/空间推断、B.7 组合枚举、融合树反解）。
    function hadamardproduct_ref(A, pA, B, pB, pAB)
        nO = GrassmannTensors._numout(pA)
        nS = GrassmannTensors._numin(pA)
        nB = GrassmannTensors._numin(pB)
        N = nO + nS + nB

        # natural 序中每条腿的类型（codomain/domain）与空间（domain 存 primal）
        nattype = Vector{Bool}(undef, N)
        natspace = Vector{GrassmannTensors.FermionicSpace}(undef, N)
        for n in 1:N
            T, i = GrassmannTensors._natural_leg(A, pA, B, pB, n)
            nattype[n] = i <= numout(T)
            natspace[n] = nattype[n] ? space(T, i) : dual(space(T, i))
        end
        codspaces = [natspace[n] for n in 1:N if nattype[n]]
        domspaces = [natspace[n] for n in 1:N if !nattype[n]]
        W = ProductSpace{length(codspaces)}(tuple(codspaces...)) ←
            ProductSpace{length(domspaces)}(tuple(domspaces...))
        C = GrassmannTensors.zerovector!(similar(A, eltype(A), W))
        C_struct = GrassmannTensors.fusionblockstructure(C)
        A_struct = GrassmannTensors.fusionblockstructure(A)
        B_struct = GrassmannTensors.fusionblockstructure(B)
        N₁C, N₂C = numout(C), numin(C)

        # natural 位置 → C 的 all-index 位置（C 的 all-index 序 = (codomain..., domain...)）
        Cpos2nat = Vector{Int}(undef, N)
        codpos, dompos = 0, 0
        for k in 1:N
            n = HD.linearize(pAB)[k]
            if nattype[n]
                codpos += 1
                Cpos2nat[codpos] = n
            else
                dompos += 1
                Cpos2nat[N₁C + dompos] = n
            end
        end
        nat2C = Vector{Int}(undef, N)
        for (m, n) in enumerate(Cpos2nat)
            nat2C[n] = m
        end
        permC = ntuple(n -> nat2C[n], N)

        # 张量 all-index 位置 → natural 位置
        Apos2nat = Vector{Int}(undef, numind(A))
        for j in 1:nO; Apos2nat[pA[1][j]] = j; end
        for k in 1:nS; Apos2nat[pA[2][k]] = nO + k; end
        Bpos2nat = Vector{Int}(undef, numind(B))
        for k in 1:nS; Bpos2nat[pB[1][k]] = nO + k; end
        for j in 1:nB; Bpos2nat[pB[2][j]] = nO + nS + j; end

        N₁A, N₂A = numout(A), numin(A)
        N₁B, N₂B = numout(B), numin(B)
        isdualA₁ = map(isdual, codomain(A).spaces)
        isdualA₂ = map(isdual, domain(A).spaces)
        isdualB₁ = map(isdual, codomain(B).spaces)
        isdualB₂ = map(isdual, domain(B).spaces)
        shspaces = ntuple(k -> space(A, pA[2][k]), nS)
        permA = HD.linearize(pA)
        permB = HD.linearize(pB)

        for (fC₁, fC₂) in C_struct.fusiontreelist
            sectorC(n) = nat2C[n] <= N₁C ? fC₁.uncoupled[nat2C[n]] :
                                           fC₂.uncoupled[nat2C[n] - N₁C]
            opts = ntuple(k -> GrassmannTensors._shared_ab_options(sectorC(nO + k)), nS)
            for combo in Iterators.product(opts...)
                as = ntuple(k -> combo[k][1], nS)
                bs = ntuple(k -> combo[k][2], nS)
                # 逐点相乘要求共享腿两侧同 sector 的块维度对齐。GT 定义下被融合的
                # 指标必须完全一致，若不对齐说明输入空间不一致，直接抛错（与主实现
                # 行为一致，而非静默跳过）。
                for k in 1:nS
                    dA = blockdim(shspaces[k], as[k])
                    dB = blockdim(shspaces[k], bs[k])
                    dA == dB || throw(GrassmannTensors.SpaceMismatch(
                        "shared index $k: block dims of fused sectors $(as[k]) (A) and $(bs[k]) (B) do not align: $dA ≠ $dB in space $(shspaces[k])"))
                end
                iA, fA₁, fA₂ = GrassmannTensors._hadamard_fusiontree(
                    A_struct, fC₁, fC₂, Apos2nat, nat2C, N₁C, N₁A, N₂A,
                    nO, nS, isdualA₁, isdualA₂, as)
                iA == 0 && continue
                iB, fB₁, fB₂ = GrassmannTensors._hadamard_fusiontree(
                    B_struct, fC₁, fC₂, Bpos2nat, nat2C, N₁C, N₁B, N₂B,
                    nO, nS, isdualB₁, isdualB₂, bs)
                iB == 0 && continue
                # 外积（permutedims 拷贝）+ 融合（三重标量循环）。
                # 用 collect 转成普通 Array，走纯 Base 路径，避开 Strided 的 lazy 视图。
                Ablk = permutedims(collect(A[fA₁, fA₂]), permA) # (oA..., sh...)
                Bblk = permutedims(collect(B[fB₁, fB₂]), permB) # (sh..., oB...)
                Cblk = permutedims(collect(C[fC₁, fC₂]), permC) # (oA..., sh..., oB...)
                ΠoA = prod(size(Ablk, i) for i in 1:nO)
                Πs = prod(size(Ablk, nO + i) for i in 1:nS)
                ΠoB = prod(size(Bblk, nS + i) for i in 1:nB)
                Am = reshape(Ablk, (ΠoA, Πs))
                Bm = reshape(Bblk, (Πs, ΠoB))
                Cm = reshape(Cblk, (ΠoA, Πs, ΠoB))
                for i in 1:ΠoA, m in 1:Πs, k in 1:ΠoB
                    Cm[i, m, k] += Am[i, m] * Bm[m, k]
                end
                C[fC₁, fC₂] .= permutedims(Cblk, invperm(permC))
            end
        end
        return C
    end

    blocks_approx(C1, C2; rtol=1e-10, atol=1e-12) = begin
        space(C1) == space(C2) || return false
        for (f₁, f₂) in fusiontrees(C1)
            isapprox(C1[f₁, f₂], C2[f₁, f₂]; rtol=rtol, atol=atol) || return false
        end
        return true
    end

    # 等维空间（even/odd sector 维度相同，满足 GT 定义下共享腿各 sector 等维的
    # 要求；维度 > 1 可验证奇 sector 下 (0,1)/(1,0) 两块相加的逐点对齐）
    Vr = FermionicSpace(0 => 3, 1 => 3)
    Random.seed!(2024)

    @testset "ref cross-check: codomain shared" begin
        for trial in 1:3
            Ar = rand(ComplexF64, Vr ⊗ Vr, one(Vr))
            Br = rand(ComplexF64, Vr ⊗ Vr, one(Vr))
            pA, pB, pAB = HD.hadamard_indices((:i, :j), (:j, :k), (:i, :j, :k))
            C1r = hadamardproduct(Ar, pA, false, Br, pB, false, pAB)
            C2r = hadamardproduct_ref(Ar, pA, Br, pB, pAB)
            @test blocks_approx(C1r, C2r)
        end
    end

    @testset "ref cross-check: domain shared" begin
        for trial in 1:3
            Ar = rand(ComplexF64, Vr, Vr)
            Br = rand(ComplexF64, Vr, Vr)
            pA, pB, pAB = HD.hadamard_indices((:i, :k), (:j, :k), (:i, :j, :k))
            C1r = hadamardproduct(Ar, pA, false, Br, pB, false, pAB)
            C2r = hadamardproduct_ref(Ar, pA, Br, pB, pAB)
            @test blocks_approx(C1r, C2r)
        end
    end

    @testset "ref cross-check: multi shared + outer" begin
        for trial in 1:3
            Ar = rand(ComplexF64, Vr ⊗ Vr, Vr ⊗ Vr)
            Br = rand(ComplexF64, Vr, Vr ⊗ Vr)
            pA, pB, pAB = HD.hadamard_indices((:i, :j, :k, :l), (:m, :k, :l),
                                              (:i, :j, :m, :k, :l))
            C1r = hadamardproduct(Ar, pA, false, Br, pB, false, pAB)
            C2r = hadamardproduct_ref(Ar, pA, Br, pB, pAB)
            @test blocks_approx(C1r, C2r)
        end
    end

    @testset "ref cross-check: permuted outer order, small dims" begin
        # 共享腿 l 在两侧必须同型同向：A/B 中 l 都是 domain 腿（primal 空间 Vd）。
        Vd = FermionicSpace(0 => 1, 1 => 1)
        for trial in 1:3
            Ar = rand(ComplexF64, Vd ⊗ Vd, Vd)      # (i, k) ← (l)，共享 l：domain
            Br = rand(ComplexF64, Vd, Vd ⊗ Vd)      # (m) ← (l, n)，共享 l：domain
            pA, pB, pAB = HD.hadamard_indices((:i, :k, :l), (:m, :l, :n),
                                              (:m, :i, :n, :k, :l))
            C1r = hadamardproduct(Ar, pA, false, Br, pB, false, pAB)
            C2r = hadamardproduct_ref(Ar, pA, Br, pB, pAB)
            @test blocks_approx(C1r, C2r)
        end
    end

    @testset "unequal sector dims on shared leg throws" begin
        # GT 定义下被融合的指标必须完全一致。共享空间 even/odd sector 维度不等时，
        # 奇 sector 融合组合 (0,1)/(1,0) 两侧块维度无法对齐（3 ≠ 2），说明输入空间
        # 不一致，主实现与参考实现都必须抛 SpaceMismatch 而非静默跳过。
        Vun = FermionicSpace(0 => 3, 1 => 2)
        Aun = rand(ComplexF64, Vun ⊗ Vun, one(Vun))
        Bun = rand(ComplexF64, Vun ⊗ Vun, one(Vun))
        pun = HD.hadamard_indices((:i, :j), (:j, :k), (:i, :j, :k))
        @test_throws SpaceMismatch hadamardproduct(Aun, pun[1], false, Bun,
                                                   pun[2], false, pun[3])
        @test_throws SpaceMismatch hadamardproduct_ref(Aun, pun[1], Bun, pun[2], pun[3])
    end
end
