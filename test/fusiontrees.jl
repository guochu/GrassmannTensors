# FusionTree 测试：参照 TensorKit test/symmetries/singletree.jl，适配 Z₂ 费米子。
# FermionParity 为 UniqueFusion：每个 uncoupled 组合唯一耦合、Nsymbol == 1，
# 故 GT 只实现 split / permute（无 join / multi_Fmove / insertat / merge 等）。
using GrassmannTensors: split, couple, fusiontreetype, SectorMismatch

@testset "FusionTree" begin
    e = FermionParity(false)
    o = FermionParity(true)

    @testset "construction and properties" begin
        # 显式构造与字段
        f = FusionTree((e, o), o)
        @test sectortype(f) == FermionParity
        @test length(f) == 2
        @test f.uncoupled == (e, o)
        @test f.coupled == o
        @test f.isdual == (false, false)
        # 默认参数：coupled = unit, isdual 全 false
        f2 = FusionTree((e, o))
        @test f2.coupled == e
        @test f2.isdual == (false, false)
        # 显式 isdual
        f3 = FusionTree((e, o), o, (true, false))
        @test f3.isdual == (true, false)
        # 空树
        f0 = FusionTree((), e)
        @test length(f0) == 0
        @test f0.uncoupled == ()
        @test f0.isdual == ()
        # 类型工具
        @test fusiontreetype(3) === FusionTree{3}
        @test length(FusionTree{3}) == 3
        # 相等性
        @test FusionTree((e, o), o) == FusionTree((e, o), o)
        @test FusionTree((e, o), o) != FusionTree((e, o), e)
        @test FusionTree((e, o), o) != FusionTree((o, e), o)
        @test FusionTree((e, o), o) != FusionTree((e, o), o, (true, false))
        @test FusionTree((e, o), o) != FusionTree((e, o, e), o)
        # 哈希（相等树同哈希，可作 Dict 键）
        fa = FusionTree((e, o), o)
        fb = FusionTree((e, o), o)
        @test fa == fb && hash(fa) == hash(fb)
        d = Dict(fa => 1)
        @test d[fb] == 1
    end

    @testset "fusiontrees iterator" begin
        # 单腿：仅耦合匹配时返回
        @test length(fusiontrees((e,), e)) == 1
        @test isempty(fusiontrees((e,), o))
        @test length(fusiontrees((o,), o)) == 1
        @test isempty(fusiontrees((o,), e))
        # 多腿
        @test length(fusiontrees((e, o, e), o)) == 1   # e⊗o⊗e = o
        @test isempty(fusiontrees((e, o, e), e))
        # 组合枚举：uncoupleds 的笛卡尔积中耦合到 coupled 的所有组合
        ts = fusiontrees(((e, o), (e, o)), e)          # (e,e) 与 (o,o) 耦合到 e
        @test length(ts) == 2
        @test all(f.coupled == e for f in ts)
        @test Set(f.uncoupled for f in ts) == Set(((e, e), (o, o)))
        ts = fusiontrees(((e, o), (e, o)), o)          # (e,o) 与 (o,e) 耦合到 o
        @test length(ts) == 2
        @test Set(f.uncoupled for f in ts) == Set(((e, o), (o, e)))
        # isdual 标志保留
        f = only(fusiontrees((e, o), o, (true, false)))
        @test f.isdual == (true, false)
        # 计数一致性
        out = ((e, o), (e, o), (e, e))
        for c in (e, o)
            ts = fusiontrees(out, c)
            @test length(ts) == count(_ -> true, fusiontrees(out, c))
        end
        # couple 与 ⊗ 一致
        @test couple(()) == e
        @test couple((o,)) == o
        @test couple((e, o, o)) == e
        @test couple((e, o, o)) == ⊗(e, o, o)
    end

    @testset "split" begin
        N = 6
        for trial in 1:10
            uncoupled = ntuple(_ -> rand((e, o)), N)
            coupled = couple(uncoupled)
            isdual = ntuple(_ -> rand(Bool), N)
            f = FusionTree(uncoupled, coupled, isdual)
            for i in 0:N
                f₁, f₂ = split(f, i)
                @test length(f₁) == i
                @test length(f₂) == N - i + 1
                @test f₁.coupled == couple(f₁.uncoupled)
                @test f₂.coupled == coupled
                if i == 0
                    @test f₁.uncoupled == ()
                    @test f₂.uncoupled == (e, uncoupled...)
                    @test f₂.isdual == (false, isdual...)
                elseif i == N
                    @test f₁ == f
                    @test f₂.uncoupled == (coupled,)
                    @test f₂.isdual == (false,)
                else
                    @test f₁.uncoupled == uncoupled[1:i]
                    @test f₁.isdual == isdual[1:i]
                    @test f₂.uncoupled[1] == couple(uncoupled[1:i])
                    @test f₂.uncoupled[2:end] == uncoupled[(i + 1):end]
                    @test f₂.isdual[1] == false
                    @test f₂.isdual[2:end] == isdual[(i + 1):end]
                end
            end
            # 往返：split 的两半按定义重接后恢复原树
            for i in 0:N
                f₁, f₂ = split(f, i)
                f′ = FusionTree((f₁.uncoupled..., f₂.uncoupled[2:end]...),
                                f₂.coupled, (f₁.isdual..., f₂.isdual[2:end]...))
                @test f′ == f
            end
        end
        # 越界
        f = FusionTree((e, o, e), o)
        @test_throws ArgumentError split(f, 4)
        @test_throws ArgumentError split(f, -1)
    end

    @testset "permute (FusionTree)" begin
        # domain 空树的 coupled 必须与 f₁ 一致
        empty2(f) = FusionTree((), f.coupled)
        # 两个 odd 腿交换：费米符号 -1
        f = only(fusiontrees((o, o), e))
        ((f₁′, _), coeff) = only(permute(f, empty2(f), (2, 1), ()))
        @test coeff == -1
        @test f₁′.uncoupled == (o, o)
        @test f₁′.coupled == e
        # 含 even 腿的交换：符号 +1
        f = only(fusiontrees((e, o), o))
        ((f₁′, _), coeff) = only(permute(f, empty2(f), (2, 1), ()))
        @test coeff == 1
        @test f₁′.uncoupled == (o, e)
        @test f₁′.coupled == o
        # 三腿循环 (2,3,1)：(o,o,o) 两次 odd-odd 交换，符号 +1
        f = only(fusiontrees((o, o, o), o))
        ((f₁′, _), coeff) = only(permute(f, empty2(f), (2, 3, 1), ()))
        @test coeff == 1
        @test f₁′.uncoupled == (o, o, o)
        # (o,e,o) → (e,o,o)：循环中一次 odd-odd 交换，符号 -1
        f = only(fusiontrees((o, e, o), e))
        ((f₁′, _), coeff) = only(permute(f, empty2(f), (2, 3, 1), ()))
        @test coeff == -1
        @test f₁′.uncoupled == (e, o, o)
        # 恒等置换：符号 +1
        f = only(fusiontrees((e, o, o), e))
        ((f₁′, _), coeff) = only(permute(f, empty2(f), (1, 2, 3), ()))
        @test coeff == 1
        @test f₁′ == f
        # 置换往返：sign(p) * sign(inv(p)) == 1，且恢复原树
        for trial in 1:20
            N = 5
            uncoupled = ntuple(_ -> rand((e, o)), N)
            f = only(fusiontrees(uncoupled, couple(uncoupled)))
            p = Tuple(randperm(N))
            ip = Tuple(invperm(collect(p)))
            ((f₁′, _), s1) = only(permute(f, empty2(f), p, ()))
            ((f₂′, _), s2) = only(permute(f₁′, empty2(f₁′), ip, ()))
            @test s1 * s2 == 1
            @test f₂′ == f
        end
        # 非法置换与 sector 不匹配
        f = only(fusiontrees((e, o), o))
        @test_throws ArgumentError permute(f, empty2(f), (1, 1), ())
        @test_throws SectorMismatch permute(f, only(fusiontrees((e,), e)), (1, 2), (3,))
    end

    @testset "permute: tensor level consistency" begin
        # (2,0) 张量交换 codomain 腿：块数据搬运与 FusionTree 符号一致
        V = FermionicSpace(0 => 2, 1 => 1)
        t = randn(ComplexF64, V ⊗ V ← one(V))
        t2 = permute(t, (2, 1))
        for (f1, f2) in fusiontrees(t)
            f1′ = FusionTree((f1.uncoupled[2], f1.uncoupled[1]), f1.coupled)
            sgn = Rsymbol(f1.uncoupled[1], f1.uncoupled[2], f1.coupled)
            @test t2[f1′, f2] ≈ sgn .* permutedims(t[f1, f2])
        end
    end

    @testset "fusion trees as block keys" begin
        V = FermionicSpace(0 => 2, 1 => 1)
        t = randn(ComplexF64, V ⊗ V ← one(V))
        # (2,0) 空间 V⊗V ← 1 只有耦合到 e 的块：(e,e) 与 (o,o)
        blocks = [(f1, f2) for (f1, f2) in fusiontrees(t)]
        @test Set(f1.uncoupled for (f1, f2) in blocks) == Set(((e, e), (o, o)))
        for (f1, f2) in blocks
            @test length(f2) == 0
            @test f1.coupled == couple(f1.uncoupled)
        end
        # (1,1) 张量 V ← V：块 (e) 与 (o)
        t2 = randn(ComplexF64, V ← V)
        @test Set(f1.uncoupled for (f1, f2) in fusiontrees(t2)) == Set(((e,), (o,)))
        # 多腿空间：所有块可遍历，块维数之和与张量维数一致
        W = V ⊗ V ← V
        t3 = randn(ComplexF64, W)
        @test sum(length(t3[f1, f2]) for (f1, f2) in fusiontrees(t3)) == dim(t3)
    end

    @testset "many legs (regression: no flatten recursion)" begin
        # 12 腿时空结构生成会枚举 2^12 个 sector 组合。早期实现先收集成嵌套元组再
        # TupleTools.flatten，嵌套过深导致类型推断递归爆栈（Internal error: stack
        # overflow in type inference）。回归测试：多腿张量可正常构建与遍历。
        V = FermionicSpace(0 => 1, 1 => 1)
        for N in (12, 16)
            W = ProductSpace{N}(ntuple(_ -> V, N))
            t = rand(ComplexF64, W, one(W))
            # codomain 12 腿两个 sector 都有块，domain 空积只有 e，故张量总 sector 为 e
            @test GrassmannTensors.dim(t) == 2^(N - 1)
            @test length(GrassmannTensors.fusiontrees(t)) == 2^(N - 1)
            @test GrassmannTensors.blockdim(W, e) == 2^(N - 1)
            @test GrassmannTensors.blockdim(W, o) == 2^(N - 1)
            @test GrassmannTensors.hasblock(W, e)
            @test GrassmannTensors.hasblock(W, o)
        end
        # 求和运算在 12 腿张量上可用
        W12 = ProductSpace{12}(ntuple(_ -> V, 12))
        t = rand(ComplexF64, W12, one(W12))
        @test t + t ≈ 2 * t
    end
end
