@testset "FermionParity sectors" begin
    e = FermionParity(false)
    o = FermionParity(true)

    @testset "sector algebra" begin
        @test e ⊗ e == e
        @test e ⊗ o == o
        @test o ⊗ e == o
        @test o ⊗ o == e
        @test dual(e) == e
        @test dual(o) == o
        @test one(e) == e
        @test one(o) == e
        @test dim(e) == dim(o) == 1
        @test typeof(e) === FermionParity
        # 融合唯一（UniqueFusion）与费米型编织（Fermionic）为本包固定配置，无独立 trait 类型
    end

    @testset "Rsymbol" begin
        @test Rsymbol(e, e, e) == 1
        @test Rsymbol(e, o, o) == 1
        @test Rsymbol(o, e, o) == 1
        @test Rsymbol(o, o, e) == -1          # 奇×奇 = -1
        @test Rsymbol(o, o, o) == 0           # 耦合不匹配
        @test Rsymbol(e, e, o) == 0
    end

    @testset "twist" begin
        @test twist(e) == 1
        @test twist(o) == -1
    end
end
