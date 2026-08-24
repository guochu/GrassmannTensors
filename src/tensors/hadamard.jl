# hadamard.jl
#-----------
# Hadamard product support for `AbstractTensorMap`, implementing the
# `HadamardProduct.jl` interface. 对应文献（SciPost Phys. Core 7, 063 (2024）
# 附录 B）中两个 Grassmann Tensor 的乘法：先把两个 GT 做 outer product，再把
# 要乘积的指标 permute 到相邻位置（符号由 permute 机制自动处理），最后按
# Eq. (B.7) 的 Grassmann 乘法关系把相邻指标对融合为一个指标：
#
#     ξ^a · ξ^b： 00 → 1（sector even），10/01 → ξ（sector odd，两块相加），
#                 11 → 0（ξ²=0，直接丢弃）。
#
# 符号处理与张量积逻辑一致（Eq. (B.5)-(B.6) 的 swapping signs）：先把 A、B 各自
# permute 到 natural 序 (oA..., sh..., oB...) 中自己腿的位置，融合后再把结果 permute
# 到 C 的指标序，每一段重排的费米符号都由 GT 的 `permute`（树级）机制给出：
#
#     sgn = σ_A · σ_B · σ_C
#     σ_A = permute(fA₁, fA₂, p1_A, p2_A) 的符号（A 的腿重排到 natural 序）
#     σ_B = permute(fB₁, fB₂, p1_B, p2_B) 的符号（B 的腿重排到 natural 序）
#     σ_C = permute(fC₁, fC₂, p1_C, p2_C) 的符号（融合结果重排到 C 的指标序）
#
# 其中 p1_*/p2_* 是张量级 permute 参数：新 codomain（p1）/domain（p2）各腿的旧位置，
# 按目标序排列。融合步骤本身不引入额外符号：被丢弃的组合 (a,b)=(1,1) 恰是唯一
# R 符号为 -1 的共享扇区组合（两个奇 Grassmann 变量交换），所有保留组合的交叉
# 费米符号恒为 +1。融合的两个指标必须同空间
# 同向（同为 codomain 或同为 domain，由 `_check_hadamard_spaces` 保证）：空间
# 必须严格相等，空间与其对偶空间也不匹配（不能是对偶指标）。
# 逐点相乘要求共享腿两侧对应扇区的块维度对齐（PDF 中每个 Grassmann variable 的
# 指数空间各 sector 均 1 维，恒满足）。GT 定义下被融合的指标必须完全一致：若
# 某组合两侧块维度不对齐（例如共享空间 even/odd sector 维度不等时 (0,1)/(1,0)
# 组合），说明输入空间不一致，是调用方 bug，直接抛 `SpaceMismatch` 而非静默跳过。

# Index2Tuple 上的腿计数（GT 的 numin/numout/numind 只接受张量/HomSpace）
_numout(p::Index2Tuple) = length(p[1])
_numin(p::Index2Tuple) = length(p[2])
_numind(p::Index2Tuple) = length(p[1]) + length(p[2])

# 若 conj 为真，先取伴随（空间自动 dual 化），并转换索引置换到伴随张量的框架
function _hadamardconj(t::AbstractTensorMap, p::Index2Tuple, conj::Bool)
    if conj
        return adjoint(t), adjointtensorindices(t, p)
    else
        return t, p
    end
end

# natural 序中的位置 n（1..nO+nS+nB）→ (源张量, 源 all-index 位置)
function _natural_leg(A, pA::Index2Tuple, B, pB::Index2Tuple, n::Int)
    nO, nS = _numout(pA), _numin(pA)
    if n <= nO
        return A, pA[1][n]
    elseif n <= nO + nS
        return A, pA[2][n - nO]
    else
        return B, pB[2][n - nO - nS]
    end
end

# 共享腿空间检查：两条腿必须同型同向（同为 codomain 或同为 domain）且空间
# 严格相等。注意 `space(t, i)` 对 domain 腿返回已 dual 化的空间：若直接比较
# 返回值，会让"一条腿是空间 V、另一条腿是其对偶空间 dual(V) 的 domain 腿"
# （两者的 space() 值恰好都等于 V）漏检，因此这里显式比较腿的类型与 primal
# 空间——空间与其对偶空间不匹配。
function _check_hadamard_spaces(A, pA::Index2Tuple, B, pB::Index2Tuple)
    nS = _numin(pA)
    for k in 1:nS
        iA = pA[2][k]
        iB = pB[1][k]
        outA = iA <= numout(A)
        outB = iB <= numout(B)
        outA == outB ||
            throw(SpaceMismatch("shared index $k must be of the same type in both tensors (A: $(outA ? 'c' : 'd'), B: $(outB ? 'c' : 'd'))"))
        sA = outA ? space(A, iA) : dual(space(A, iA))
        sB = outB ? space(B, iB) : dual(space(B, iB))
        sA == sB ||
            throw(SpaceMismatch("incompatible spaces for shared index $k: $sA ≠ $sB"))
    end
    return nothing
end

# 基本索引合法性检查
function _argcheck_hadamardproduct(A, pA::Index2Tuple, B, pB::Index2Tuple, pAB::Index2Tuple)
    isperm(linearize(pA)) && length(linearize(pA)) == numind(A) ||
        throw(IndexError("invalid index permutation $pA for a tensor with $(numind(A)) indices"))
    isperm(linearize(pB)) && length(linearize(pB)) == numind(B) ||
        throw(IndexError("invalid index permutation $pB for a tensor with $(numind(B)) indices"))
    _numin(pA) == _numout(pB) ||
        throw(IndexError("non-matching number of shared indices: $(_numin(pA)) ≠ $(_numout(pB))"))
    N = _numout(pA) + _numin(pA) + _numin(pB)
    isperm(linearize(pAB)) && _numind(pAB) == N ||
        throw(IndexError("invalid output permutation $pAB for $N output indices"))
    return nothing
end

#-------------------------------------------------------------------------------------------
# Structure and type information
#-------------------------------------------------------------------------------------------

# 输出空间（7/8 参数共用）：natural 序 (oA..., sh..., oB...) 中每条腿继承源腿的
# codomain/domain 类型与空间，得到中间空间 D 后按 pAB 重排到 C 的索引序。
# pC 是 `@hadamard` LHS 分号分组（(codomain 位置, domain 位置)，位置按 LHS 指标序 =
# linearize(pAB) 序），遵循 TensorOperations/TensorKit 的 @tensor 约定：`;` 左侧 =
# codomain、右侧 = domain。GT 下输出腿的类型由源腿继承，因此 pC 分组必须与继承类型
# 一致，否则属于调用方类型错误（例如把 codomain 源腿声明为输出的 domain 腿），直接
# 抛 SpaceMismatch（与 TensorKit 语义一致：对已有张量分号无效果、以对象实际结构为准，
# 对新建输出则以分号声明的结构为准）。
# 注意：HomSpace 的 domain 存的是 primal 空间（`W[i]` 会对 domain 腿再 dual 化），
# 而 `space(t, i)` 对 domain 腿返回的是已 dual 化的空间，因此要 `dual` 回去。
# HD 的 pAB 契约：C 的第 k 个指标（all-index 序，codomain 在前）对应 natural 序的
# 第 `linearize(pAB)[k]` 个指标；这与 GT 的 `select(W, p)` 语义不同，因此手动重排。
function _hadamard_structure(A′, pA′::Index2Tuple, B′, pB′::Index2Tuple,
                             pAB::Index2Tuple, pC::Union{Index2Tuple,Nothing})
    nO, nS, nB = _numout(pA′), _numin(pA′), _numin(pB′)
    N = nO + nS + nB
    # natural 序中每条腿的类型（codomain/domain）与空间（domain 存 primal）
    nattype = Vector{Bool}(undef, N)
    natspace = Vector{FermionicSpace}(undef, N)
    for n in 1:N
        T, i = _natural_leg(A′, pA′, B′, pB′, n)
        if i <= numout(T)
            nattype[n] = true
            natspace[n] = space(T, i)
        else
            nattype[n] = false
            natspace[n] = dual(space(T, i))
        end
    end
    plin = linearize(pAB)
    if pC !== nothing
        # pC 的 codomain 组（位置按 LHS 指标序）必须与源腿继承的类型逐位一致
        incod = falses(N)
        for m in pC[1]
            incod[m] = true
        end
        for k in 1:N
            nattype[plin[k]] == incod[k] || throw(SpaceMismatch(
                "left hand side semicolon grouping of the output is inconsistent with the " *
                "index types inherited from the input tensors: output index at position $k " *
                "(natural position $(plin[k])) is declared $(incod[k] ? "codomain" : "domain") " *
                "but its source leg is $(nattype[plin[k]] ? "codomain" : "domain")"))
        end
    end
    codspaces = FermionicSpace[]
    domspaces = FermionicSpace[]
    for k in 1:N
        n = plin[k]
        if nattype[n]
            push!(codspaces, natspace[n])
        else
            push!(domspaces, natspace[n])
        end
    end
    cod = ProductSpace{length(codspaces)}(tuple(codspaces...))
    dom = ProductSpace{length(domspaces)}(tuple(domspaces...))
    return cod ← dom
end

function HadamardProduct.hadamardproduct_structure(A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                                   B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                                   pAB::Index2Tuple)
    A′, pA′ = _hadamardconj(A, pA, conjA)
    B′, pB′ = _hadamardconj(B, pB, conjB)
    _check_hadamard_spaces(A′, pA′, B′, pB′)
    return _hadamard_structure(A′, pA′, B′, pB′, pAB, nothing)
end

# 8 参版本：`@hadamard` 的 LHS 使用分号（如 `C[i; j k] := ...`）时由
# `tensoralloc_hadamard` 转发 pC，输出的 codomain/domain 分组以分号声明为准。
function HadamardProduct.hadamardproduct_structure(A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                                   B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                                   pAB::Index2Tuple, pC::Index2Tuple)
    A′, pA′ = _hadamardconj(A, pA, conjA)
    B′, pB′ = _hadamardconj(B, pB, conjB)
    _check_hadamard_spaces(A′, pA′, B′, pB′)
    return _hadamard_structure(A′, pA′, B′, pB′, pAB, pC)
end

# 输出张量类型：codomain/domain 划分不能从 pAB（标签 API 把全部指标放入 pAB[1]）
# 推断，只能由 `hadamardproduct_structure` 中 natural 序腿的类型决定。
function HadamardProduct.hadamardproduct_type(TC, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                              B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                              pAB::Index2Tuple)
    structure = HadamardProduct.hadamardproduct_structure(A, pA, conjA, B, pB, conjB, pAB)
    N₁, N₂ = numout(structure), numin(structure)
    M = similarstoragetype(A, TC)
    return tensormaptype(N₁, N₂, M)
end

# 8 参版本：`@hadamard` 的 LHS 使用分号时，codomain/domain 划分以分号声明为准。
function HadamardProduct.hadamardproduct_type(TC, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                              B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                              pAB::Index2Tuple, pC::Index2Tuple)
    structure = HadamardProduct.hadamardproduct_structure(A, pA, conjA, B, pB, conjB, pAB, pC)
    N₁, N₂ = numout(structure), numin(structure)
    M = similarstoragetype(A, TC)
    return tensormaptype(N₁, N₂, M)
end

# 复用 TensorOperations 的分配逻辑（allocator 感知）
function HadamardProduct.tensoralloc(::Type{TT},
                                     structure::TensorMapSpace{N₁,N₂}) where {T,N₁,N₂,
                                                             TT<:AbstractTensorMap{T,N₁,N₂}}
    return TO.tensoralloc(TT, structure, Val(false), TO.DefaultAllocator())
end

function HadamardProduct.checkhadamardproduct(C::AbstractTensorMap, A::AbstractTensorMap,
                                              pA::Index2Tuple, B::AbstractTensorMap,
                                              pB::Index2Tuple, pAB::Index2Tuple)
    _argcheck_hadamardproduct(A, pA, B, pB, pAB)
    space(C) == HadamardProduct.hadamardproduct_structure(A, pA, false, B, pB, false, pAB) ||
        throw(SpaceMismatch("incompatible output space:\n$space"))
    return nothing
end

#-------------------------------------------------------------------------------------------
# Implementation
#-------------------------------------------------------------------------------------------

# 单个块对的元素级组合：把 A/B/C 子块重排到 natural 序 (oA..., sh..., oB...) 后
# 交给 Strided 的 fused kernel。StridedView 对 permuted 维度做 lazy 重排（重算
# strides、不拷贝），sreshape 把 A/B 对齐到 natural 序并插入平凡维；最后的
# broadcast 由 Strided 融合执行（共享维度逐点相乘、外积维度扩展），无中间分配。
# C 的初值处理（β 缩放/清零）已在 kernel 外层完成，这里只累加 α*A⊙B。
function _hadamard_block!(C_block, A_block, B_block,
                          permA, permB, permC,
                          nO::Int, nS::Int, nB::Int,
                          α::Number)
    szA, szB = size(A_block), size(B_block)
    # natural 序下 A/B 的维度：A = (oA..., sh..., 1...1)，B = (1...1, sh..., oB...)
    dimsA = (ntuple(d -> szA[permA[d]], nO)...,
             ntuple(d -> szA[permA[nO + d]], nS)...,
             ntuple(_ -> 1, nB)...)
    dimsB = (ntuple(_ -> 1, nO)...,
             ntuple(d -> szB[permB[d]], nS)...,
             ntuple(d -> szB[permB[nS + d]], nB)...)
    Av = sreshape(StridedView(PermutedDimsArray(A_block, permA)), dimsA)
    Bv = sreshape(StridedView(PermutedDimsArray(B_block, permB)), dimsB)
    Cv = StridedView(PermutedDimsArray(C_block, permC)) # 维度序 = natural 序
    if isone(α)
        @. Cv = Cv + Av * Bv
    else
        @. Cv = Cv + α * Av * Bv
    end
    return nothing
end

# 共享腿融合时 A 侧指数 a 与 B 侧指数 b 的候选组合。融合规则（Eq. B.7）：
#     ξ^a ξ^b = 0   if (a,b) = (1,1)          （ξ² = 0，直接丢弃）
#                1   if (a,b) = (0,0)          （sector even）
#                ξ   if (a,b) ∈ {(0,1),(1,0)}  （sector odd，两块相加）
function _shared_ab_options(c::FermionParity)
    if c.isodd
        return ((FermionParity(false), FermionParity(true)),
                (FermionParity(true), FermionParity(false)))
    else
        return ((FermionParity(false), FermionParity(false)),)
    end
end

# 从 C 的 fusiontree 反解张量（A 或 B）的 fusiontree：外积腿的 sector 直接从
# fC 继承，共享腿的 sector 取组合值 `shared`。`pos2nat` 为该张量 all-index 位置
# → natural 位置；`nat2C` 为 natural 位置 → C 的 all-index 位置；natural 共享腿
# 区间为 (nO, nO+nS]。
function _hadamard_fusiontree(bst, fC₁, fC₂, pos2nat, nat2C, N₁C,
                              N₁, N₂, nO, nS, isdual₁, isdual₂, shared)
    sectorat(j) = begin
        n = pos2nat[j]
        if nO < n <= nO + nS
            shared[n - nO]
        else
            m = nat2C[n]
            m <= N₁C ? fC₁.uncoupled[m] : fC₂.uncoupled[m - N₁C]
        end
    end
    unc₁ = ntuple(j -> sectorat(j), N₁)
    unc₂ = ntuple(j -> sectorat(N₁ + j), N₂)
    f₁ = FusionTree(unc₁, couple(unc₁), isdual₁)
    f₂ = FusionTree(unc₂, couple(unc₂), isdual₂)
    i = get(bst.fusiontreeindices, (f₁, f₂), 0)
    return i, f₁, f₂
end

# 块级主循环：遍历 C 的每个 fusiontree 对，对每条共享腿按 B.7 枚举 A/B 的
# 指数组合 (aₖ, bₖ)，映射回 A/B 的 fusiontree 查块并做元素级组合（共享腿逐点
# 相乘、外积腿做张量积）。查不到 A/B 块、或 (aₖ,bₖ) 为 (1,1) 的组合，其对
# A⊙B 的贡献恒为 0，仅保留 β*C 的初值；两侧块维度不对齐属于输入空间不一致
# （调用方 bug），直接抛 SpaceMismatch。
function _hadamardproduct_kernel!(C::AbstractTensorMap,
                                  A′::AbstractTensorMap, pA′::Index2Tuple,
                                  B′::AbstractTensorMap, pB′::Index2Tuple,
                                  pAB::Index2Tuple, α::Number, β::Number)
    iszero(β) && zerovector!(C) # β=0：C 未初始化，被跳过的块必须为 0
    # β≠0 且 β≠1：先整体缩放 C（被跳过的块也要乘 β）
    !iszero(β) && !isone(β) && scale!(C, β)

    nO, nS, nB = _numout(pA′), _numin(pA′), _numin(pB′)
    N = nO + nS + nB
    N₁C, N₂C = numout(C), numin(C)
    # natural 序中每条腿的类型（codomain/domain，由源腿继承，融合不改变）
    nattype = Vector{Bool}(undef, N)
    for n in 1:N
        T, i = _natural_leg(A′, pA′, B′, pB′, n)
        nattype[n] = i <= numout(T)
    end
    # C 的 all-index 序 = (codomain 腿..., domain 腿...)，各自保持 linearize(pAB)
    # 中的相对顺序；`nat2C`：natural 位置 → C 的 all-index 位置。
    Cpos2nat = Vector{Int}(undef, N)
    codpos, dompos = 0, 0
    for k in 1:N
        n = linearize(pAB)[k]
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
    permA = linearize(pA′)
    permB = linearize(pB′)

    A_struct = fusionblockstructure(A′)
    B_struct = fusionblockstructure(B′)
    C_struct = fusionblockstructure(C)
    N₁A, N₂A = numout(A′), numin(A′)
    N₁B, N₂B = numout(B′), numin(B′)
    isdualA₁ = map(isdual, codomain(A′).spaces)
    isdualA₂ = map(isdual, domain(A′).spaces)
    isdualB₁ = map(isdual, codomain(B′).spaces)
    isdualB₂ = map(isdual, domain(B′).spaces)

    # 共享腿空间（`_check_hadamard_spaces` 已保证 A/B 同型同空间）
    shspaces = ntuple(k -> space(A′, pA′[2][k]), nS)

    # 张量 all-index 位置 → natural 位置
    Apos2nat = Vector{Int}(undef, N₁A + N₂A)
    for j in 1:nO
        Apos2nat[pA′[1][j]] = j
    end
    for k in 1:nS
        Apos2nat[pA′[2][k]] = nO + k
    end
    Bpos2nat = Vector{Int}(undef, N₁B + N₂B)
    for k in 1:nS
        Bpos2nat[pB′[1][k]] = nO + k
    end
    for j in 1:nB
        Bpos2nat[pB′[2][j]] = nO + nS + j
    end

    # 符号因子（张量积逻辑，Eq. (B.5)-(B.6) 的 swapping signs）：σ_A·σ_B·σ_C。
    # 每段重排都是把该张量的腿重排到 natural 序（σ_A/σ_B）或把融合结果（natural 序）
    # 重排到 C 的指标序（σ_C），符号由树级 `permute` 机制给出（domain 逆序约定、
    # 奇扇区交换 -1 均由它处理）。p1/p2 是张量级 permute 参数：新 codomain（p1）/
    # domain（p2）各腿的旧位置，按目标序排列。
    p1_A = Tuple(permA[t] for t in 1:(nO + nS) if permA[t] <= N₁A)
    p2_A = Tuple(permA[t] for t in 1:(nO + nS) if permA[t] > N₁A)
    p1_B = Tuple(permB[t] for t in 1:(nS + nB) if permB[t] <= N₁B)
    p2_B = Tuple(permB[t] for t in 1:(nS + nB) if permB[t] > N₁B)
    p1_C = Tuple(nat2C[n] for n in 1:N if nattype[n])
    p2_C = Tuple(nat2C[n] for n in 1:N if !nattype[n])

    for (fC₁, fC₂) in C_struct.fusiontreelist
        # σ_C 只依赖 C 的融合树（融合结果与 C 的扇区内容相同，仅位置不同），
        # 与共享腿的 (a,b) 组合无关，提到组合循环外只算一次。
        sgnC = only(permute(fC₁, fC₂, p1_C, p2_C))[2]
        # natural 位置 n 处 C 的 uncoupled sector（外积腿与共享腿均从 fC 读取）
        sectorC(n) = nat2C[n] <= N₁C ? fC₁.uncoupled[nat2C[n]] :
                                       fC₂.uncoupled[nat2C[n] - N₁C]
        opts = ntuple(k -> _shared_ab_options(sectorC(nO + k)), nS)
        for combo in Iterators.product(opts...)
            as = ntuple(k -> combo[k][1], nS)
            bs = ntuple(k -> combo[k][2], nS)
            # 逐点相乘要求共享腿两侧同 sector 的块维度对齐。GT 定义下被融合的
            # 指标必须完全一致，若某组合两侧块维度不对齐，说明输入空间不一致，
            # 是调用方 bug，直接抛错而非静默跳过。
            for k in 1:nS
                dA = blockdim(shspaces[k], as[k])
                dB = blockdim(shspaces[k], bs[k])
                dA == dB || throw(SpaceMismatch(
                    "shared index $k: block dims of fused sectors $(as[k]) (A) and $(bs[k]) (B) do not align: $dA ≠ $dB in space $(shspaces[k])"))
            end
            iA, fA₁, fA₂ = _hadamard_fusiontree(A_struct, fC₁, fC₂, Apos2nat, nat2C,
                                                N₁C, N₁A, N₂A, nO, nS,
                                                isdualA₁, isdualA₂, as)
            iA == 0 && continue
            iB, fB₁, fB₂ = _hadamard_fusiontree(B_struct, fC₁, fC₂, Bpos2nat, nat2C,
                                                N₁C, N₁B, N₂B, nO, nS,
                                                isdualB₁, isdualB₂, bs)
            iB == 0 && continue

            # σ_A·σ_B·σ_C：A/B 重排到 natural 序、融合结果重排到 C 序的费米符号
            sgn = only(permute(fA₁, fA₂, p1_A, p2_A))[2] *
                  only(permute(fB₁, fB₂, p1_B, p2_B))[2] * sgnC
            _hadamard_block!(C[fC₁, fC₂], A′[fA₁, fA₂], B′[fB₁, fB₂],
                             permA, permB, permC, nO, nS, nB, α * sgn)
        end
    end
    return C
end

#-------------------------------------------------------------------------------------------
# Public interface
#-------------------------------------------------------------------------------------------

function HadamardProduct.hadamardproduct!(C::AbstractTensorMap,
                                          A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                          B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                          pAB::Index2Tuple, α::Number=1, β::Number=0)
    _argcheck_hadamardproduct(A, pA, B, pB, pAB)
    A′, pA′ = _hadamardconj(A, pA, conjA)
    B′, pB′ = _hadamardconj(B, pB, conjB)
    _check_hadamard_spaces(A′, pA′, B′, pB′)
    space(C) == HadamardProduct.hadamardproduct_structure(A, pA, conjA, B, pB, conjB, pAB) ||
        throw(SpaceMismatch("incompatible output space: $(codomain(C))←$(domain(C))"))
    return _hadamardproduct_kernel!(C, A′, pA′, B′, pB′, pAB, α, β)
end
