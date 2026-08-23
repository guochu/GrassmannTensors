# SPLITTING
#-----------
@inline function split(f::FusionTree{N}, M::Int) where {N}
    if M > N || M < 0
        throw(ArgumentError("M should be between 0 and N = $N"))
    elseif M === N
        (f, FusionTree((f.coupled,), f.coupled, (false,)))
    elseif M === 1
        isdual1 = (f.isdual[1],)
        isdual2 = TupleTools.setindex(f.isdual, false, 1)
        f₁ = FusionTree((f.uncoupled[1],), f.uncoupled[1], isdual1)
        f₂ = FusionTree(f.uncoupled, f.coupled, isdual2)
        return f₁, f₂
    elseif M === 0
        f₁ = FusionTree((), one(FermionParity), ())
        uncoupled2 = (one(FermionParity), f.uncoupled...)
        coupled2 = f.coupled
        isdual2 = (false, f.isdual...)
        return f₁, FusionTree(uncoupled2, coupled2, isdual2)
    else
        uncoupled1 = ntuple(n -> f.uncoupled[n], M)
        isdual1 = ntuple(n -> f.isdual[n], M)
        coupled1 = couple(uncoupled1) # abelian: f.innerlines[M-1]

        uncoupled2 = ntuple(N - M + 1) do n
            return n == 1 ? coupled1 : f.uncoupled[M + n - 1]
        end
        isdual2 = ntuple(N - M + 1) do n
            return n == 1 ? false : f.isdual[M + n - 1]
        end
        coupled2 = f.coupled
        f₁ = FusionTree(uncoupled1, coupled1, isdual1)
        f₂ = FusionTree(uncoupled2, coupled2, isdual2)
        return f₁, f₂
    end
end

# PERMUTE (with fermionic sign)
#-------------------------------
# `p1`/`p2` 是整体腿索引空间（codomain 腿 `1:length(f₁)`，domain 腿
# `length(f₁)+1:length(f₁)+length(f₂)`）中的置换，指定哪些原腿成为新的 codomain
# (`p1`) / domain (`p2`) 腿。`N₁`/`N₂` 为新 codomain/domain 腿数，可与
# `length(f₁)`/`length(f₂)` 不同。
function permute(f₁::FusionTree, f₂::FusionTree,
                 p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {N₁,N₂}
    @assert length(f₁) + length(f₂) == N₁ + N₂
    f₁.coupled == f₂.coupled || throw(SectorMismatch())
    p = linearizepermutation(p1, p2, length(f₁), length(f₂))
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))
    # 直接把两棵树合并为一棵单树（线性化顺序：codomain 原序 + domain 逆序）。
    # FermionParity 为 UniqueFusion，任意耦合顺序都合法，且合并系数为 1；
    # domain 腿在合并时经历 dual 化，因此其 isdual 标志被翻转。
    uncoupled = (f₁.uncoupled..., reverse(f₂.uncoupled)...)
    isdual = (f₁.isdual..., reverse(map(!, f₂.isdual))...)
    # 把置换分解为一系列相邻交换（从身份排列出发到达 p），逐次累积费米符号
    legs = collect(uncoupled)
    sign = 1
    for s in permutation2swaps(p)
        a, b = legs[s], legs[s + 1]
        sign *= Rsymbol(a, b, a ⊗ b)
        legs[s], legs[s + 1] = legs[s + 1], legs[s]
    end
    # 应用置换并分裂回 codomain / domain 两棵树（domain 部分再次翻转 isdual）
    f′ = FusionTree(TupleTools.getindices(uncoupled, p), couple(uncoupled),
                    TupleTools.getindices(isdual, p))
    uncoupled1 = ntuple(n -> f′.uncoupled[n], N₁)
    isdual1 = ntuple(n -> f′.isdual[n], N₁)
    uncoupled2 = reverse(ntuple(n -> f′.uncoupled[N₁ + n], N₂))
    isdual2 = reverse(map(!, ntuple(n -> f′.isdual[N₁ + n], N₂)))
    c = couple(uncoupled1)
    f₁′ = FusionTree(uncoupled1, c, isdual1)
    f₂′ = FusionTree(uncoupled2, c, isdual2)
    return SingletonDict((f₁′, f₂′) => sign)
end
