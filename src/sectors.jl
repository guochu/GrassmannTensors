
# FermionParity sector with ℤ₂ fermionic symmetry
#==================================================#
# 本包唯一确定的 sector 是 FermionParity：融合唯一（UniqueFusion）、编织为费米型
# （Fermionic / SymmetricBraiding），因此不需要 Sector/FusionStyle/BraidingStyle
# 抽象类型层次，后续代码直接默认这些配置。

"""
    struct FermionParity
    FermionParity(isodd::Bool)

Represents sectors with fermion parity: a ``ℤ₂`` quantum number, where exchanging two
odd fermions introduces a relative sign `-1`. The core of the symmetry is just the
`Bool` field `isodd` (`true` = odd / fermion, `false` = even / boson).
"""
struct FermionParity
    isodd::Bool
end
const fℤ₂ = FermionParity

# sector values (exactly two: even, odd)
#----------------------------------------
struct SectorValues
end

Base.values(::Type{FermionParity}) = SectorValues()

Base.IteratorSize(::Type{SectorValues}) = HasLength()
Base.length(::SectorValues) = 2
function Base.iterate(::SectorValues, i = 0)
    return i == 2 ? nothing : (FermionParity(i), i + 1)
end
function Base.getindex(::SectorValues, i::Int)
    return 1 <= i <= 2 ? FermionParity(i - 1) : throw(BoundsError(values(FermionParity), i))
end
findindex(::SectorValues, f::FermionParity) = f.isodd ? 2 : 1
# 直接按 sector 取值确定其在 (even, odd) 元组中的下标
findindex(f::FermionParity) = f.isodd ? 2 : 1

# basic sector methods
#----------------------
Base.convert(::Type{FermionParity}, f::FermionParity) = f
Base.convert(::Type{FermionParity}, n::Integer) = FermionParity(isodd(n))

unit(::Type{FermionParity}) = FermionParity(false)
unit(a::FermionParity) = unit(typeof(a))
Base.one(a::FermionParity) = unit(a)
Base.one(::Type{FermionParity}) = unit(FermionParity)
Base.isone(a::FermionParity) = a == unit(a)

dual(f::FermionParity) = f # self-dual
Base.conj(a::FermionParity) = dual(a)

dim(f::FermionParity) = 1

# 融合结果的标量类型：sector 系数（Rsymbol 等）为 Int
sectorscalartype(::Type{FermionParity}) = Int

# fusion: returns a scalar (not a tuple, in line with Z2Tensors conventions)
⊗(c::FermionParity) = c
⊗(a::FermionParity, b::FermionParity) = FermionParity(a.isodd ⊻ b.isodd)
⊗(a::FermionParity, b::FermionParity, cs::Vararg{FermionParity}) = ⊗(⊗(a, b), cs...)
const otimes = ⊗

"""
    Rsymbol(a, b, c)

Fermionic R-symbol: `-1` if both `a` and `b` are odd (and `c == a ⊗ b`), `+1` otherwise
(and `0` if `c` is not the unique fusion outcome of `a` and `b`). This is the core sign
for exchanging two fermions; the mere existence check `(a.isodd ⊻ b.isodd) == c.isodd`
is inlined directly wherever needed (no separate `Nsymbol`).
"""
function Rsymbol(a::FermionParity, b::FermionParity, c::FermionParity)
    (a.isodd ⊻ b.isodd) == c.isodd || return 0
    return a.isodd && b.isodd ? -1 : 1
end

"""
    twist(a::FermionParity)

Twist factor: `-1` for an odd (fermionic) sector, `+1` for an even one.
"""
twist(a::FermionParity) = a.isodd ? -1 : 1

# hashing and ordering
Base.hash(f::FermionParity, h::UInt) = hash(f.isodd, h)
Base.isless(a::FermionParity, b::FermionParity) = isless(a.isodd, b.isodd)

function Base.show(io::IO, a::FermionParity)
    return if get(io, :typeinfo, nothing) === typeof(a)
        print(io, Int(a.isodd))
    else
        print(io, type_repr(typeof(a)), "(", Int(a.isodd), ")")
    end
end
type_repr(::Type{FermionParity}) = "FermionParity"
