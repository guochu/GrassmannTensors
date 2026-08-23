# Simple reference to getting and setting BLAS threads
#------------------------------------------------------
set_num_blas_threads(n::Integer) = LinearAlgebra.BLAS.set_num_threads(n)
get_num_blas_threads() = LinearAlgebra.BLAS.get_num_threads()

# 矩阵级正交分解统一委托 MatrixAlgebraKit（与 TensorKit 0.16 一致），不再重复
# 实现 LAPACK 包装。算法标记类型（QR/QRpos/LQ/LQpos/SVD/SDD/Polar）是张量级
# 公共 API，MatrixAlgebraKit 无等价物，故保留。
import MatrixAlgebraKit as MAK
# one! 矩阵实现与 MAK 完全等价，直接复用；张量级方法在 tensors/linalg.jl 等处扩展
import MatrixAlgebraKit: one!

# Factorization algorithms
#--------------------------
abstract type FactorizationAlgorithm end
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

struct QRpos <: OrthogonalFactorizationAlgorithm
end
struct QR <: OrthogonalFactorizationAlgorithm
end
struct LQ <: OrthogonalFactorizationAlgorithm
end
struct LQpos <: OrthogonalFactorizationAlgorithm
end
struct SDD <: OrthogonalFactorizationAlgorithm # lapack's default divide and conquer algorithm
end
struct SVD <: OrthogonalFactorizationAlgorithm
end
struct Polar <: OrthogonalFactorizationAlgorithm
end

Base.adjoint(::QRpos) = LQpos()
Base.adjoint(::QR) = LQ()
Base.adjoint(::LQpos) = QRpos()
Base.adjoint(::LQ) = QR()

Base.adjoint(alg::Union{SVD,SDD,Polar}) = alg

const OFA = OrthogonalFactorizationAlgorithm
const SVDAlg = Union{SVD,SDD}

# 左正交分解 A = Q * R；SVD/SDD 按 atol 截断奇异值，Polar 走极分解
function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return MAK.left_orth!(A; alg = :qr, positive = alg isa QRpos)
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
    if alg isa Polar
        iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
        return MAK.left_polar!(A)
    else
        svdalg = alg isa SVD ? MAK.LAPACK_QRIteration() : MAK.LAPACK_DivideAndConquer()
        U, S, Vᴴ = MAK.svd_compact!(A, svdalg)
        n = count(s -> s > atol, S.diag)
        if n != length(S.diag)
            return U[:, 1:n], lmul!(Diagonal(S.diag[1:n]), Vᴴ[1:n, :])
        else
            return U, lmul!(Diagonal(S.diag), Vᴴ)
        end
    end
end

# 右正交分解 A = L * Q；SVD/SDD 按 atol 截断，Polar 走极分解
function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return MAK.right_orth!(A; alg = :lq, positive = alg isa LQpos)
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
    if alg isa Polar
        iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
        return MAK.right_polar!(A)
    else
        svdalg = alg isa SVD ? MAK.LAPACK_QRIteration() : MAK.LAPACK_DivideAndConquer()
        U, S, Vᴴ = MAK.svd_compact!(A, svdalg)
        n = count(s -> s > atol, S.diag)
        if n != length(S.diag)
            return rmul!(U[:, 1:n], Diagonal(S.diag[1:n])), Vᴴ[1:n, :]
        else
            return rmul!(U, Diagonal(S.diag)), Vᴴ
        end
    end
end

function _svd!(A::StridedMatrix{T}, alg::Union{SVD,SDD}) where {T<:BlasFloat}
    svdalg = alg isa SVD ? MAK.LAPACK_QRIteration() : MAK.LAPACK_DivideAndConquer()
    U, S, Vᴴ = MAK.svd_compact!(A, svdalg)
    return U, S.diag, Vᴴ
end
