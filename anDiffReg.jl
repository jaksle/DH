using Statistics, LinearAlgebra

function estMSD(X::AbstractVector{T}, w::Integer = length(X) - 1) where T <: Real
    msd = [mean( (X[i]-X[i+j])^2 for i in 1:length(X)-j  ) for j in 1:w]
    return msd
end

function estMSD(X::AbstractMatrix{T}, w::Integer = size(X)[1] - 1) where T <: Real
    msd = Matrix{T}(undef, w, size(X)[2])
    for i in 1:size(X)[2]
        msd[:,i] .= estMSD(view(X,:,i), w)
    end
    return msd
end

function errTAMSD(ts::AbstractVectorVector{T}, w::Integer, K::Function; dim::Integer = 1, logBase::Integer = 10) where T

    function incrCov(ts,i,j,k,l,K) 
        a, b, c, d = ts[i], ts[j], ts[k], ts[l]
        K(a,b) + K(a+c,b+d) - K(a,b+d) - K(a+c,b)
    end
    
    function theorCovEff(ts,k,l,ln,K)
        if k > l
            k, l = l, k
        end
        N1 = h-> ln-l-h+1
        N2 = h-> 
            if h <= l-k+1
                ln-l
            else
                ln-k-h+1
            end
        return 2/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov(ts,1,h,k,l,K)^2 for h in 2:ln-l; init=0) + sum( N2(h)*incrCov(ts,h,1,k,l,K)^2 for h in 1:ln-k ) )
    end

    ln = length(ts)
    S = float(T)
    errCov = Matrix{S}(undef, w, w)
    logErrCov = Matrix{S}(undef, w, w)

    for i in 1:w, j in i:w
        c = theorCovEff(ts,i,j,ln,K)
        errCov[i,j] = dim*c
        logErrCov[i,j] = c / ( dim * K(ts[i],ts[i]) * K(ts[j],ts[j]) * log(logBase)^2 ) 
    end

    return Symmetric(errCov), Symmetric(logErrCov)
end
ᴮ
"""
MSD = A⋅tᴮ

The parameters returned are log₁₀A, B
"""

function anDiffFitOLS(msd::Union{AbstractVector, AbstractMatrix}, ts::AbstractVector,  w::Integer = size(X)[2] ÷ 10)
    Ts = [ones(w) log10.(ts[1:w])]
    estPar = (Ts'*Ts)^-1*Ts' * log10.(msd[1:w,:])
    return estPar
end

log₁₀A, B