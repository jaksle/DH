using Statistics, LinearAlgebra


function tamsd(X::AbstractArray{dim,T}) where {T <: Real, dim}
    ln, n =  size(X)
    msd = Matrix{T}(undef, ln-1, n)
    for i in 1:size(msd)[1]
        msd[i,:] .=  mean(sum((X[k,:,:] .- X[k+i,:,:])^2, dim = 2) for k in 1:ln-i)
    end
    return msd
end

function errCov_tamsd(ts::AbstractVector{T}, dim::Integer, α::Real,  logBase::Integer = 10) where T<:Real
    K(s,t) = (α ≈ 1.0) ? min(s,t) : 0.5*(s^α + t^α + abs(s-t)^α)

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
    errCov = Matrix{S}(undef, ln-1, ln-1)
    logErrCov = Matrix{S}(undef, ln-1, ln-1)

    for i in 1:ln-1, j in 1:ln-1
        c = theorCovEff(ts,i,j,ln,K)
        errCov[i,j] = dim*c
        logErrCov[i,j] = c / ( dim * K(ts[i],ts[i]) * K(ts[j],ts[j]) * log(logBase)^2 ) 
    end

    return Symmetric(errCov), Symmetric(logErrCov)
end






"""
Return TA-MSD fit.
Input:
- tamsd: ln-1×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
- dim: original trajectory dimension (1,2 or 3)
- Δt: sampling interval
Optional:
- precompute: if true first tabularise error covariance for XXX, if false calculate it for trajectories (could be computationally demanding) 
"""
function fit_tamsd(tamsd::AbstractMatrix, dim::Integer, Δt::Real;
    precompute::Bool = true
    )
    ln, n = size(tamsd)[1]+1, size(tamsd)[2]

    w = max(ln ÷ 10,5)
    ols = fit_ols(tamsd, dim, Δt::Real, w)
end


function fit_ols(tamsd::AbstractMatrix, dim::Integer, Δt::Real, w::Integer)
    Ts = [ones(w) log10.(Δt*(1:w))]
    estPar = (Ts'*Ts)^-1*Ts' * log10.(tamsd[1:w,:])
    estPar[1,:] .-= log10(2dim)
    return estPar
end

"""
Fitting TA-MSD with theGLS method.
Input:
- tamsd: ln-1×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
- dim: original trajectory dimension (1,2 or 3)
- Δt: sampling interval
- init_α: initial approximate values of anomalous exponent
Optional:
- precompute: if true first tabularise error covariances, if false calculate it for trajectories (could be computationally demanding) 
- precompute_αs = 0.1:0.02:1.6: points at which precompute
Output:
- gls: 2×n matrix values of (log10 D, α) estimates
- errCov: 2×2×n matrix with estimated parameter error covariances 
"""
function fit_gls(tamsd::AbstractMatrix, dim::Integer, Δt::Real, init_α::AbstractVector;
     precompute::Bool = true,
     precompute_αs::AbstractVector = 0.1:0.02:1.6
     )
    ln, n = size(tamsd)[1]+1, size(tamsd)[2]
    ts = Δt*(1:ln)
    Ts = [ones(ln-1) log10.(ts[1:ln-1])]
    gls = Matrix{Float64}(undef, 2, n)
    errCov = Array{Float64}(undef, 2, 2, n)
    lmsd = log10.(tamsd)

    if precompute
        # precompute covariances
        na = length(precompute_αs)
        errC = Array{Float64}(undef,ln-1,ln-1,na)
        iC = Array{Float64}(undef,ln-1,ln-1,na)
        bias = Array{Float64}(undef,ln-1,na)

        @showprogress for k in 1:na
            c = errCov_tamsd(ts, dim, precompute_αs[k])[2]
            errC[:,:,k] .= c
            bias[:,k] .=  -log(10) .* diag(c) ./2
            iC[:,:,k] = inv(c)
        end

        # estimate
        for i in 1:n
            j = argmin(abs.(αs .- init_α[i]))
            gR = (Ts'*iC[:,:,j]*Ts)^-1*Ts'*iC[:,:,j]
            gls[:,i] .= gR*(lmsd[:,i] .- bias[:,j])

            j2 = argmin(abs.(αs .- gls[2,i]))
            errCov[:,:,i] .= (Ts'*iC[:,:,j2]*Ts)^-1
        end
    else
        
        @showprogress for i in 1:n
            errC = errCov_tamsd(ts, dim, init_α[i])[2]
            bias = -log(10) .* diag(errC) ./2
            iC = inv(errC)
            gR = (Ts'*iC*Ts)^-1*Ts'*iC
            gls[:,i] .= gR*(lmsd[:,i] .- bias)

            errC2 = errCov_tamsd(ts, dim, gls[2,i])[2]
            errCov[:,:,i] .= (Ts'*inv(errC2)*Ts)^-1
        end
    end

    gls[1,:] .-= log10(2dim)
    return gls, errCov
end


