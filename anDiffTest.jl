
using BenchmarkTools
using LoopVectorization

##
cr = errCov(ts, 1, 0.3)[2]

errC[:,:,1] ./ cr # ok

# load plotSData

init_A = B[2,:]

gls = fit_internal(msd, 2, dt, init_A)


## sim FBM

n = 10^4
h = 0.3
ts = 1:100
dt = step(ts)

K = (s,t) -> 1*(t^(2h)+s^(2h)-abs(s-t)^(2h)) # D = 1
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ
msd = tamsd([X ;;; Y])

gls, c = fit_gls(msd, 2, dt, fill(0.6,n))

cov(gls')

exErr = (Ts'*errCov(ts, 2, 2h)[2]^(-1)*Ts)^-1

gls2, c2 = fit_gls(msd[:,1:200], 2, dt, fill(0.6,n), precompute=false)

## sim FBM + noise

σ = 1.0
H0, D0 = 0.6, 5.
n, ln = 10^4, 100
ts = 1:100
dt = step(ts)

f = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [f(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ σ .* randn(ln,n)
ξ = randn(length(ts), n)
Y = A'*ξ .+ σ .* randn(ln,n)
# msd = Matrix{Float64}(undef,ln-1,n)
# for i in 1:n
#     msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1)
# end
msd = tamsd([X ;;; Y])
lg(x) = x >= 0 ? log10(x) : NaN
lmsd = lg.(msd .- 2*2*σ^2)
lmsd[isnan.(lmsd)] .= 0.

# with plotsNoise.jl
α0 = 2H0
orgC = errCov(ts, 2, α0)[1] # dim = 2
crossC = crossCov(ts, 2, α0) # dim = 2
noiseC = noiseCov.(ln,1:ln-1,(1:ln-1)')

errC = @. (D0^2*orgC + σ^2*D0*crossC + σ^4*2*noiseC) # dim = 2
errC0 = @. 1/(log(10)^2) * (D0^2*orgC + σ^2*D0*crossC + σ^4*2*noiseC)/((2D0*2*ts[1:ln-1]^(α0)) *(2D0*2*ts[1:ln-1]'^(α0)))

gls1, c1 = fit_gls(msd, 2, dt, fill(α0,n))
gls2, c2 = fit_gls(msd, 2, dt, fill(α0,n), fill(D0,n),σ)
gls3, c3 = fit_gls(msd[:,1:1], 2, dt, fill(α0,n), fill(D0,n),σ, precompute = false)

mean(gls2,dims=2)
cov(gls3')

Ts = [ones(ln-1) log10.(ts[1:ln-1])]
parErr = (Ts'*inv(errC0)*Ts)^-1

## numerical benchmarks 

@benchmark errCov(1:100,2, 0.7)
@benchmark errCov2(1:100,2, 0.7)

@code_typed errCov(1:100,2, 0.7)


function incrCov2(ts,i,j,k,l,K) 
    @inbounds a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    @fastmath K(a,b) + K(a+c,b+d) - K(a,b+d) - K(a+c,b)
end

function errCov2(ts::AbstractVector{T}, dim::Integer, α::Real,  logBase::Integer = 10) where T<:Real
    K(s,t) = (α ≈ 1.0) ? 2min(s,t) : (s^α + t^α - abs(s-t)^α) # FBM cov
    
    function theorCovEff(ts,k,l,ln,K)
        if k > l
            k, l = l, k
        end
        N1 = h -> ln-l-h+1
        N2 = h -> (h <= l-k+1) ? ( ln-l ) : ( ln-k-h+1 )

        return 2/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov2(ts,1,h,k,l,K)^2 for h in 2:ln-l; init=0) + sum( N2(h)*incrCov2(ts,h,1,k,l,K)^2 for h in 1:ln-k ) )
    end

    ln = length(ts)
    S = float(T)
    errCov = Matrix{S}(undef, ln-1, ln-1)
    logErrCov = Matrix{S}(undef, ln-1, ln-1)

    @inbounds for i in 1:ln-1, j in i:ln-1
        c = theorCovEff(ts,i,j,ln,K)
        errCov[i,j] = dim*c
        logErrCov[i,j] = c / ( dim * K(ts[i],ts[i]) * K(ts[j],ts[j]) * log(logBase)^2 ) 
    end

    return Symmetric(errCov), Symmetric(logErrCov)
end

function errCov!(M, ts::AbstractVector{T}, dim::Integer, α::Real,  logBase::Integer = 10) where T<:Real
    K(s,t) = (α ≈ 1.0) ? 2min(s,t) : (s^α + t^α - abs(s-t)^α) # FBM cov
    
    function theorCovEff(ts,k,l,ln,K)
        if k > l
            k, l = l, k
        end
        N1 = h -> ln-l-h+1
        N2 = h -> (h <= l-k+1) ? ( ln-l ) : ( ln-k-h+1 )

        return 2/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov2(ts,1,h,k,l,K)^2 for h in 2:ln-l; init=0) + sum( N2(h)*incrCov2(ts,h,1,k,l,K)^2 for h in 1:ln-k ) )
    end

    ln = length(ts)

    @inbounds for i in 1:ln-1, j in i:ln-1
        c = theorCovEff(ts,i,j,ln,K)
        M[i,j] = c / ( dim * K(ts[i],ts[i]) * K(ts[j],ts[j]) * log(logBase)^2 )
        M[j,i] = M[i,j]
    end
end

M1 = errCov(1:100,2,0.7)[2]
M2 = Matrix{Float64}(undef,99,99)
errCov!(M2,1:100,2,0.7)