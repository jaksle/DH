

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

f = (s,t) -> 1*(t^(2h)+s^(2h)-abs(s-t)^(2h)) # D = 1
S = [f(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1)
end

gls, c = fit_gls(msd, 2, dt, fill(0.6,n))

cov(gls')

exErr = (Ts'*errCov(ts, 2, 2h)[2]^(-1)*Ts)^-1

gls2, c2 = fit_gls(msd[:,1:200], 2, dt, fill(0.6,n), precompute=false)

## sim FBM + noise

σ = 1.0
H0, D0 = 0.6, 5.
n = 10^5
f = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [f(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ σ .* randn(ln,n)
ξ = randn(length(ts), n)
Y = A'*ξ .+ σ .* randn(ln,n)
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1)
end
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
gls3, c3 = fit_gls(msd[:,1:500], 2, dt, fill(α0,n), fill(D0,n),σ, precompute = false)

Ts = [ones(ln-1) log10.(ts[1:ln-1])]
parErr = (Ts'*inv(errC0)*Ts)^-1