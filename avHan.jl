using Plots, LinearAlgebra, ProgressMeter


H0 = 0.35
D0  = 1.
n = 10^4
ln = 200
ts = 1:ln
m = 50

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

msd = Matrix{Float64}(undef,m,n)

for i in 1:n
    msd[:,i] .= estMSD(X[:,i],m)
end

lmsd = log.(msd)


## ols

Ts = [ones(m) log.(ts[1:m])]

R = (Ts'*Ts)^-1*Ts'
B = R * lmsd

## gls

hs = LinRange(0.25,0.45, 10)
vD = Vector{Float64}(undef,10)
vH = Vector{Float64}(undef,10)

for (k,h) in enumerate(hs)
    f = (s,t) -> 1/2*(t^(2h)+s^(2h)-abs(s-t)^(2h))
    thcM = @showprogress [theorCov(i,j,ln,f)/(f(ts[i],ts[i])*f(ts[j],ts[j])) for i in 1:m, j in 1:m]
    gR = (Ts'*thcM^-1*Ts)^-1*Ts'*thcM^-1
    gB = gR * lmsd
    vD[k] = var(gB[1,:])
    vH[k] = var(gB[2,:])
end
