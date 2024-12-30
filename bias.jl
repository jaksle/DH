using Plots, Statistics, LinearAlgebra
using ProgressMeter

## FBM sim

H0 = 0.35
D0  = 1.
ln = 100
ts = 1:ln
n = 10000

K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

##




m = 99

msd = Matrix{Float64}(undef,m,n)

for i in 1:n
    msd[:,i] .= estMSD(X[:,i],m)
end

lmsd = log10.(msd)
e = @. msd - D0*(ts[1:m])^(2H0)
le = @. log(msd) - log( D0*(ts[1:m])^(2H0) )

ce = cov(e')
cle = cov(le')

## OLS

ts = (1:ln-1)
lts = log10.(ts)
Ts = [ones(ln-1) log10.(ts)]
l = 10
l2 = 99


errM =[theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
errVar = diag(errM)
bias = @. -log(10)*errVar/2
gR = (Ts[1:l2,:]'*errM[1:l2,1:l2]^-1*Ts[1:l2,:])^-1*Ts[1:l2,:]'*errM[1:l2,1:l2]^-1


B = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)
gB = Matrix{Float64}(undef, 2, n)
gbB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
    bB[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'* (lmsd[1:l,i] .- bias[1:l])
    gB[:,i] .= gR * lmsd[1:l2,i]
    gbB[:,i] .= gR * (lmsd[1:l2,i] .- bias[1:l2])
end




## bias

plot(log.(ts[1:m]),mean(lmsd,dims = 2)) # działa
plot!(l->2H0*l+log(D0),0,4,linestyle=:dash)
plot!(log.(ts[1:m]), 2H0 .* log.(ts[1:m]) .+ log(D0) .- lerrV.(1:m,ln) ./2,
    linestyle=:dot,
)

## bias różne H

H0 = 0.25
D0  = 1.
ln = 200
ts = 1:ln
n = 10000

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))


plot( - lerrV.(1:100,ln) ./2 )

plot!( - lerrV.(1:m,ln) ./2  ./ (2H0 .* log.(ts[1:m]) .+ log(D0)) )