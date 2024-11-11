using Plots, Statistics, LinearAlgebra
using ProgressMeter

## FBM sim

H0 = 0.35
D0  = 1.
ln = 200
ts = 1:ln
n = 10000

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

##

m = 150

msd = Matrix{Float64}(undef,m,n)

for i in 1:n
    msd[:,i] .= estMSD(X[:,i],m)
end

lmsd = log.(msd)
e = @. msd - D0*(ts[1:m])^(2H0)
le = @. log(msd) - log( D0*(ts[1:m])^(2H0) )

ce = cov(e')
cle = cov(le')

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