using Plots, LinearAlgebra, ProgressMeter, LaTeXStrings

##


H0 = 0.4
D0  = 1.
ln = 200
ts = 1:ln
n = 10^4

K(s,t) = D0*min(s,t)^(2H0)
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