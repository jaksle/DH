using DataFrames, CSV
using LinearAlgebra

##

H, D = 0.4, 1 # FBM parameters: Hurst index H and diffusivity D
n = 10^3 # number of trajectories
ln = 100 # trajectory length
dt = 1. # time interval
ts = dt*(1:ln)

K = (s,t) -> D*(t^(2H)+s^(2H)-abs(s-t)^(2H)) # FBM covariance
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U

ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ


d = []
for k in 1:n
    push!(d,"X$k" => X[:,k])
    push!(d,"Y$k" => Y[:,k])
end

dat = DataFrame(d)