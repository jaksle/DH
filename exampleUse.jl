
using Statistics, LinearAlgebra
using Plots

## exemplary data: simulated fractional Brownian motion

H, D = 0.35, 1/2 # FBM parameters: Hurst index H and diffusivity D
n = 10^4 # number of trajectories
ln = 100 # trajectory length
dt = 1. # time interval

ts = dt*(1:ln)

K = (s,t) -> D*(t^(2H)+s^(2H)-abs(s-t)^(2H)) # depending on notation is D or D/2
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)

X = A'*ξ

## TA-MSD analysis

msd = estMSD(X)

estPar = anDiffFitOLS(msd,ts, 10)