using QuadGK, SpecialFunctions
using ProgressMeter, BenchmarkTools
using LinearAlgebra
using CairoMakie
using Base.Threads

include("funs.jl")
##

function spectrum(ω,s, t, H, ζ) # ω > 0, m = 1, kBT = 1
    g = gamma(2H+1)
    return (cos((t-s)*ω) - cos(t*ω) - cos(s*ω) + 1  ) * ω^(-2) *  ζ*2g*sinpi(H)*ω^(1-2H) / abs(ζ*g*ω^(1-2H)*(sinpi(H) - im*cospi(H)) - im*ω)^2
end

function covGLE(s, t, H, ζ) # ω > 0, m = 1, kBT = 1
    quadgk(ω->spectrum(ω,s, t, H, ζ),0, Inf)[1]/π
end

## covariance

ln = 200
dt = 0.0567
const ts = dt*(1:ln)
H, ζ = 0.8,  20
M = fill(NaN, ln, ln)
for i in 1:ln, j in i:ln
    M[i,j] = covGLE(ts[i],ts[j], H, ζ)
    #M[i,j] = (ts[i]^0.8+ts[j]^0.8 - abs(ts[i]-ts[j])^0.8 ) # FBM
end
E = Symmetric(M)

## sim process

n = 10^4
A = cholesky(E).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)


lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]


l1, l2 = 10, 20

olsTh = (Ts[l1:l2,:]'*Ts[l1:l2,:])^-1*Ts[l1:l2,:]' * log10.(ζ^-1*sinpi(2H)/(pi*H*(1-2H)*(2-2H)) .* ts .^ (2-2H))[l1:l2] # teoria

ols = Matrix{Float64}(undef, 2, n)
for i in 1:n
    ols[:,i] .= (Ts[l1:l2,:]'*Ts[l1:l2,:])^-1*Ts[l1:l2,:]'*lmsd[l1:l2,i]
end

ols[1,:] .-= log10(4)

tZ = @. sinpi(2H)/(pi*H*(1-2H)*(2-2H)) / (10^olsTh[1,:])  # teoria

Z = @. sinpi(2H)/(pi*H*(1-2H)*(2-2H)) / (2*10^ols[1,:])

