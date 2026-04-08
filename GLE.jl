using QuadGK, SpecialFunctions
using ProgressMeter
using LinearAlgebra
using CairoMakie

include("funs.jl")
##

function spectrum(ω,s, t, H, ζ) # ω > 0, m = 1, kBT = 1
    g = gamma(2H+1)
    return (cos((t-s)*ω) - cos(t*ω) - cos(s*ω) + 1  ) * ω^(-2) *  ζ*2g*sinpi(H)*ω^(1-2H) / abs(ζ*g*ω^(1-2H)*(sinpi(H) - im*cospi(H)) - im*ω)^2
end

function covGLS(s, t, H, ζ) # ω > 0, m = 1, kBT = 1
    quadgk(ω->spectrum(ω,s, t, H, ζ),0, Inf)[1]/π
end

## covariance

ln = 100
dt = 0.0567
const ts = dt*(1:ln)
H, ζ = 0.6,  100
M = fill(NaN, ln, ln)
for i in 1:ln, j in i:ln
    M[i,j] = covGLS(ts[i],ts[j], H, ζ)
end
C = Symmetric(M)


## plot
fig = Figure()
ax = Axis(fig[1,1],
    xscale = log10,
    yscale = log10,
)

scatter!(ax, 1:ln,diag(C))
lines!(ax,1:ln, ζ^-1*sinpi(2H)/(pi*H*(1-2H)*(2-2H)) .* ts .^ (2-2H))
#lines!(ax,ts, ts .^ 2)
fig



## sim process

n = 10^4
A = cholesky(Symmetric(C)).U
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


m, l = 1, 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[m:m+l,:]'*Ts[m:m+l,:])^-1*Ts[m:m+l,:]'*lmsd[m:m+l,i]
end

B[1,:] .-= log10(4)

## spr liczenia covEff
using BenchmarkTools
D = 1 #
ln = 100
dt = 0.0567
const ts = dt*(1:ln)
mA = 0.9 
K = (s,t) -> D*(t^(mA)+s^(mA)-abs(s-t)^(mA))
K = (s,t) -> covGLS(s,t,0.8,10) 
T1 = theorCovEff2(ln,K)
T2 = [theorCovEff(i,i2,ln,K) for i in 1:ln-1,i2 in 1:ln-1]


## GLS prep

as = 0.1:0.02:1
hs = @. 1 - as/2
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    ff = (s,t) -> covGLS(s,t,hs[k],ζ) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,ff)/(ff(ts[i],ts[i])*ff(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

##

##