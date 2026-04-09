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
H, ζ = 0.6,  20
M = fill(NaN, ln, ln)
for i in 1:ln, j in i:ln
    M[i,j] = covGLE(ts[i],ts[j], H, ζ)
    #M[i,j] = (ts[i]^0.8+ts[j]^0.8 - abs(ts[i]-ts[j])^0.8 ) # FBM
end
E = Symmetric(M)


## plot
fig = Figure()
ax = Axis(fig[1,1],
    xscale = log10,
    yscale = log10,
)

scatter!(ax, 1:ln,diag(E))
lines!(ax,1:ln, ζ^-1*sinpi(2H)/(pi*H*(1-2H)*(2-2H)) .* ts .^ (2-2H), color = :tomato)
#lines!(ax,1:ln-1, vec(mean(msd,dims=2)) ./2, color = :green) # OK
#lines!(ax,ts, ts .^ 2)
fig



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

## spr liczenia covEff
using BenchmarkTools
D = 1 #
ln = 100
dt = 0.0567
const ts = dt*(1:ln)
mA = 0.9 
K = (s,t) -> D*(t^(mA)+s^(mA)-abs(s-t)^(mA))
#K = (s,t) -> covGLE(s,t,0.8,10) 
T1 = theorCovEff2(ln,K)
T2 = [theorCovEff(i,i2,ln,K) for i in 1:ln-1,i2 in 1:ln-1]


## GLS 

n2 = 10
nt = 0
l1, l2 = 10, 199
gls = Matrix{Float64}(undef, 2, n2)
mH = 1 - mean(ols[2,:])/2
mZ = sinpi(2mH)/(pi*mH*(1-2mH)*(2-2mH)) / (2*10^mean(ols[1,:]))
@showprogress for i in 1:n2
    h = 1 - ols[2,nt+i]/2
    if h < 1/2 || h > 1
        h = mH
    end 
    z = sinpi(2h)/(pi*h*(1-2h)*(2-2h)) / (2*10^ols[1,nt+i])
    if z < 0.1 || z > 100
        z = mZ
    end 
    #h, z = H, ζ
    C = theorCovEff2(ln,(s,t) -> covGLE(s,t,h,z), l1, l2)
    thMSD = covGLE.(ts,ts,h,z) # dim = 1
    #C = theorCovEff2(ln,(s,t) -> ζ^-1*sinpi(2H)/(2pi*H*(1-2H)*(2-2H)) *(t^(2-2H)+s^(2-2H)-abs(s-t)^(2-2H))) # FBM
    #thMSD = ζ^-1*sinpi(2H)/(pi*H*(1-2H)*(2-2H)) .* ts .^ (2-2H) # FBM
    #errC = [theorCovEff(i,j,ln,(s,t) -> ζ^-1*sinpi(2H)/(2pi*H*(1-2H)*(2-2H))*(t^(2-2H)+s^(2-2H)-abs(s-t)^(2-2H)))/(2thMSD[i]*thMSD[j]) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1] # FBM
    
    errC = [ C[i,j]/(2*thMSD[i] * thMSD[j]) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1 ] # dim = 2
    bias =  -log(10) .* diag(errC[l1:l2,l1:l2]) ./2
    gR = (Ts[l1:l2,:]'*errC[l1:l2,l1:l2]^-1*Ts[l1:l2,:])^-1*Ts[l1:l2,:]'*errC[l1:l2,l1:l2]^-1
    gls[:,i] .= gR*(lmsd[l1:l2,nt+i] .- bias)
end

gls[1,:] .-= log10(4)

##
l1, l2 = 10, 20

ols2 = Matrix{Float64}(undef, 2, n)
for i in 1:n
    ols2[:,i] .= (Ts[l1:l2,:]'*Ts[l1:l2,:])^-1*Ts[l1:l2,:]'*lmsd[l1:l2,i]
end

ols2[1,:] .-= log10(4)


##

k = 17
l = 99
fig = Figure()
ax = Axis(fig[1,1],
    xscale = log10,
    yscale = log10,
)


scatter!(ax,ts[1:l], msd[1:l,k], color = :black)
lines!(ax,ts[1:l], 4*10^ols[1,k] .* ts[1:l] .^ ols[2,k],
    color = :blue
)
lines!(ax,ts[1:l], 4*10^gls[1,k] .* ts[1:l] .^ gls[2,k],
    color = :red
)
fig


## plots gls vs ols

fig, ax, s =  scatter(ols[1,:], ols[2,:])
ax.limits = (-3,0,-1,2.5)
scatter!(ax, gls[1,:], gls[2,:],
    color = :tomato
)
fig

##

fig, ax, s =  scatter(1:100, ols[1,1:100])

scatter!(1:100, gls[1,1:100],
    color = :tomato
)
fig

## use JLD2

# @load "simGLE" # PC WMat
# @load "simGLE2"