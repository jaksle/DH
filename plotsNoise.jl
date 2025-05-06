
using Makie, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


##

H0, D0 = 0.35, 1.
σ = 5
n = 10^5
ln = 100
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [K(s,t) for s in ts, t in ts] 
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ σ .* randn(ln,n)


## GLS prep

hs = 0.05:0.01:0.95
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> D0*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) + ( (s==t) ? 2σ^2 : 0. )
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(f(ts[i],ts[i])*f(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2 # 1D
end


## msd fit
 
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end

lmsd = log10.(msd)


l = 10 # window

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)


## GLS fit
w = 99 # window
gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)
#eB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    #jmin, jmax = findfirst(hs .>= 0.2), findfirst(hs .>= 0.5)
    #j = max(j,jmin); j = min(j,jmax)
    gR = (Ts[1:w,:]'*errC[1:w,1:w,j]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errC[1:w,1:w,j]^-1
    gB[:,i] .= gR*lmsd[1:w,i]
    bB[:,i] .= gR*(lmsd[1:w,i] .- bias[1:w,j])

    gR = (Ts[1:w,:]'*errCEx[1:w,1:w,j]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errCEx[1:w,1:w,j]^-1
    #eB[:,i] .= gR*(lmsd[1:w,i] .- biasEx[1:w,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)
#eB[1,:] .-= log10(4)


##

mean(B[1,:])
mean(gB[1,:])
mean(bB[1,:])
#mean(eB[1,:])
mean(pB[1,:])

mean(B[2,:])
mean(gB[2,:])
mean(bB[2,:])
#mean(eB[2,:])
mean(pB[2,:])


var(bB[1,:])/var(B[1,:])
#var(eB[1,:])/var(B[1,:])
var(pB[1,:])/var(B[1,:])

var(bB[2,:])/var(B[2,:])
#var(eB[2,:])/var(B[2,:])
var(pB[2,:])/var(B[2,:])

mean(B[2,B[2,:] .> 0])
mean(gB[2,gB[2,:] .> 0])
mean(bB[2,bB[2,:] .> 0])
#mean(eB[2,eB[2,:] .> 0])

count(B[2,:] .<= 0 )
count(bB[2,:] .<= 0 )
#count(eB[2,:] .<= 0 )
count(pB[2,:] .<= 0 )

count(B[2,:] .>= 1 )
count(pB[2,:] .>= 1 )

