using Plots, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


n = 10^5
ln = 100
dt =  0.0567
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

## GLS prep

hs = 0.05:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    f = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

## GLS prep exact

errCEx = Array{Float64}(undef,ln-1,ln-1,length(hs))
biasEx = Array{Float64}(undef,ln-1,length(hs))
msdTemp = Matrix{Float64}(undef,ln-1,n)

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) 
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    x = A'*ξ
    ξ = randn(length(ts), n)
    y = A'*ξ

    for i in 1:n
        msdTemp[:,i] .= estMSD(x[:,i],ln-1) .+ estMSD(y[:,i],ln-1)  # 2D
    end
    ltemp = log10.(msdTemp)
    errCEx[:,:,k] .= cov(ltemp')

    biasEx[:,k] .=  mean(ltemp,dims = 2) .- 2hs[k]*lts .- log10(4)  # 2D bias
end


##


H0, D0 = 0.35, 1.
n = 10^4
ln = 100
dt =  0.0567
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

l = 10
lmsd = log10.(msd)
B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)

bB = Matrix{Float64}(undef, 2, n)
eB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    #jmin, jmax = findfirst(hs .>= 0.2), findfirst(hs .>= 0.5)
    #j = max(j,jmin); j = min(j,jmax)
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])

    gR = (Ts'*errCEx[:,:,j]^-1*Ts)^-1*Ts'*errCEx[:,:,j]^-1
    eB[:,i] .= gR*(lmsd[:,i] .- biasEx[:,j])
end

bB[1,:] .-= log10(4)
eB[1,:] .-= log10(4)

##


mean(B[1,:])
#mean(gB[1,:])
mean(bB[1,:])
mean(eB[1,:])
#mean(pB[1,:])

mean(B[2,:])
#mean(gB[2,:])
mean(bB[2,:])
mean(eB[2,:])
#mean(pB[2,:])


var(bB[1,:])/var(B[1,:])
var(eB[1,:])/var(B[1,:])
#var(pB[1,:])/var(B[1,:])

var(bB[2,:])/var(B[2,:])
var(eB[2,:])/var(B[2,:])
#var(pB[2,:])/var(B[2,:])

mean(B[2,B[2,:] .> 0])
#mean(gB[2,gB[2,:] .> 0])
mean(bB[2,bB[2,:] .> 0])
mean(eB[2,eB[2,:] .> 0])

count(B[2,:] .<= 0 )
count(bB[2,:] .<= 0 )
count(eB[2,:] .<= 0 )
#count(pB[2,:] .<= 0 )

count(B[2,:] .>= 1 )
#count(pB[2,:] .>= 1 )