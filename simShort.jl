
using Plots, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


##

H0, D0 = 0.75, 10^-2
n = 10^5
ln = 10
dt =  0.0567
ts = dt*(1:ln)

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ


## GLS prep

hs = 0.05:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(f(ts[i],ts[i])*f(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./4
end


## msd fit
 
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]
l = 2

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)

## GLS fit
w = 9 # window
gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts[1:w,:]'*errC[1:w,1:w,j]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errC[1:w,1:w,j]^-1
    gB[:,i] .= gR*lmsd[1:w,i]
    bB[:,i] .= gR*(lmsd[1:w,i] .- bias[1:w,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)

## perfect GLS
w = 9
pB = Matrix{Float64}(undef, 2, n)
errM =[theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
errM = cov(lmsd')
errVar = diag(errM)
b2 = @. -log(10)*errVar/4
gR = (Ts[1:w,:]'*errM[1:w,1:w]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errM[1:w,1:w]^-1
for i in 1:n
    #gB[:,i] .= gR*lmsd[1:w,i]
    pB[:,i] .= gR*(lmsd[1:w,i] .- b2[1:w])
end

#gB[1,:] .-= log10(4)
pB[1,:] .-= log10(4)

## test ortogonalizacji

v = Matrix{Float64}(undef, 9, n)
#W = sqrt((errM .- b2 .* b2')^-1)
W = sqrt(errM^-1)
for i in 1:n
    #gB[:,i] .= gR*lmsd[1:w,i]
    v[:,i] .= W*(lmsd[1:w,i] .- b2[1:w])
end

##


scatter(B[1,:],B[2,:],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[1],
    label = "",
)

scatter!(bB[1,:],bB[2,:],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[3],
    label = "",
)

var(bB[1,:])/var(B[1,:])
var(pB[1,:])/var(B[1,:])

var(gB[2,:])/var(B[2,:])

var(bB[2,:])/var(B[2,:])
var(pB[2,:])/var(B[2,:])

count(B[2,:] .<= 0 )
count(pB[2,:] .<= 0 )

mean(B[2,:])
mean(gB[2,:])
mean(bB[2,:])
mean(pB[2,:])

mean(B[1,:])
mean(gB[1,:])
mean(bB[1,:])
mean(pB[1,:])

mean(B[2,B[2,:] .> 0])
mean(gB[2,B[2,:] .> 0])
mean(pB[2,B[2,:] .> 0])
## bias

mlmsd = mean(lmsd,dims=2)

plot(lts,2H0 .*lts .-2*log(2))
plot!(lts,mlmsd .- b2) # 1D

errM =[theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
errVar = diag(errM)
b2 = @. -log(10)*errVar/4