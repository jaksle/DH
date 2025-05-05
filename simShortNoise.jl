
using Plots, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


##


H0, D0, σ = 0.3, 1., 0.2
n = 10^5
ln = 10
dt =  0.0567
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n) 
X = A'*ξ .+ σ .* randn(ln,n)
ξ = randn(length(ts), n)
Y = A'*ξ .+ σ .* randn(ln,n)


## GLS prep exact

hs = 0.05:0.01:0.95
errCEx = Array{Float64}(undef,ln-1,ln-1,length(hs))
biasEx = Array{Float64}(undef,ln-1,length(hs))
msdTemp = Matrix{Float64}(undef,ln-1,n)
l10(x) = x >= 0 ? log10(x) : 0.

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) 
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    x = A'*ξ .+ σ .* randn(ln,n)
    ξ = randn(length(ts), n)
    y = A'*ξ .+ σ .* randn(ln,n)

    for i in 1:n
        msdTemp[:,i] .= estMSD(x[:,i],ln-1) .+ estMSD(y[:,i],ln-1)  # 2D
    end
    ltemp = l10.(msdTemp .- 4σ^2)
    errCEx[:,:,k] .= cov(ltemp')

    biasEx[:,k] .=  mean(ltemp,dims = 2) .- 2hs[k]*lts .- log10(4)  # 2D bias
end


## msd fit
 
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = l10.(msd .- 4σ^2)


l = 5 # window

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)


## GLS fit
w = 9 # window
eB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    jmin, jmax = findfirst(hs .>= 0.2), findfirst(hs .>= 0.6)
    j = max(j,jmin); j = min(j,jmax)
    gR = (Ts[1:w,:]'*errCEx[1:w,1:w,j]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errCEx[1:w,1:w,j]^-1
    eB[:,i] .= gR*(lmsd[1:w,i] .- biasEx[1:w,j])
end

eB[1,:] .-= log10(4)

## perfect GLS
w = 9
H = H0

pB = Matrix{Float64}(undef, 2, n)
#K = (s,t) -> 2D0*(t^(2H)+s^(2H)-abs(s-t)^(2H))
#errM =[theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]

j = findfirst(hs .>= H) # H not α
errM = errCEx[:,:,j]
b2 = biasEx[:,j]
#errM = cov(lmsd')
#b2 =  mean(lmsd,dims = 2) .- 2H*lts .- log10(4D0)  # 2D

gR = (Ts[1:w,:]'*errM[1:w,1:w]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errM[1:w,1:w]^-1

for i in 1:n
    pB[:,i] .= gR*(lmsd[1:w,i] .- b2[1:w])
end


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

mean(B[1,:])
mean(eB[1,:])
mean(pB[1,:])

mean(B[2,:])
mean(eB[2,:])
mean(pB[2,:])


var(eB[1,:])/var(B[1,:])
var(pB[1,:])/var(B[1,:])

var(eB[2,:])/var(B[2,:])
var(pB[2,:])/var(B[2,:])

mean(B[2,B[2,:] .> 0])
mean(gB[2,gB[2,:] .> 0])
mean(bB[2,bB[2,:] .> 0])
mean(eB[2,eB[2,:] .> 0])

count(B[2,:] .<= 0 )
count(bB[2,:] .<= 0 )
count(eB[2,:] .<= 0 )
count(pB[2,:] .<= 0 )

count(B[2,:] .>= 1 )
count(pB[2,:] .>= 1 )
## bias

mlmsd = mean(lmsd,dims=2)

plot(lts,2H0 .*lts .-2*log(2))
plot!(lts,mlmsd .- b2) # 1D

errM =[theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
errVar = diag(errM)
b2 = @. -log(10)*errVar/4


## additional plots


scatter(B[2,:] .- 2H0,eB[2,:] .- 2H0,
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[1],
    axisratio = 1,
    label = "",
)

scatter!(bB[1,:],bB[2,:],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[3],
    label = "",
)


histogram(B[2,:],normed=true)
histogram!(eB[2,:],alpha=0.5,normed=true)