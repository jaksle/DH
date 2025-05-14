
using CairoMakie, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


## noise cov

function noiseC(n,k,l)
    if k > l
        l, k = k, l
    end
    if k == l 
        return 4/(n-k)^2 * ( (n >= 2k) ? (3n-4k) : (2n-2k) )
    else
        return 4/((n-k)*(n-l)) * ( (n >= k+l) ? ( 2n-k-2l) : (n-l) )
    end
end


ln = 6
n = 10^6
X = randn(ln,n)

thC = noiseC.(ln,1:ln-1,(1:ln-1)')

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end

cov(msd')


lmsd = log.(msd)

nC = [ thC[i,j]/()]
##

H0, D0 = 0.35, 10.
σ = 2
n = 10^5
ln = 100
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
lts2 = log10.(ts[2:ln])
Ts = [ones(ln-1) lts]

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [K(s,t) for s in ts, t in ts] 
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ σ .* randn(ln,n) # 1D

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end
msd .-= 2σ^2

l(x) = (x> 0) ? log10(x) : 0.
lmsd = l.(msd)

p =plot(ts[1:end-1],mean(msd,dims=2)[:])
lines!(ts[1:end-1],K.(ts[1:end-1],ts[1:end-1]), color=:red)
p

p =plot(lts,mean(lmsd,dims=2)[:])
lines!(lts,2H0 .* lts .+ log10(2D0), color=:red)
p

p =plot(lts,mean(lmsd,dims=2)[:] .+ log(10) .*diag(eMErr)/2' )
lines!(lts,2H0 .* lts .+ log10(2D0), color=:red)
p
## test of err cov, brute force

f = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) + ( (s==t) ? σ^2 : 0. )
eM = [theorCovEff(i,j,ln,f) for i in 1:ln-1, j in 1:ln-1]

eMErr =  1/(log(10)^2) * [ (eM[k,l])/((2D0*ts[k]^(2H0))*(2D0*ts[l]^(2H0))) for k in 1:ln-1, l in 1:ln-1]

cov(lmsd')
cov((lmsd .+ log(10) .*diag(eMErr)/2)')
## test of err cov, equation

f = (s,t) -> 1*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
e1 = [theorCovEff(i,j,ln,f) for i in 1:ln-1, j in 1:ln-1]

thC = noiseC.(ln,1:ln-1,(1:ln-1)')




crossTh = [4/((ln-k)*(ln-l1))*sum(incrCov(i,j,k,l1,f)*(==(i,j) + ==(i+k,j+l1) - ==(i,j+l1) - ==(i+k,j)) for i in 1:(ln-k), j in 1:(ln-l1)) for k in 1:ln-1,l1 in 1:ln-1]

eMTh = D0^2*e1 .+ σ^2*D0*crossTh .+ σ^4*thC


eTh =  1/(log(10)^2) * [(D0^2*e1[k,l] + σ^4*thC[k,l])/((2D0*ts[k]^(2H0))*(2D0*ts[l]^(2H0))) for k in 1:ln-1, l in 1:ln-1]

## GLS prep

hs = 0.05:0.01:0.95
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) + ( (s==t) ? 2σ^2 : 0. )
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

