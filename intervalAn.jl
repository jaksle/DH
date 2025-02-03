
using Plots, ProgressMeter, LaTeXStrings
using Statistics, HypothesisTests, Distributions, LinearAlgebra

include("funs.jl")

##

H0, D0 = 0.3, 1
n = 10^4
ln = 200
ts = 1:ln

K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ cumsum(randn(ln,n),dims=1)
X = cumsum(randn(ln,n),dims=1)
##

plot(t->t+t^(2H0),0.001,0.1,
    #xscale = :log10,
    #yscale= :log10
)
plot!(t->t^(2H0),0.001,0.1,
    #xscale = :log10,
    #yscale= :log10
)


#

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end

lmsd = log10.(msd)


lts = log10.(ts[1:ln-1])
#Ts = [ones(ln-1) lts]
Ts = [ones(ln-1) ts[1:ln-1]]
l = 10

B1 = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B1[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*msd[1:l,i]
end

w = 10
B2 = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B2[:,i] .= (Ts[w:w+l,:]'*Ts[w:w+l,:])^-1*Ts[w:w+l,:]'*msd[w:w+l,i]
end



##
gB2 = Matrix{Float64}(undef, 2, n)

tA = 1
K = (s,t) -> D0/2*(t^(tA)+s^(tA)-abs(s-t)^(tA))
Σ = Matrix{Float64}(undef,ln-1,ln-1)
@showprogress for i in 1:ln-1,i2 in 1:i
    Σ[i,i2] = theorCovEff(i,i2,ln,K)
    Σ[i2,i] = Σ[i,i2]
end


gR = (Ts[w:end,:]'*Σ[w:end,w:end]^-1*Ts[w:end,:])^-1*Ts[w:end,:]'*Σ[w:end,w:end]^-1
for i in 1:n
    gB2[:,i] .= gR*(msd[w:end,i] +log(10)*diag(Σ[w:end,w:end])/2)
end

eO = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1
eO2 = (Ts[w:w+l,:]'*Ts[w:w+l,:])^-1*Ts[w:w+l,:]'*Σ[w:w+l,w:w+l]*Ts[w:w+l,:]*(Ts[w:w+l,:]'*Ts[w:w+l,:])^-1
eG = (Ts[w:end,:]'*Σ[w:end,w:end]^-1*Ts[w:end,:])^-1