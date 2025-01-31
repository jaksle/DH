
using Plots, ProgressMeter, LaTeXStrings
using Statistics, HypothesisTests, Distributions, LinearAlgebra

include("funs.jl")

##

H0, D0 = 0.05, 10^-4
n = 10^4
ln = 100
ts = 1:ln

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ
## msd fit

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]
l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)

## GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)

##


scatter(B[1,:],B[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.1,
    color = palette(:default)[1],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "OLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)

scatter(bB[1,:],bB[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.1,
    color = palette(:default)[3],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "GLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)

scatter!([log10(D0)],[2H0],marker=:x,color=:red)

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
eM = (Ts'*Σ^-1*Ts)^-1
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1

f(t) = cos(t)*sqrt(5.99) # 95% elipse
g1(t) = sin(t)*sqrt(5.99)

C = sqrt(eM2) /2

plot!(t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "noise 95% ellipse",
)

