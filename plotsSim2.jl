
using ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


## 3 elypses plots + margin plot

n = 10^4
ln = 100
dt = 0.0567
ts = dt*(1:ln)

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

## empty plots

p1 = Plots.scatter([],[],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.7,
    #color = palette(:default)[1],
    color = palette(:default)[1],
    xticks = (-3.5:0.25:-2.5, [L"10^{%$s}" for s in -3.5:0.25:-2.5]),
    xlim = (-3.5,-2.5),
    ylim = (-0.1,1.5),
    yticks = -0.1:0.1:1.5,
    #label = "OLS",
    label= "",
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    ylabel = L"$α$",
)
p2 = Plots.scatter([],[],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.7,
    color = palette(:default)[1],
    label = "OLS",
    #color = palette(:default)[2],
    xticks = (-3.5:0.25:-2.5, [L"10^{%$s}" for s in -3.5:0.25:-2.5]),
    xlim = (-3.5,-2.5),
    ylim = (-0.1,1.5),
    yticks = -0.1:0.1:1.5,
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    #ylabel = L"α\ [1]",
)
# plot!(p1,[],[],
#     linewidth = 1,
#     color = :black,
#     linestyle = :dash,
#     label = "error 95% ellipses",
# )

# Plots.scatter!(p2,[],[],
#     markerstrokewidth=0,
#     markersize=0.5,
#     alpha = 0.7,
#     #color = palette(:default)[1],
#     color = palette(:default)[2],
#     label = "GLS",
# )
# Plots.plot!(p2,[],[],
#     linewidth = 1,
#     color = :black,
#     linestyle = :dash,
#     #label = "error 95% ellipse",
# )
# Plots.scatter!(p1, [],[],marker=:x,color=:black,
#     #label = L"exact $(D,\alpha)$"
#     label = "",
# )
#Plots.scatter!(p2, [],[],marker=:x,color=:black, label = L"exact $(D,\alpha)$")

Plots.hline!(p1,[0.,0.7,1.2],linecolor=:black,linestyle=:dash,
    #label = L"exact $\alpha$",
    label = "",
    linewidth= 0.5,
    linealpha=0.5,
)
Plots.hline!(p2,[0.,0.7,1.2],linecolor=:black,linestyle=:dash,
    #label = L"exact $\alpha$",
    label = "",
    linewidth= 0.5,
    linealpha=0.5,
)

## trapped 
n = 10^5

D0, H0 = 10^-3, 0.

X = sqrt(D0)*randn(length(ts), n) # czynnik 2 inny
Y = sqrt(D0)*randn(length(ts), n)
K = (s,t) -> D0*(abs(t-s)< 1e-12)

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]
l = 10

B = Matrix{Float64}(undef, 2, n)
@showprogress for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)

@showprogress for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)

Plots.scatter!(p1,B[1,1:10^4],B[2,1:10^4],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[1],
    label = "",
)

Plots.scatter!(p2,bB[1,1:10^4],bB[2,1:10^4],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[2],
    label = "",
)

Plots.scatter!(p1, [log10(D0)],[2H0],marker=:x,color=:black, label = "")
Plots.scatter!(p2, [log10(D0)],[2H0],marker=:x,color=:black, label = "")

Σ = [2theorCovEff(i,i2,ln,K)/(4K(ts[i],ts[i])*4K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!

eM = (Ts'*errC[:,:,1]^-1*Ts)^-1*Ts'*errC[:,:,1]^-1*Σ*errC[:,:,1]^-1*Ts*(Ts'*errC[:,:,1]^-1*Ts)^-1

eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1


f = t-> cos(t)*sqrt(5.99) # 95% elipse
g1 = t-> sin(t)*sqrt(5.99)

C = sqrt(eM2)

Plots.plot!(p1, t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
    linewidth = 1,
    color = :blue,
    linestyle = :dash,
    label = "",
)
Plots.plot!(p2, t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
    linewidth = 1,
    color = :blue,
    linestyle = :dash,
    label = "",
)
C = sqrt(eM)

Plots.plot!(p2, t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
    linewidth = 1,
    color = :red,
    linestyle = :dash,
    label = "",
)

dat11, dat12 = copy(B), copy(bB)

## 1st pop
D0, H0 = 10^-3, 0.35

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

include("simPlotsScript.jl")

dat21, dat22 = copy(B), copy(bB)

## 2nd pop
D0, H0 = 10^-3, 0.6

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

include("simPlotsScript.jl")

dat31, dat32 = copy(B), copy(bB)
## side plot

using KernelDensity

aOLS = vcat(dat11[2,:],dat21[2,:],dat31[2,:])
dOLS = kde(aOLS)#,bandwidth=0.01)

aGLS = vcat(dat12[2,:],dat22[2,:],dat32[2,:])
dGLS = kde(aGLS)#,bandwidth=0.01)

p3 = Plots.plot(dOLS.x,dOLS.density, permute=(:x,:y),
    fontfamily = "Computer Modern",
    #ylim = (0, 2.2),
    xlim = (-0.1,1.5),
    label = "",
    #xlabel = L"α\ [1]",
    ylabel = L"density $p{}_\alpha$",
    linewidth = 1.0,
    #yscale = :log10,
)
Plots.plot!(dGLS.x,dGLS.density,permute=(:x,:y),
    label = "",
    xticks = -0.1:0.1:1.5,
    linewidth = 1.0,
    linecolor = palette(:default)[2],
)
Plots.hline!([0.,0.7,1.2],linecolor=:black,linestyle=:dash,
    #label = L"exact $\alpha$",
    label = "",
    linewidth= 0.5,
    linealpha=0.5,
)


## plots together

l = @layout [a{0.4w} b{0.4w} c{0.2w}]
Plots.plot!(p1,legend=(0.5,0.3))
Plots.plot!(p2,legend=(0.5,0.3))
Plots.plot!(p3,legend=(0.5,0.3))
Plots.plot(p1,p2,p3,layout = l)

savefig("simPlots2.pdf")