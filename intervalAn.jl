
using Plots, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")

##

n = 10^4
ln = 200
ts = 1:ln


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

w = 11
B2 = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B2[:,i] .= (Ts[w:w+l-1,:]'*Ts[w:w+l-1,:])^-1*Ts[w:w+l-1,:]'*msd[w:w+l-1,i]
end



##

gB2 = Matrix{Float64}(undef, 2, n)

tA = 1
K = (s,t) -> 1/2*(t^(tA)+s^(tA)-abs(s-t)^(tA))
Σ = Matrix{Float64}(undef,ln-1,ln-1)
@showprogress for i in 1:ln-1,i2 in 1:i
    Σ[i,i2] = theorCovEff(i,i2,ln,K)
    Σ[i2,i] = Σ[i,i2]
end


gR = (Ts[w:end,:]'*Σ[w:end,w:end]^-1*Ts[w:end,:])^-1*Ts[w:end,:]'*Σ[w:end,w:end]^-1

for i in 1:n
    gB2[:,i] .= gR*(msd[w:end,i])
end

eO = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1
eO2 = (Ts[w:w+l-1,:]'*Ts[w:w+l-1,:])^-1*Ts[w:w+l-1,:]'*Σ[w:w+l-1,w:w+l-1]*Ts[w:w+l-1,:]*(Ts[w:w+l-1,:]'*Ts[w:w+l-1,:])^-1
eG = (Ts[w:end,:]'*Σ[w:end,w:end]^-1*Ts[w:end,:])^-1

## weights analysis

ws = Matrix{Float64}(undef,length(w:ln-1),2)
ll = size(ws)[1]
for k in 1:ll
    v = zeros(ll)
    v[k] = 1
    ws[k,:] .= gR * v
end 

l2= 2
ws2 = Matrix{Float64}(undef,l2,2)
for k in 1:l2
    v = zeros(l2)
    v[k] = 1
    ws2[k,:] .= (Ts[w:w+l2-1,:]'*Ts[w:w+l2-1,:])^-1*Ts[w:w+l2-1,:]' * v
end 


## ta-msd plot

k = 556
errVar = diag(Σ)

p = plot(
    fontfamily = "Computer Modern",
    xlabel = "t [T]",
    ylabel = L"MSD [L$^2$]",
    #xticks = (log10.(xt), string.(xt)),
    #yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
    label = "",
    legend = :top,
    ylim = (1,400),
    xlim = (1,201),
    #xscale = :log10,
    #yscale = :log10
 )
 scatter!([], [],
    marker = :circle,
    markercolor = :white,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
    label = "TA-MSD",
)
plot!(Shape([10,1,1,10],[550,550,1,1]),
    color = palette(:default)[1],
    alpha = 0.3,
    linewidth = 0,
    label = "window used for OLS1",
)
plot!(Shape([20,10,10,20],[550,550,1,1]),
    color = palette(:default)[4],
    alpha = 0.3,
    linewidth = 0,
    label = "window used for OLS2",
)
scatter!(ts[1:2:end], msd[1:2:end,k],
    color = palette(:default)[3],
    marker = :square,
    yerrors = sqrt.(errVar[1:2:end]),
    markersize = 0,
    label = "",
)

scatter!(ts, msd[:,k],
    marker = :circle,
    markercolor = :white,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
    label = "",
)

plot!(t->B1[2,k]*t+B1[1,k],ts[1],4ts[l],
    color = palette(:default)[1],
    linestyle = :dash,
    linewidth = 2.5,
    label = "OLS1 fit"
)
plot!(t->B2[2,k]*t+B2[1,k],ts[w],maximum(ts),
    color = palette(:default)[4],
    linestyle = :dash,
    linewidth = 2.5,
    label = "OLS2 fit"
)
plot!(t->gB2[2,k]*t+gB2[1,k],ts[w],maximum(ts),
    color = palette(:default)[3],
    linestyle = :dash,
    linewidth = 2.5,
    label = "GLS fit"
)


## scatter plot

p = plot(layout=(2,1))
scatter!(p[1],B2[1,:],B2[2,:] .-1/2,
    fontfamily = "Computer Modern",
    ylabel = L"\hat D\ [L^2/T]",
    xlabel = L"\hat  σ\ [L^2]",
    markerstrokewidth=0,
    markersize=2.0,
    alpha = 0.3,
    color = palette(:default)[4],
    label = "OLS2",
    xlim = (-20,20),
    ylim = (-1,3)
)
scatter!(p[2],gB2[1,:],gB2[2,:] .-1/2,
    fontfamily = "Computer Modern",
    ylabel = L"\hat D\ [L^2/T]",
    xlabel = L"\hat σ\ [L^2]",
    markerstrokewidth=0,
    markersize=2.0,
    alpha = 0.3,
    color = palette(:default)[3],
    label = "GLS",
    xlim = (-25,20),
    ylim = (-1,3)
)

f(t) = cos(t)*sqrt(5.99) # 95% elipse
g1(t) = sin(t)*sqrt(5.99)
C = sqrt(eO2)
plot!(p[1],t->C[1,1]*f(t)+C[1,2]*g1(t),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 1/2,0,2pi,
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "error 95% ellipse",
)

C = sqrt(eG)
plot!(p[2],t->C[1,1]*f(t)+C[1,2]*g1(t),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 1/2,0,2pi,
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "error 95% ellipse",
)

scatter!(p[1], [0],[1/2],marker=:x,color=:red, label = L"exact $(\sigma, D)$")
scatter!(p[2], [0],[1/2],marker=:x,color=:red, label = L"exact $(\sigma, D)$")