using Plots, MAT, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")
##

file = matopen("interphase_traj_l100.mat")
#matread("interphase_traj_l100.mat")

dat = read(file,"traj")

X = dat[1:100,1:2:end]
Y = dat[1:100,2:2:end]

ln, n = size(X)
nn = size(dat,2)
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

@showprogress for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

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
#cB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    #cR = (Ts[l+1:end,:]'*errC[l+1:end,l+1:end,j]^-1*Ts[l+1:end,:])^-1*Ts[l+1:end,:]'*errC[l+1:end,l+1:end,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
    #cB[:,i] .= cR*(lmsd[l+1:end,i] .- bias[l+1:end,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)
#cB[1,:] .-= log10(4)
## msd traj plot



xt = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5]
yt = [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1,1]

## making plots

k = 470# 220! 470! 98 550 880
j = findfirst(hs .>= B[2,k]/2)
errVar = diag(errC[:,:,j])

p = plot(
    fontfamily = "Computer Modern",
    xlabel = "t [s]",
    ylabel = L"MSD [μm$^2$]",
    xticks = (log10.(xt), string.(xt)),
    yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
    label = "",
    legend = :topleft,
 )
plot!(lts, lmsd[:,k] .-bias[:,j],
    ribbon = sqrt.(2errVar),
    color = :grey,#palette(:default)[2],
    label = "",
    linewidth = 0,
    #alpha = 0,
)
 scatter!(lts, lmsd[:,k] .-bias[:,j],
    #yerrors = sqrt.(2errVar),
    ribbon = sqrt.(2errVar),
    color = :black,#palette(:default)[2],
    marker = :xcross,
    markersize = 4,
    label = "",
)
scatter!(lts, lmsd[:,k],
    marker = :circle,
    markercolor = :white,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
    label = "original TA-MSD",
)
# scatter!(lts[1:10], lmsd[1:10,k],
#     marker = :circle,
#     color = palette(:default)[1],
#     markerstrokewidth = 0,
#     markersize = 3,
#     label = "TA-MSD used for OLS",
# )
scatter!([],[],
    color = :black,#palette(:default)[2],
    marker = :xcross,
    markersize = 4,
    label = "bias corrected TA-MSD"
)
plot!(lt->B[2,k]*lt+B[1,k]+log10(4),minimum(lts),maximum(lts),
    color = palette(:default)[1],
    linestyle = :dash,
    linewidth = 3,
    label = "OLS fit"
)
plot!(lt->bB[2,k]*lt+bB[1,k]+log10(4),minimum(lts),maximum(lts),
    color = palette(:default)[2],
    linestyle = :dash,
    linewidth = 3,
    label = "GLS fit"
)

display(p)

savefig("trajMSD2.pdf")

## errors

K = (s,t) -> 10^bB[1,k]*(t^(bB[2,k])+s^(bB[2,k])-abs(s-t)^(bB[2,k]))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1

plot!(t->log10(K(10^t,10^t)),-1,0.8)
## scatter plot

scatter(bB[1,:],bB[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=2.5,
    alpha = 0.3,
    color = palette(:default)[3],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "GLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)

savefig("scattGLS.pdf")

d = kde((B[1,:],B[2,:]))
contour!(d.x,d.y,d.density', linecolor=:black)

## changes plot

scatter(B[2,:],bB[2,:],
    axisratio = 1,
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    xlim = (-0.15,1.6),
    ylim = (-0.15,1.6),
    markersize=1.0,
    alpha = 0.3,
    color = palette(:default)[4],
    xlabel = L"\hat\alpha_{\mathrm{OLS}}\ [1]",
    ylabel = L"\hat\alpha_{\mathrm{GLS}}\ [1]",
    label = "estimated α"

)
plot!(x->x,-0.15,1.5,
    linestyle = :dash,
    linewidth = 1,
    linecolor = :black,
    label = "reference y = x",
)
savefig("jointA.pdf")

scatter(B[1,:],bB[1,:],
    axisratio = 1,
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    xlim = (-5,-1),
    ylim = (-5,-1),
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    yticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    markersize=1.0,
    alpha = 0.3,
    color = palette(:default)[6],
    xlabel = L"\hat D_{\mathrm{OLS}}\ [\mu m^2/s^{\alpha}]",
    ylabel = L"\hat D_{\mathrm{GLS}}\ [\mu m^2/s^{\alpha}]",
    label = "estimated D"

)
plot!(x->x,-5,-1,
    linestyle = :dash,
    linewidth = 1,
    linecolor = :black,
    label = "reference y = x",
)

savefig("jointD.pdf")

## density plots

using KernelDensity

den1 = kde((B[1,:],B[2,:]), 
    #bandwidth = (0.05,0.05)
)
den2 = kde((bB[1,:],bB[2,:]),
    #bandwidth = (0.05,0.05)
)

contour(den1.x,den1.y,den1.density',
    xlim = (-5,-1),
    ylim = (-0.2,1.4),
)
contour(den2.x,den2.y,den2.density',color=:viridis,
    xlim = (-5,-1),
    ylim = (-0.2,1.4),
)

xs = LinRange(-5,-1,100)
ys = LinRange(-0.2,1.4,100)
df = [pdf(den2,x,y) - pdf(den1,x,y) for x in xs, y in ys]

heatmap(xs,ys,df',color=:plasma,
    clim = (-0.2,0.2)
)
#scatter!(bB[1,:],bB[2,:],strokelinewidth=0,markersize=0.2)

contour(xs,ys,df',color=:plasma)

nb = 50
lds = LinRange(-5,-0.5,nb+1)
as = LinRange(-0.2,1.4,nb+1)

hist1 = [count((lds[i] .< B[1,:] .< lds[i+1]) .& (as[j] .< B[2,:] .< as[j+1])) for i in 1:nb, j in 1:nb]
hist2 = [count((lds[i] .< bB[1,:] .< lds[i+1]) .& (as[j] .< bB[2,:] .< as[j+1])) for i in 1:nb, j in 1:nb]
heatmap(hist2' .- hist1', clim=(-5,5))