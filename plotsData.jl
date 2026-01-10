using CairoMakie, MAT, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra
using StatsBase
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

for k in eachindex(hs)
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

##

using KernelDensity

den1 = kde((B[1,:],B[2,:]); boundary=((-5,-0.5),(-0.3,1.7)),npoints=(500,500), 
    bandwidth = (0.06,0.06)
    #bandwidth = (0.04,0.04)
)
den2 = kde((bB[1,:],bB[2,:]); boundary=((-5,-0.5),(-0.3,1.7)),npoints=(500,500), 
    bandwidth = (0.06,0.06)
    #bandwidth = (0.04,0.04)
)


## msd traj plot

xt = [0.1,0.2,0.5,1,2,5]
yt = [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1,1]
#errVar = diag(errC[:,:,j])

##

with_theme(theme_latexfonts()) do
fig = Figure(size = (1200,800),
    fontsize = 22,
)
ax1 = Axis(fig[1,1],
    xlabel = L"$t$ [s]",
    ylabel = L"MSD [μm$^2$]",
    xticks = (log10.(xt), string.(xt)),
    yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
)
k = 476 # 220
j = findfirst(hs .>= B[2,k]/2)
band!(ax1, lts, lmsd[:,k] .-bias[:,j] .- sqrt.(2diag(errC[:,:,j])), lmsd[:,k] .-bias[:,j] .+ sqrt.(2diag(errC[:,:,j])),
    color = :grey,
    alpha = 0.5,
)

CairoMakie.scatter!(ax1,lts, lmsd[:,k],
    label = "original TA-MSD",
    color = :white,
    strokewidth = 1.5,
)
CairoMakie.scatter!(ax1,lts, lmsd[:,k] .-bias[:,j],
    label = "bias corrected TA-MSD",
    marker='⨉',
    color = :black,
    markersize = 10,
    #strokewidth = 1,
)
CairoMakie.lines!(ax1, lts, B[2,k] .* lts .+ B[1,k] .+ log10(4),
    linestyle = :dash,
    linewidth = 3,
    color = :dodgerblue2,
    label = "OLS"
)
CairoMakie.lines!(ax1, lts, bB[2,k] .* lts .+ bB[1,k] .+ log10(4),
    linestyle = :dash,
    linewidth = 3,
    color = :tomato,
    label = "GLS"
)

axislegend(ax1, position = :lt)
ax2 = Axis(fig[1,2],
    xlabel = L"$t$ [s]",
    ylabel = L"MSD [μm$^2$]",
    xticks = (log10.(xt), string.(xt)),
    yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
)
# 476 for showing difference
k =  470 # 220! 470! 98 550 880
j = findfirst(hs .>= B[2,k]/2)
CairoMakie.scatter!(ax2,lts, lmsd[:,k],
    color = :white,
    strokewidth = 1.5,
)
CairoMakie.scatter!(ax2,lts, lmsd[:,k] .-bias[:,j],
    label = "bias corrected TA-MSD",
    marker='⨉',
    color = :black,
    markersize = 10,
    #strokewidth = 1,
)
band!(ax2, lts, lmsd[:,k] .-bias[:,j] .- sqrt.(2diag(errC[:,:,j])), lmsd[:,k] .-bias[:,j] .+ sqrt.(2diag(errC[:,:,j])),
    color = :grey,
    alpha = 0.5,
)
CairoMakie.lines!(ax2, lts, B[2,k] .* lts .+ B[1,k] .+ log10(4),
    linestyle = :dash,
    linewidth = 3,
    color = :dodgerblue2,
    label = "OLS"
)
CairoMakie.lines!(ax2, lts, bB[2,k] .* lts .+ bB[1,k] .+ log10(4),
    linestyle = :dash,
    linewidth = 3,
    color = :tomato,
    label = "GLS"
)


gc = fig[2,1] = GridLayout()
axc1 = Axis(gc[1,1],
    xlabel = L"$D$ [μm$^2$/s^\alpha]",
    ylabel = L"α",
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    yticks = 0:0.3:1.5,
    limits = (-5,-0.5,-0.1,1.5),
    xgridvisible = false,
    ygridvisible = false,
)
CairoMakie.scatter!(axc1,B[1,:],B[2,:],
    markersize = 2,
    color = :dodgerblue2,

)
CairoMakie.scatter!(axc1,bB[1,:],bB[2,:],
    markersize = 2,
    color = :tomato,

)
axislegend(axc1,[
    MarkerElement(color = :dodgerblue2, marker=:circle, markersize = 12),
    MarkerElement(color = :tomato, marker=:circle, markersize = 12), ],["OLS","GLS"],
    position = :lt,
)

axc2 = Axis(gc[1,2],
    xlabel = L"histogram of $p_\alpha$",
    limits = (0,nothing,-0.1,1.5),
    yticklabelsvisible = false,
    xticks = [0, 1.5],
    xgridvisible = false,
    ygridvisible = false,
)
bins = LinRange(-0.2,1.6,20)
h = fit(Histogram,B[2,:],bins)
h = normalize(h,mode=:pdf)
CairoMakie.stairs!(axc2,h.weights,h.edges[1][1:end-1],
    color = :blue,
    linewidth = 2,
)
CairoMakie.hist!(axc2, B[2,:], direction=:x,
    color = (:dodgerblue2,0.3),
    bins = bins,
    normalization = :pdf
)
h = fit(Histogram,bB[2,:],bins)
h = normalize(h,mode=:pdf)
CairoMakie.stairs!(axc2,h.weights,h.edges[1][1:end-1],
    color = :red,
    linewidth = 2,
)
CairoMakie.hist!(axc2, bB[2,:], direction=:x,
    color = (:tomato,0.3),
    bins = bins,
    normalization = :pdf
)
colsize!(gc,2,Relative(1/4))

gd = fig[2,2] = GridLayout()
axd1 = Axis(gd[1,1],
    limits = (-5,-0.5,-0.1,1.5),
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlabel = L"$D$ [μm$^2$/s$^\alpha$]",
    yticks = 0:0.3:1.5,
    ylabel = L"\alpha",
)
hm = CairoMakie.heatmap!(axd1,den1.x,den1.y, den2.density .- den1.density,
    colormap = :seismic,
    colorrange = (-0.2,0.2),
)
text!(axd1,-0.55,1.35, text = "Density difference between GLS and OLS",
    align = (:right,:bottom)
)
Colorbar(gd[1, 2], hm,
    #minorticks = IntervalsBetween(2),
    #minorticksvisible = true,
)

save("data.pdf",fig)
fig
end


##
Plots.scatter!(bB[1,:],bB[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.7,
    #alpha = 0.5,
    #color = :black,
    color = palette(:default)[2],
    framestyle = :box,
    label = "GLS"
)


fig

##


# Plots.plot!(lt->B[2,k]*lt+B[1,k]+log10(4),minimum(lts),maximum(lts),
#     color = palette(:default)[1],
#     linestyle = :dash,
#     linewidth = 3,
#     label = "OLS fit"
# )
# Plots.plot!(lt->bB[2,k]*lt+bB[1,k]+log10(4),minimum(lts),maximum(lts),
#     color = palette(:default)[2],
#     linestyle = :dash,
#     linewidth = 3,
#     label = "GLS fit"
# )


## making plots

k = 220# 220! 470! 98 550 880
j = findfirst(hs .>= B[2,k]/2)
errVar = diag(errC[:,:,j])

p = Plots.plot(
    size = 0.8 .* (600,450),
    fontfamily = "Computer Modern",
    xlabel = L"$t$ [s]",
    ylabel = L"MSD [μm$^2$]",
    xticks = (log10.(xt), string.(xt)),
    yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
    label = "",
    legend = :topleft,
 )
Plots.plot!(lts, lmsd[:,k] .-bias[:,j],
    ribbon = sqrt.(2errVar),
    color = :grey,#palette(:default)[2],
    label = "",
    linewidth = 0,
    #alpha = 0,
)
Plots.scatter!(lts, lmsd[:,k] .-bias[:,j],
    #yerrors = sqrt.(2errVar),
    ribbon = sqrt.(2errVar),
    color = :black,#palette(:default)[2],
    marker = :xcross,
    markersize = 4,
    label = "",
)
Plots.scatter!(lts, lmsd[:,k],
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
Plots.scatter!([],[],
    color = :black,#palette(:default)[2],
    marker = :xcross,
    markersize = 4,
    label = "bias corrected TA-MSD"
)
Plots.plot!(lt->B[2,k]*lt+B[1,k]+log10(4),minimum(lts),maximum(lts),
    color = palette(:default)[1],
    linestyle = :dash,
    linewidth = 3,
    label = "OLS fit"
)
Plots.plot!(lt->bB[2,k]*lt+bB[1,k]+log10(4),minimum(lts),maximum(lts),
    color = palette(:default)[2],
    linestyle = :dash,
    linewidth = 3,
    label = "GLS fit"
)

display(p)

savefig("trajMSD1.pdf")

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
    xlabel = L"MSD [μm$^2$]",
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
    xlabel = L"$\hat D_{\mathrm{OLS}}$  [μm$^2$]",
    ylabel = L"$\hat D_{\mathrm{GLS}}$ [μm$^2$]",
    label = "estimated D"

)
plot!(x->x,-5,-1,
    linestyle = :dash,
    linewidth = 1,
    linecolor = :black,
    label = "reference y = x",
)

savefig("jointD.pdf")

##  difference plot
scatter(bB[2,:],bB[2,:] .- B[2,:],
    markersize = 1,markerstrokewidth = 0, xlabel = "GLS α",ylabel = "GLS α - OLS α", label = "estimates")

denCmp = kde((bB[2,:],bB[2,:] .- B[2,:]))

contour!(denCmp.x,denCmp.y,denCmp.density',)

savefig("OLSvsGLS.pdf")

## density plots

using KernelDensity

den1 = kde((B[1,:],B[2,:]); boundary=((-5,-0.5),(-0.3,1.7)),npoints=(500,500), 
    bandwidth = (0.06,0.06)
    #bandwidth = (0.04,0.04)
)
den2 = kde((bB[1,:],bB[2,:]); boundary=((-5,-0.5),(-0.3,1.7)),npoints=(500,500), 
    bandwidth = (0.06,0.06)
    #bandwidth = (0.04,0.04)
)

heatmap(den2.x,den2.y,den2.density',
    xlim = (-5,-1),
    ylim = (-0.1,1.4),
    color = :OrRd_9,
    linewidth = 1,
    #linestyle = :dash,
)

scatter!(bB[1,:],bB[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.7,
    alpha = 0.5,
    #color = :black,
    color = :black,#palette(:default)[2],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "OLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)
contour!(den1.x,den1.y,den1.density',
    xlim = (-5,-1),
    ylim = (-0.2,1.4),
    color = :GnBu_9,
)
savefig("scattOLS.pdf")

scatter(bB[1,:],bB[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.7,
    alpha = 0.5,
    color = palette(:default)[2],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "GLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)
contour(den2.x,den2.y,den2.density',
    xlim = (-5,-1),
    ylim = (-0.1,1.4),
    color = :OrRd_9,
    linewidth = 1,
    #linestyle = :dash,
)
savefig("scattOLS.pdf")


den1 = kde((B[1,:],B[2,:]); boundary=((-5,-0.5),(-0.3,1.7)),npoints=(500,500), 
    bandwidth = (0.08,0.05)
    #bandwidth = (0.04,0.04)
)
den2 = kde((bB[1,:],bB[2,:]); boundary=((-5,-0.5),(-0.3,1.7)),npoints=(500,500), 
    bandwidth = (0.08,0.05)
    #bandwidth = (0.04,0.04)
)

df = den2.density .- den1.density

Plots.heatmap(den1.x,den1.y,df',
    size = 0.8 .* (600,450), 
    fontfamily = "Computer Modern",
    color = :seismic,
    clim = (-0.25,0.25),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    #title = "Density difference between GLS and OLS",
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    label = "GLS",
    xlabel = L"$D$ [μm$^2$]",
    ylabel = L"α\ [1]",
    framestyle = :box,
    titlelocation = :center,
    grid = :on,

)
annotate!(-2.75,1.6,Plots.text("Density difference between GLS and OLS",:center, 12, "Computer Modern"))

savefig("datadenDiff.pdf")


contour(xs,ys,df',color=:plasma)

nb = 25
lds = LinRange(-5,-0.5,nb+1)
as = LinRange(-0.2,1.4,nb+1)

hist1 = [count((lds[i] .< B[1,:] .< lds[i+1]) .& (as[j] .< B[2,:] .< as[j+1])) for i in 1:nb, j in 1:nb]
hist2 = [count((lds[i] .< bB[1,:] .< lds[i+1]) .& (as[j] .< bB[2,:] .< as[j+1])) for i in 1:nb, j in 1:nb]
heatmap(hist2' .- hist1', clim=(-5,5))

## new scatterplot + histogram

p1 = Plots.scatter(B[1,:],B[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.7,
    #alpha = 0.5,
    #color = :black,
    color = palette(:default)[1],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-1),
    ylim = (-0.1,1.7),
    label = "OLS",
    xlabel = L"$D$ [μm$^2$]",
    ylabel = L"α\ [1]",
    framestyle = :box
)

Plots.scatter!(bB[1,:],bB[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.7,
    #alpha = 0.5,
    #color = :black,
    color = palette(:default)[2],
    framestyle = :box,
    label = "GLS"
)

p2 = Plots.stephist(B[2,:],normed=true,
    fontfamily = "Computer Modern",
    fill=true, fillalpha=0.2,
    label = "OLS ",
    xlim = (-0.1,1.7),
    ylim = (0,2.1),
    ylabel = L"histogram of $p{}_\alpha$",
    permute = (:x,:y)
)
Plots.stephist!(bB[2,:],normed=true,
    color = palette(:default)[2],
    fill=true, fillalpha=0.2,
    label = "GLS ",
    permute = (:x,:y)
)


l = @layout [a{0.7w} b{0.3w}]
Plots.plot!(p1,legend=(0.2,0.8))
Plots.plot!(p2,legend=(0.7,0.8))

Plots.plot(p1,p2,layout = l,
    size = 0.8 .* (600,450),
)

savefig("dataScatt.pdf")