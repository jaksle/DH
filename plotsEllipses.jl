

using ProgressMeter, LaTeXStrings, CairoMakie
using Statistics, Distributions, LinearAlgebra
using KernelDensity
include("funs.jl")


## GLS prep
ln = 100
dt = 0.0567
ts = dt*(1:ln)


θs = LinRange(0,2pi,200)
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


##

include("plotsEllipsesSim1.jl")
include("plotsEllipsesSim2.jl")

##
lw = 2

with_theme(theme_latexfonts()) do
fig = Figure(size = (600,400),
    fontsize = 22,
)
ax1 = Axis(fig[1,1],
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    ylabel = L"$α$",
    limits = (-3.4, -2.6, -0.1,1.5),
    xticks = [-3.3,-3.0,-2.7],
    yticks = 0:0.3:1.5,
    xtickformat = xs -> [L"10^{%$x}" for x in xs],
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(3),
    #label = "OLS"
)

ax2 = Axis(fig[1,2],
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    limits = (-3.4, -2.6, -0.1,1.5),
    yticklabelsvisible = false,
    yticks = 0:0.3:1.5,
    xticks = [-3.3,-3.0,-2.7],
    xtickformat = xs -> [L"10^{%$x}" for x in xs],
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(3),
    #ylabel = L"$α$",
    #label = "GLS"
)

ax3 = Axis(fig[1,3],
    xlabel = L"density $p{}_\alpha$",
    limits = (0, nothing, -0.1,1.5),
    yticks = 0:0.3:1.5,
    yticklabelsvisible = false,
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(3),
    #ylabel = L"$α$",
    #label = "GLS"
)
# for ax in [ax1,ax2,ax3]
#     hlines!(ax, [0, 0.7, 1.2],
#         color = :black,
#         linestyle = :dash,
#         linewidth = 1,
#     )
# end
for (ols,gls) in zip((dat11,dat21,dat31), (dat12,dat22,dat32))

    scatter!(ax1,ols[1,1:10^4], ols[2,1:10^4],
        markersize=1.5,
        alpha = 0.3,
        color = :dodgerblue2, 
    )
    scatter!(ax2,gls[1,1:10^4], gls[2,1:10^4],
        markersize=1.5,
        alpha = 0.3,
        color = :tomato,
    )
end



for (C1,C2,H0) in zip((C11,C21,C31), (C12,C22,C32), (0.,0.35,0.6))
    lines!(ax1, C1[1,1]*f1.(θs)+C1[1,2]*g1.(θs) .+ log10(D0), C1[2,1]*f1.(θs)+C1[2,2]*g1.(θs) .+ 2H0,
        linestyle = :dash,
        color =:dodgerblue2,
        linewidth = lw,
    )
    lines!(ax2, C1[1,1]*f1.(θs)+C1[1,2]*g1.(θs) .+ log10(D0), C1[2,1]*f1.(θs)+C1[2,2]*g1.(θs) .+ 2H0,
        linestyle = :dash,
        color = :dodgerblue2, 
        linewidth = lw,
    )
    lines!(ax2, C2[1,1]*f1.(θs)+C2[1,2]*g1.(θs) .+ log10(D0), C2[2,1]*f1.(θs)+C2[2,2]*g1.(θs) .+ 2H0,
        linestyle = :dash,
        linewidth = lw,
        color = :red,
    )
    scatter!(ax1, [log10(D0)],[2H0], marker='⨉', color = :black,markersize = 15)
    scatter!(ax2, [log10(D0)],[2H0], marker='⨉', color = :black,markersize = 15)
end




aOLS = vcat(dat11[2,:],dat21[2,:],dat31[2,:])
dOLS = kde(aOLS,bandwidth=0.015)

aGLS = vcat(dat12[2,:],dat22[2,:],dat32[2,:])
dGLS = kde(aGLS,bandwidth=0.015)

lines!(ax3, dOLS.density, dOLS.x,
    color = :dodgerblue2,
    linewidth = lw,
)
lines!(ax3, dGLS.density, dGLS.x,
    color = :tomato,
    linewidth = lw,
)
axislegend(ax2,[
    MarkerElement(color = :dodgerblue2, marker=:circle, alpha = 0.6, markersize = 12),
    MarkerElement(color = :tomato, marker=:circle, alpha = 0.6, markersize = 12), ],["OLS","GLS"],
    position = (0.9,0.2),
)

colsize!(fig.layout, 3, Relative(1/6))

save("ellipses.pdf",fig)
fig
end


