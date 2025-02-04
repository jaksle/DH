
using Plots, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")

##

p1 = scatter([],[],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.5,
    #color = palette(:default)[1],
    color = palette(:default)[2],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "split OLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)
p2 = scatter([],[],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.5,
    #color = palette(:default)[3],
    color = palette(:default)[7],
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    ylim = (-0.1,1.7),
    label = "split GLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)
plot!(p1,[],[],
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "error 95% ellipse",
)
plot!(p2,[],[],
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "error 95% ellipse",
)
scatter!(p1, [],[],marker=:x,color=:red, label = L"exact $(D,\alpha)$")
scatter!(p2, [],[],marker=:x,color=:red, label = L"exact $(D,\alpha)$")

