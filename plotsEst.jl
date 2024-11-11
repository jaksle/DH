using Plots, Distributions, LaTeXStrings

## GLS illustration 1

plot(x->2x+1,-0.1,4.1,
    xlim = (-0.05,4.05),
    ylim = (0,10),
    linestyle = :dash,
    linecolor = :grey,
    label = "",
    alpha = 0.6,
    linewidth = 2,
    #axisratio = 1,
    #framestyle = :origin,
)
ts1 =0:.2:1.8
scatter!(ts1, 2ts1 .+ 1 .+ 0.2*randn(size(ts1)),
    #markerstrokewidth = 0,
    color = palette(:default)[1],
    label = "",
)

ts2 =2:.2:4
scatter!(ts2, 2ts2 .+ 1 .+ randn(size(ts2)),
    #markerstrokewidth = 0,
    color = palette(:default)[2],
    label = "",
)

#savefig("glsIllustration1.pdf")


## GLS illustration 2

plot(x->2x+1,-0.1,4.1,
    xlim = (-0.05,4.05),
    ylim = (0,10),
    linestyle = :dash,
    linecolor = :grey,
    label = "",
    alpha = 0.6,
    linewidth = 2,
    #axisratio = 1,
    #framestyle = :origin,
)
ts1 =0:.2:1.8
scatter!(ts1, 2ts1 .+ 1 .+ randn(size(ts1)),
    #markerstrokewidth = 0,
    color = palette(:default)[1],
    label = "",
)

ts2 =2:.2:4
r = 0.3
scatter!(ts2, 2ts2 .+ 1 .+ r*randn(size(ts2)) .+ sqrt(1-r^2)*randn(),
    #markerstrokewidth = 0,
    color = palette(:default)[2],
    label = "",
)

#savefig("glsIllustration2.pdf")