using Plots, ProgressMeter, LaTeXStrings, Printf

## bias plot


H0 = 0.25
D0  = 1.
ln = 200
ts = 1:ln
n = 10000
m = 150


xt = vcat( 1:10, 20:10:100)
xl = string.(xt)
xl[end-1] = ""
yl = copy(xl)
yl[8] = ""
yl[9] = ""
yl[end-2] = ""
p = Plots.plot([],[],
    fontfamily = "Computer Modern",
    xlabel = L"k",
    ylabel = "MSD",
    label ="",
    xlim = (1,m),
    ylim = (1,m),
    xticks = (xt, xl),
    yticks = (xt, yl),
    xscale = :log10,
    yscale = :log10,
    xlabelfontsize = 13,
    ylabelfontsize = 13,
    legendfontsize = 13,
)

hs = 0.2:0.1:0.5
hl = length(hs)
pal = palette([palette(:tab10)[4],palette(:tab10)[5],palette(:tab10)[1]],hl)

@showprogress for (k,H0) in enumerate(hs)

    f = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    X = A'*ξ

    msd = Matrix{Float64}(undef,m,n)

    for i in 1:n
        msd[:,i] .= estMSD(X[:,i],m)
    end

    lmsd = log10.(msd)
    mlmsd = mean(lmsd,dims = 2)

    col= pal[k]#palette(:viridis,hl)[k]
    Plots.plot!(t->t^(2H0),1,m,
        linestyle = :dash,
        label = "",
        linecolor = col,
    )

    plot!( 10 .^ mlmsd,
        label = (L"α = " * @sprintf "%.1f" 2H0 ),
        linecolor = col,
    )
    plot!( (1:m) .^(2H0) .* 10 .^( .- log(10)*lerrV.(1:m,ln,f,10) ./2 ),
        label = "",
        linestyle = :dashdotdot,
        linecolor = col,
    )

end
plot!([],[],
    linecolor = :black,
    label = "simulated TA-MSD",
)
plot!([],[],
    linecolor = :black,
    label = "MSD",
    linestyle = :dash,
)
plot!([],[],
    linecolor = :black,
    label = "approximation",
    linestyle = :dashdotdot,
)

display(p)

savefig("bias.pdf")