using CairoMakie, ProgressMeter, LaTeXStrings, Printf, LinearAlgebra
include("funs.jl")


## bias Makie plot


H0 = 0.25
D0  = 1.
ln = 200
ts = 1:ln
n = 10_000
m = 150


xt = [1,2,5,10,20,50,100]
xl = string.(xt)
yt = copy(xt)
yl = copy(xl)
with_theme(theme_latexfonts()) do
fig = Figure(size = (600,500),
    fontsize = 22,
)

ax = Axis(fig[1,1],
    #fontfamily = "Computer Modern",
    xlabel = L"$t$ [T]",
    ylabel = L"MSD [L$^2$]",
    xscale = log10,
    yscale = log10,
    xticks = (xt,xl),
    yticks = (yt,yl),
    limits = (1,150,1,150)

)

hs = 0.2:0.1:0.5
hl = length(hs)
pal = [:firebrick1,:purple1,:slateblue1,:dodgerblue]

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

    col= pal[k]
    lines!(ax, ts, ts .^ (2H0),
        linestyle = :dash,
        label = "",
        color = col,
    )

    lines!(ax, 1:m, 10 .^ vec(mlmsd),
        label = (L"α = " * @sprintf "%.1f" 2H0 ),
        color = col,
    )
    lines!(ax, 1:m, (1:m) .^(2H0) .* 10 .^( .- log(10)*lerrV.(1:m,ln,f,10) ./2 ),
        label = "",
        linestyle = :dot,
        color = col,
    )
    axislegend(ax,
       [LineElement(color=pal[4]),
        LineElement(color=pal[3]),
        LineElement(color=pal[2]),
        LineElement(color=pal[1]),
        LineElement(color=:black),
        LineElement(linestyle = :dash),
        LineElement(linestyle = :dot) ],
        [L"\alpha=1.0", L"\alpha=0.8",L"\alpha=0.6",L"\alpha=0.4","simulated TA-MSD","MSD","approximation"],
        position = :lt,
    )
    end


save("bias.pdf",fig)
fig
end

##

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