using Plots, LsqFit, Loess

##

n = 10^4
h1, h2 = 0.25,0.35
c(h) = 50(0.3-h)^2 + 0.05

c(h) = exp(8h)

H = rand(Uniform(0.25,0.35),n)
D = rand(Exponential(),n) .* c.(H)

scatter(H, D,
    label = nothing,
    markersize = 0.5,
    c = :black,
    markerstrokewidth = 0,
    ylabel = "diffusivity [1/time]",
    xlabel = "Hurst exponent [none]",
    #title = "λ = 10",
    yscale = :log10,
    ylim = (10^-1,100),
    legend = :bottomright,
)
plot!(c,h1,h2,
    label = "true c(H)",
    linewidth = 2,
    xlim = (h1,h2)
)

##

plot(c,0.25,0.35)

##
using Optim

r = optimize(l-> sum(@. abs(D / first(l)*exp(last(l)*H)-1)), [1., 1.0]) 

r = optimize(l-> sum(@. (D - l[1]*exp(l[2]*H))^2), [1.,1.]) 

f = curve_fit((x,p)-> exp.(p[1]*x), H, log.(D), [1.])

plot!(h->r.minimizer[1]*exp(r.minimizer[2]*h),0.25,0.35,
    linestyle = :dash,
    linewidth = 2,
    label = "lsq estimation",
)

## loess

using Loess

srt = sortperm(H)
model = loess(H[srt], D[srt], span=0.5)

hs = LinRange(minimum(H),maximum(H),500)
vs = predict(model, hs)

plot!(hs, vs,
    linestyle = :dot,
    #marker = :square,
    linewidth = 2,
    label = "loess estimation",
)

savefig("depDHest.pdf")