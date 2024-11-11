using Plots, Distributions


wrapped(d::Distribution) = begin
    x = rand(d)
    while x <= 0 || x >= 1
        x = rand(d)
    end
    x
end

###########################

##

xs = LinRange(0,17,200)
xs2 = xs .- 2
p = plot(xs, pdf.(Gamma(4,1),xs),
    label = nothing,
    linewidth = 3,
    #linestyle = :dash,
    xlim = (0,15),
    xlabel = "diffusivity [distance²/time?]",
    ylabel = "pdf",
    title = "λ = 1"
)
plot!([NaN], [NaN],
    label = "H = 0.5",
    c = palette(:default)[1],
    linestyle = :dash,
    linewidth = 3,
)
plot!(xs2, pdf.(Gamma(4,1),xs2),
    linestyle = :dash,
    linewidth = 3,
    label = "H = 0.3",
    c = palette(:default)[2]
)

savefig(p,"l1.pdf")
## 

xs = LinRange(0,10,200)
xs2 = xs .- 2
s1 = (0.1)^0.5
s2 = (0.1)^0.3
plot(xs, pdf.(Gamma(4,1),xs ./ s1) ./ s1,
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlim = (0,10),
    xlabel = "diffusivity [distance²/time?]",
    ylabel = "pdf",
    title = "λ = 0.1"
)
plot!([NaN], [NaN],
    label = "H = 0.5",
    c = palette(:default)[1],
    linestyle = :dash,
    linewidth = 3,
)
plot!(xs2, pdf.(Gamma(4,1),xs2 ./ s2) ./ s2,
    linestyle = :dash,
    linewidth = 3,
    label = "H = 0.3",
    c = palette(:default)[2]
)

#savefig("l01.pdf")


##

xs = LinRange(0,30,200)
xs2 = xs .- 2
s1 = (10)^0.5
s2 = (10)^0.3
plot(xs, pdf.(Gamma(4,1),xs ./ s1) ./ s1,
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlim = (0,30),
    xlabel = "diffusivity [distance²/time?]",
    ylabel = "pdf",
    title = "λ = 10"
)
plot!([NaN], [NaN],
    label = "H = 0.5",
    c = palette(:default)[1],
    linestyle = :dash,
    linewidth = 3,
)
plot!(xs2, pdf.(Gamma(4,1),xs2 ./ s2) ./ s2,
    linestyle = :dash,
    linewidth = 3,
    label = "H = 0.3",
    c = palette(:default)[2]
)

#savefig("l10.pdf")

##
############################

trans(p,λ) =  x -> 1/(x*log(λ)) * p(log(x)/log(λ))
λ = 20
p(x) = 0.2 <= x <= 0.5 ? 1/(0.3) : 0.0
plot(trans(p,λ), min(λ^0.2,λ^0.5), max(λ^0.2,λ^0.5))

ff(x) = begin
    σ = 0.05
    (x <= 0 || x >=1) && return 0
    0.5 * ( exp(-(x-0.35)^2/2σ^2)/sqrt(2pi*σ^2) + exp(-(x-0.50)^2/2σ^2)/sqrt(2pi*σ^2) )
end

plot(ff,0,0.56)

plot(trans(ff,10),1.5,4.5,
    label = nothing,
    linewidth = 3,
    xlabel = "diffusivity [distance²/time?]",
    ylabel = "pdf",
    grid = :off
)

savefig("2diff.pdf")

##
##################################

λ = 10
n = 10^4
#sD = rand(Exponential(1), n)
sD = rand(Normal(1,0.05), n)
#sH = rand(Uniform(0.3,0.5), n)
sH = [wrapped(Normal(0.35,0.03)) for _ in 1:n]
sD2 = (λ) .^ sH .* sD

scatter(sD2, sH,
    label = nothing,
    markersize = 0.5,
    c = :black,
    markerstrokewidth = 0,
    xlabel = "diffusivity [distance²/time²ᴴ]",
    ylabel = "Hurst exponent [none]",
    title = "λ = 10"
)

savefig("phaseDH10.pdf")
####################################

## dependent H, D
using Plots,Distributions

n = 10^4
#H = rand(Uniform(0.3,0.5), n)
H =rand(Normal(0.35,0.07), n)
D = rand(Exponential(),n) .* H .^ 2

scatter(H, D, 
    label = nothing,
    markersize = 0.5,
    c = :black,
    markerstrokewidth = 0,
    ylabel = "diffusivity [distance²/time²ᴴ]",
    xlabel = "Hurst exponent [none]",
    #xscale = :log10,
    #yscale = :log10,
)

plot!(h->h^2*mean(Exponential()),0.2,0.5)


## K to D plot

pdfK(k,h) = exp(-k)*pdf(Normal(0.3,0.05),h)

ks = LinRange(0.0,4,200)
hs = LinRange(0.15,0.45,200)
contour(ks,hs,pdfK,
    nlevels = 50,
    xlabel = "K [1/time]",
    ylabel = "H [none]"
)


pdfD(d,h) = 1/(2h*d) * (d)^(1/2h) * pdfK(d^(1/2h),h)

ds = LinRange(0.0,3,200)
hs = LinRange(0.15,0.45,200)
contour(ds,hs,pdfD,
    nlevels = 50,
    xlabel = "D [distance²/time²ᴴ]",
    ylabel = "H [none]",
    #cscale = :log10,
)

##

plot(
    framestyle = :none,
    xlim = (-5,5),
    ylim = (-3,2),
)
plot!([t->0.5t+1, t->0.5t+ 0.5, t->0.5t+0.7, t->0.5t+0.62, t->0.5t+0.45, t->0.5t-2, t->0.5t-1.9, t->0.5t-2.05],
    -5,5,
    color =palette(:default)[1],
    label = "",
)

savefig("linesL.pdf")


## lines


plot(
    framestyle = :none,
    xlim = (-2,8),
    ylim = (-3,2),
)
plot!([t->0.5t, t->0.8t, t->0.1t, t->1.9t, t->1.5t,t->1.6t],
    -5,8,
    color =palette(:default)[1],
    label = "",
)

plot!([t->0.4*(t-4)-1, t->0.2*(t-4)-1,  t->0.8*(t-4)-1,  t->1*(t-4)-1],
    -5,8,
    color =palette(:default)[1],
    label = "",
)

savefig("linesR.pdf")

## lines2 L

genH() = begin
    h = 0.05randn() + 0.4
    while !(0 <= h <= 1)
        h = 0.05randn() + 0.4
    end
    h
end

p = plot(
    #framestyle = :origin,
    xlabel = "log t",
    ylabel = "log msd",
    left_margin = 7Plots.mm,
    xticks = [],
    yticks = [],
    xlim = (-4,4),
    ylim = (-4,4),
    title = "Sample of log msds"
)
for k in 1:50
    h = genH()
    w = 0.4randn()^2-1
    plot!(t->2h*t + w,-4,4,
        alpha = 0.5,
        color =palette(:default)[1],
        label = ""
    )
end
display(p)
savefig("lines2L.pdf")

## lines2 R

using KernelDensity

DD = Matrix{Float64}(undef,500,500)
ys = LinRange(-4,4,500)
ts = LinRange(-4,4,500)
for k in 1:500
    t = ts[k]
    v = [2genH()*t + 0.4randn()^2-1 for k in 1:10^5]
    den = kde(v)
    DD[k,:] .= pdf(den,ys)
end

heatmap(DD',
    #framestyle = :origin,
    xticks = [],
    yticks = [],
    xlabel = "log t",
    ylabel = "log msd",
    left_margin = 7Plots.mm,
    title = "Probability density of log msd",
    cgrad=(scale=:log10)
)

savefig("lines2R.pdf")

## affine

X = rand(MvNormal([0.5 0.4; 0.4 0.8]),300)
H = @. (X[1,:] + 2)/4
lD = X[2,:]
p = scatter(H,lD,
    xlim = (0,1),
    label = "data points",
    markersize = 3,
    c = :black,
    markerstrokewidth = 0,
    xlabel = "H",
    ylabel = "log D",
    title = "Before transformation"
)

plot!(h->4h-2,0,1,
    label = "affine line",
    color = palette(:default)[1],
    linewidth = 2
)

H2 = 0.8 .+ 0.03*randn(20)
lD2 = -2 .+ 0.2*randn(20)

scatter!(H2,lD2,
    label = nothing,
    markersize = 3,
    c = :black,
    markerstrokewidth = 0,
)

display(p)

savefig("affineL.pdf")

## 

p = scatter(H,lD .- 4H .+ 2,
    xlim = (0,1),
    label = "data points",
    markersize = 3,
    c = :black,
    markerstrokewidth = 0,
    xlabel = "H",
    ylabel = "log D",
    title = "After transformation"
)
scatter!(H2,lD2 .- 4H2 .+ 2,
    label = nothing,
    markersize = 3,
    c = :black,
    markerstrokewidth = 0,
)

display(p)

savefig("affineR.pdf")