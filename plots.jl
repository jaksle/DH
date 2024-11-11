using Plots, Distributions, LaTeXStrings


wrapped(d::Distribution) = begin
    x = rand(d)
    while x <= 0 || x >= 1
        x = rand(d)
    end
    x
end

###########################

## ex 1

xs = LinRange(0,17,200)
xs2 = xs .- 2
p = plot(xs, pdf.(Gamma(4,1),xs),
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    #linestyle = :dash,
    xlim = (0,15),
    xlabel = L"D_{T}",
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
display(p)
savefig(p,"l1.pdf")
## 

xs = LinRange(0,10,200)
xs2 = xs .- 2
s1 = (0.1)^0.5
s2 = (0.1)^0.3
plot(xs, pdf.(Gamma(4,1),xs ./ s1) ./ s1,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlim = (0,10),
    xlabel = L"D_{0.1T}",
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

savefig("l01.pdf")


##

xs = LinRange(0,30,200)
xs2 = xs .- 2
s1 = (10)^0.5
s2 = (10)^0.3
plot(xs, pdf.(Gamma(4,1),xs ./ s1) ./ s1,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlim = (0,30),
    xlabel = L"D_{10T}",
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

savefig("l10.pdf")

##
############################ ex 2 2diff

trans(p,λ) =  x -> 1/(x*log(λ)) * p(log(x)/log(λ))
λ = 20
f(x) = 0.2 <= x <= 0.5 ? 1/(0.3) : 0.0
plot(trans(f,λ), min(λ^0.2,λ^0.5), max(λ^0.2,λ^0.5))

ff(x) = begin
    σ = 0.05
    (x <= 0 || x >=1) && return 0
    0.5 * ( exp(-(x-0.35)^2/2σ^2)/sqrt(2pi*σ^2) + exp(-(x-0.50)^2/2σ^2)/sqrt(2pi*σ^2) )
end
function genH()
end

plot(ff,0.15,0.7,
fontfamily = "Computer Modern",
label = nothing,
linewidth = 3,
xlabel = L"H",
ylabel = "pdf",
grid = :off
)
savefig("2diffL.pdf")


plot(trans(ff,10),1.5,4.5,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    xlabel = L"D_{20T}",
    ylabel = "pdf",
    grid = :off
)

savefig("2diffR.pdf")

genH() = 0.05* randn() + (rand() < 1/2 ? 0.35 : 0.5)
Hs = [genH() for _ in 1:10000]
Ds0 = ones(10000)
λ = 20
histogram(Ds0, #λ .^(2Hs),
    fontfamily = "Computer Modern",
    xlabel = "diffusivity",
    ylabel = "counts",
    label ="",
    xlim = (0,10),
    xticks = 0:10,
    bins = 0.05:0.1:9.795,
)
savefig("histDiff0.pdf")
##
##### phase DH

λ = 10_000
n = 10^4
#sD = rand(Exponential(1), n)
sD = rand(Normal(1,0.05), n)
#sH = rand(Uniform(0.3,0.5), n)
sH = [wrapped(Normal(0.35,0.03)) for _ in 1:n]
sD2 = (λ) .^ sH .* sD

scatter(sD2, sH,
    fontfamily = "Computer Modern",
    label = nothing,
    markersize = 0.5,
    c = :black,
    markerstrokewidth = 0,
    xlabel = L"D_{10^5T}",
    ylabel = "H",
    title = L"λ = 10^5"
)

savefig("phaseDHf5.pdf")
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
    fontfamily = "Computer Modern",
    nlevels = 50,
    xlabel = L"K_{L,T}",
    ylabel = "H"
)
savefig("compK.pdf")

pdfD(d,h) = 1/(2h*d) * (d)^(1/2h) * pdfK(d^(1/2h),h)

ds = LinRange(0.0,3,200)
hs = LinRange(0.15,0.45,200)
contour(ds,hs,pdfD,
    fontfamily = "Computer Modern",
    nlevels = 50,
    xlabel = L"D_{L,T}",
    ylabel = "H",
    #cscale = :log10,
)
savefig("compD.pdf")


# error bars plot

scatter([5], [0.3],
    title= L"\lambda = 1",
    markercolor = :black,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlabel = L"D_{T}",
    ylabel = "H",
    xlim = (3.5,6.5),
    ylim = (0.15,0.45),
    grid = :none,
)
plot!([6,4,4,6,6],[0.4,0.4,0.2,0.2,0.4],
    linestyle = :dash,
    linecolor = :black,
    label = "",
    fillalpha = 0.2,
    fillcolor = :grey,
    fillrange = 0.2,
)
savefig("errBarTL.pdf")

λ = 10


scatter([5*(λ)^0.6], [0.3],
    title= L"\lambda = 10",
    markercolor = :black,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlabel = L"D_{10T}",
    ylabel = "H",
    xlim = (8,40),
    ylim = (0.15,0.45),
    grid = :none,
)
xs = LinRange(4*(λ)^0.4,6*(λ)^0.4,100)
plot!(xs,fill(0.2,100),
    linestyle = :dash,
    linecolor = :black,
    label = "",
    fillalpha = 0.2,
    fillcolor = :grey,
    fillrange = log10.(xs ./4) ./2,
)
plot!([4*(λ)^0.8,6*(λ)^0.8],[0.4,0.4],
    linestyle = :dash,
    linecolor = :black,
    label = "",
)

plot!(d-> log10(d/4)/2,4*(λ)^0.4,4*(λ)^0.8,
    linestyle = :dash,
    linecolor = :black,
    label = ""
)
xs = LinRange(6*(λ)^0.4,6*(λ)^0.8,100)
plot!(xs, log10.(xs ./6) ./2,
    linestyle = :dash,
    linecolor = :black,
    label = "",
    fillalpha = 0.2,
    fillcolor = :grey,
    fillrange = min.(log10.(xs ./4) ./2, 0.4),
)

savefig("errBarTR.pdf")



scatter([log10(5)], [0.3],
    title= L"\lambda = 1",
    markercolor = :black,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlabel = L"\log_{10}\, D_{T}",
    ylabel = "H",
    xlim = (0.57,0.8),
    ylim = (0.15,0.45),
    grid = :none,
)
plot!(log10.([6,4,4,6,6]),[0.4,0.4,0.2,0.2,0.4],
    linestyle = :dash,
    linecolor = :black,
    label = "",
    fillalpha = 0.2,
    fillcolor = :grey,
    fillrange = 0.2,
)
savefig("errBarBL.pdf")


scatter([log10(5*(λ)^0.6)], [0.3],
    title= L"\lambda = 10",
    markercolor = :black,
    fontfamily = "Computer Modern",
    label = nothing,
    linewidth = 3,
    linestyle = :dash,
    xlabel = L"\log_{10}\,D_{10T}",
    ylabel = "H",
    xlim = (0.95,1.65),
    ylim = (0.15,0.45),
    grid = :none,
)
lxs = log10.(LinRange(4*(λ)^0.4,6*(λ)^0.4,100))
plot!(lxs,fill(0.2,100),
    linestyle = :dash,
    linecolor = :black,
    label = "",
    fillalpha = 0.2,
    fillcolor = :grey,
    fillrange = (lxs .-log10(4)) ./2,
)
plot!(log10.([4*(λ)^0.8,6*(λ)^0.8]),[0.4,0.4],
    linestyle = :dash,
    linecolor = :black,
    label = "",
)

plot!(ld-> (ld-log10(4))/2,log10(4*(λ)^0.4),log10(4*(λ)^0.8),
    linestyle = :dash,
    linecolor = :black,
    label = ""
)
lxs = log10.(LinRange(6*(λ)^0.4,6*(λ)^0.8,100))
plot!(lxs, (lxs .-log10(6)) ./2,
    linestyle = :dash,
    linecolor = :black,
    label = "",
    fillalpha = 0.2,
    fillcolor = :grey,
    fillrange = min.((lxs .-log10(4)) ./2, 0.4),
)

savefig("errBarBR.pdf")

## 2 groups pres

xs = LinRange(-100,100,200)

ys1 = rand(20)' .* (xs .- 2) .+ randn(20)' .- 5
plot([],[],
        label = "group 1",
        fontfamily = "Computer Modern",
        linecolor = palette(:default)[1],
        xlabel = "log t",
        ylabel = L"\log \delta^2",
        xlim = (-100,100),
)
plot!(xs,ys1,
    linecolor = palette(:default)[1],
    label = ""
)
ys2 = 0.5rand(20)' .* (xs .+ 4) .+ randn(20)' .+ 3
plot!(xs,ys2,
    linecolor = palette(:default)[2],
    label = ""
)
plot!([],[],
        label = "group 2",
        linecolor = palette(:default)[2],
)
savefig("lineSc2.pdf")