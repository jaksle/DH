using Statistics, Distributions, StatsBase, FourierAnalysis
using SpecialFunctions, LinearAlgebra, FFTW
using Plots
using LsqFit

##


H0 = 0.35
D0  = 1.
ts = 1:500
n = 10^5

spec(f,H,D) = D* (gamma(2H+1)*sin(pi*H)) / (2pi)^(2H-1) *  f^(1-2H)

function specDiscr(f,H,D)
    γ = gamma(2H+1)*sin(pi*H)
    D*2γ*(1-cos(2pi*f)) *(2pi)^(-1-2H) * (zeta(1+2H,1-f) + zeta(1+2H,f))
end

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ



dX = diff(X,dims=1) 


sr, wl = 1, 100
S = spectra(dX[:,1], sr, wl; tapering=rectangular)

plot(S)

##

estH = zeros(n)
estD = zeros(n)

f = fftfreq(ts[end]-1,1)[2 : end÷4]
avS = zeros(size(f))

for k in 1:n
    #S = spectra(dX[:,k], sr, wl; tapering=rectangular)
    S =  norm.(fft(dX[:,k])[2 : end÷4] ) .^2
    avS .+= S
    #M(x,p) = p[1] .* x .^(1-2p[2])
    M2(x,p) = @. specDiscr(x,p[2],p[1])*ts[end]
    fit = curve_fit(M2,f ,S, [1.,0.5])
    estH[k] = fit.param[2]
    h = estH[k]
    estD[k] = fit.param[1] #* (2pi)^(2h-1) / (gamma(2h+1)*sin(pi*h)) *  1/ts[end]
end

avS ./= n
##

M(x,p) = p[1] .* x .^(1-2p[2])
fit = curve_fit(M,f ,avS, [1.,0.5])
plot(f, avS, label = "data")
#plot!(f, M(f,fit.param) )
plot!(f->spec(f,H0,1)*ts[end], f[1],f[end])
plot!(f->specDiscr(f,H0,1)*ts[end],f[1],f[end],
    #xscale = :log10,
    #yscale = :log10,
)

#savefig("plot.png")

#plot(f->specDiscr(f,H0,1)*ts[end]/2,f[1],f[end])

##

scatter((estD), estH, 
    xlim = (0.5,1.5),
    ylim = (0.2,0.5),
    label = nothing,
    markersize = 0.5,
    c = :black,
    markerstrokewidth = 0,
    xlabel = "diffusivity [distance²/time²ᴴ]",
    ylabel = "Hurst exponent [none]",
)

##  H = 0.6

using KernelDensity

p = kde((estD,estH), boundary = ((0.5,1.5),(0.4,0.9)))

xs = LinRange(0.4,1.5,1000)
ys = LinRange(0.4,0.9,1000)
den = pdf(p,xs,ys)

lg(x) = x > 0 ? log10(x) : NaN
heatmap(xs,ys,den', # H = 0.8
    xlim = (0.6,1.4),
    ylim = (0.45,0.8),
    color = cgrad(:magma,scale=:exp),
    nlevels = 100,
    xlabel = "diffusivity [distance²/time²ᴴ]",
    ylabel = "Hurst exponent [none]",
)

##

##  H = 0.35

using KernelDensity

xl, yl = (0.5,1.6), (0.16,0.5)
p = kde((estD,estH), boundary = (xl,yl))

xs = LinRange(xl[1],xl[2],1000)
ys = LinRange(yl[1],yl[2],1000)
den = pdf(p,xs,ys)

lg(x) = x > 0 ? log10(x) : NaN
heatmap(xs,ys,den', # H = 0.8
    xlim = xl,
    ylim = yl,
    color = cgrad(:magma,scale=:exp),
    nlevels = 100,
    xlabel = "diffusivity [distance²/time²ᴴ]",
    ylabel = "Hurst exponent [none]",
)

##
savefig("specDH35.pdf")
##

p = kde((estD,estH), boundary = ((0.8,1.8),(0.1,0.5)))

xs = LinRange(0.8,1.8,1000)
ys = LinRange(0.1,0.5,1000)
den = pdf(p,xs,ys)


heatmap(xs,ys,den', # H = 0.35
    xlim = (0.8,1.8),
    ylim = (0.15,.45),
    color = cgrad(:magma,scale=:exp),
    nlevels = 100,
    xlabel = "diffusivity [distance²/time²ᴴ]",
    ylabel = "Hurst exponent [none]",
)


##
savefig("specDH35.pdf")
##


mask = (2. .> estD .> 0.5) .& (estH .< 1)
cor(lg.(estD[mask]),estH[mask])



