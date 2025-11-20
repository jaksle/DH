using Statistics, LinearAlgebra,CairoMakie
using HypothesisTests, ProgressMeter
include("../anDiffReg/AnDiffReg.jl")
using .AnDiffReg


##

α, D = 0.8, 1
n = 10^5 # number of trajectories
ln = 200 # trajectory length
dt = 0.01 # time interval
ts = dt*(1:ln)
d = 2

K = (s,t) -> D*(t^(α)+s^(α)-abs(s-t)^(α)) # FBM covariance
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U

ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

msd = tamsd([X ;;; Y]) # TA-MSD of 2D traj

##



w = 10
Ts = [ones(ln) log10.(ts[1:ln])]
S = (Ts[1:w,:]'*Ts[1:w,:])^-1 * Ts[1:w,:]'
lmsd = log10.(msd[1:w,:]) 
ols = S * lmsd
ols[1,:] .-= log10(2d)


gls, covGLS = fit_gls(msd, d, dt, ols[2,:])

(JarqueBeraTest(ols[1,:]).JB + JarqueBeraTest(ols[2,:]).JB)/10^5


## JB gaussianity
using JLD2


function f()


    d = 2
    N = 10
    α, D = 0.8, 1
    n = 10^5 # number of trajectories
    ln = 200 # trajectory length
    dt = 0.01 # time interval

    lns = 25:5:200
    ts = dt*(1:ln)
    w = 10
    Ts = [ones(ln) log10.(ts[1:ln])]
    S = (Ts[1:w,:]'*Ts[1:w,:])^-1 * Ts[1:w,:]'

    jbo = zeros(length(lns))
    jbos = zeros(length(lns))
    jbg = zeros(length(lns))
    jbgs = zeros(length(lns))
    ols = Matrix{Float64}(undef,2,n)
    gls = similar(ols)
    z = similar(ols)


    for j in 1:N 
        K = (s,t) -> D*(t^(α)+s^(α)-abs(s-t)^(α)) # FBM covariance
        A = cholesky(Symmetric([K(s,t) for s in ts, t in ts])).U

        ξ = randn(ln, n)
        X = A'*ξ
        ξ = randn(ln, n)
        Y = A'*ξ


        for (k,l) in enumerate(lns)
            msd = tamsd(@view X[1:l,:]) + tamsd(@view Y[1:l,:])
            lmsd = log10.(msd[1:w,:]) 
            ols .= S * lmsd
            ols[1,:] .-= log10(2d)

            gls .= fit_gls(msd,d,dt,ols[2,:])[1]

            jbos[k] += (JarqueBeraTest(ols[1,:]).JB + JarqueBeraTest(ols[2,:]).JB)/n
            ols .-= mean(ols,dims=2)
            c = Symmetric(cov(ols'))
            z .= sqrt(c)^-1*ols
            jbo[k] += (JarqueBeraTest(z[1,:]).JB + JarqueBeraTest(z[2,:]).JB)/n

            jbgs[k] += (JarqueBeraTest(gls[1,:]).JB + JarqueBeraTest(gls[2,:]).JB)/n
            gls .-= mean(gls,dims=2)
            c = Symmetric(cov(gls'))
            z .= sqrt(c)^-1*gls
            jbg[k] += (JarqueBeraTest(z[1,:]).JB + JarqueBeraTest(z[2,:]).JB)/n
            println((j,k))

        end
        jbo ./= N
        jbos ./= N
        jbg ./= N
        jbgs ./= N
    end

    #@save "gaussianity.jld2" lns jbo jbos jbg jbgs # PC WMat
end

f()
## 
using JLD2
@load "gaussianity.jld2" lns jbo jbos jbg jbgs 
n = 10^5

##
with_theme(theme_latexfonts()) do
fig = Figure()

fig
end

##

with_theme(theme_latexfonts()) do

fig = Figure(size=(800,400))

ax = Axis(fig[1,1],
    ylabel = "Jarque-Bera statistic",
    xlabel = "trajectory length",
    xticks = 25:25:200,
    yticks = 0.0002:0.0002:0.0008,
    limits = (20,205,nothing,nothing),
    title = L"\textbf{Non-Gaussianity of} $(\log_{10}\, \hat{D},\, \hat{\alpha})$"
)
scatter!(ax,lns,jbos,
    label = "OLS",
    color = :dodgerblue2, 
)
scatter!(ax,lns,jbgs,
    label = "GLS",
    color = :tomato,
    marker = :utriangle,
)
axislegend(ax,position = :rc)

ax2 = Axis(fig[1,2],
    ylabel = "α [1]",
    xlabel = L"$D$ [L$^2$/T$^\alpha$]",
    limits = (0.3,2.5,0.6,1.0),
    title = "Diffusivity errors in linear scale"
)
scatter!(ax2,10 .^ gls[1,1:1000], gls[2,1:1000],
    label = "GLS estimated parameters",
    color = :tomato,
    alpha = 0.7,
    markersize = 5,
)

mlogD, mA = mean(gls, dims = 2) 
cv = cov_gls(0.8, dt, ln, d)
C = sqrt(cv) # from the predicted covariance of the mean estimated parameters

# these are formulas for the confidence elipsis of 2D Gaussian
fx(t) = sqrt(5.99) * ( C[1,1]*cos(t) + C[1,2]*sin(t) ) + mlogD
fy(t) = sqrt(5.99) * ( C[2,1]*cos(t)+ C[2,2]*sin(t) ) + mA

ϕs = LinRange(0,2pi,200)
lines!(ax2, 10 .^ fx.(ϕs), fy.(ϕs),
    label = "error 95% confidence area",
    linestyle = :dash,
    color = :black
)

A = log(10) * cv[1,2] / cv[2,2]
B = log(10)*0 + (log(10))^2/2 * (cv[1,1] - cv[1,2]^2/cv[2,2])

αs = LinRange(0.5,1.2,200)

avD = @. 1/(1) * exp( A*(αs-α) + B)

lines!(ax2,avD,αs,
    color = :purple3,
    linewidth = 2,
    label = L"predicted $\langle\hat{D}\rangle_{\hat\alpha}$"
)

axislegend(ax2,position = :rb)
save("gaussianity.pdf",fig)

fig
end