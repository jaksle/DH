using Statistics, LinearAlgebra,CairoMakie
using HypothesisTests, ProgressMeter
include("../anDiffReg/AnDiffReg.jl")
using .AnDiffReg


##

α, D = 0.8, 1 # FBM parameters: Hurst index H and diffusivity D
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

#msd = tamsd([X ;;; Y]) # TA-MSD of 2D traj

##



ols, covOLS = fit_ols(msd, d, dt)  # here 10% default window is used
gls, covGLS = fit_gls(msd, d, dt, ols[2,:]) 

## JB gaussianity


XY = [X ;;; Y]
lns = 25:5:150
ts = dt*(1:ln)
w = 10
Ts = [ones(ln) log10.(ts[1:ln])]
S = (Ts[1:w,:]'*Ts[1:w,:])^-1 * Ts[1:w,:]'
jbo = Vector{Float64}(undef,length(lns))
jbos = similar(jbo)
jbg = similar(jbo)
jbgs = similar(jbo)

for (k,l) in enumerate(lns)
    msd = tamsd(XY[1:l,:,:])
    lmsd = log10.(msd) 
    ols = S * lmsd[1:w,:]
    ols[1,:] .-= log10(2d)

    gls, _ = fit_gls(msd,d,dt,ols[2,:])

    jbos[k] = JarqueBeraTest(ols[1,:]).JB + JarqueBeraTest(ols[2,:]).JB
    ols .-= mean(ols,dims=2)
    c = cov(ols')
    z = c^(-1/2)*ols
    jbo[k] = JarqueBeraTest(z[1,:]).JB + JarqueBeraTest(z[2,:]).JB

    jbgs[k] = JarqueBeraTest(gls[1,:]).JB + JarqueBeraTest(gls[2,:]).JB
    gls .-= mean(gls,dims=2)
    c = cov(gls')
    z = c^(-1/2)*gls
    jbg[k] = JarqueBeraTest(z[1,:]).JB + JarqueBeraTest(z[2,:]).JB
    println(k)

end

##
fig = Figure()
ax = Axis(fig[1,1],
    ylabel = "Jarque-Bera statistic",
    xlabel = "trajectory length",
    #limits = (0,3,0.5,1.1),
    title = L"Non-Gaussianity of $(\log_{10}\, \hat{D},\, \hat{\alpha})$"
)
lines!(ax,lns,jbo ./ n,
    label = "OLS",
    color = :dodgerblue2, 
)
lines!(ax,lns,jbg ./ n,
    label = "GLS",
    color = :tomato,
)
axislegend()
fig

##

with_theme(theme_latexfonts()) do
fig = Figure(size=(800,400))
ax = Axis(fig[1,1],
    ylabel = "α [1]",
    xlabel = L"$D$ [L$^2$/T$^\alpha$]",
    limits = (0,3,0.5,1.1),
    title = "Diffusivity errors in linear scale"
)
scatter!(ax,10 .^ gls[1,1:1000], gls[2,1:1000],
    label = "GLS estimated parameters",
    markersize = 5,
)

mlogD, mA = mean(gls, dims = 2) 
cv = cov_gls(mA, dt, ln, d)
C = sqrt(cv) # from the predicted covariance of the mean estimated parameters

# these are formulas for the confidence elipsis of 2D Gaussian
fx(t) = sqrt(5.99) * ( C[1,1]*cos(t) + C[1,2]*sin(t) ) + mlogD
fy(t) = sqrt(5.99) * ( C[2,1]*cos(t)+ C[2,2]*sin(t) ) + mA

ϕs = LinRange(0,2pi,200)
lines!(ax, 10 .^ fx.(ϕs), fy.(ϕs),
    label = "error 95% confidence area",
    linestyle = :dash,
    color = :black
)

A = log(10) * cv[1,2] / cv[2,2]
B = log(10)*0 + (log(10))^2/2 * (cv[1,1] - cv[1,2]^2/cv[2,2])

αs = LinRange(0.5,1.2,200)

avD = @. 1/(1) * exp( A*(αs-α) + B)

lines!(ax,avD,αs,
    color = :red,
    linewidth = 2,
    label = L"predicted $\langle\hat{D}\rangle_{\hat\alpha}$"
)

axislegend(ax,position = :rb)


ax = Axis(fig[1,2],
)
scatter!(ax,10 .^ gls[1,1:1000], gls[2,1:1000],
    label = "GLS estimated parameters",
    markersize = 5,
)
fig
end