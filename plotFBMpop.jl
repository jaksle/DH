using Plots, MAT, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")

##

n = 10^4
ln = 100
dt = 0.0567
ts = dt*(1:ln)

## GLS prep

hs = 0.05:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    f = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

@showprogress for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

## FBM sim

H0, D0 = 0.35, 10^-3 

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

## msd fit

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]
l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end
B[1,:] .-= log10(4)

## GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    #cR = (Ts[l+1:end,:]'*errC[l+1:end,l+1:end,j]^-1*Ts[l+1:end,:])^-1*Ts[l+1:end,:]'*errC[l+1:end,l+1:end,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
    #cB[:,i] .= cR*(lmsd[l+1:end,i] .- bias[l+1:end,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)

##


scatter(gB[1,:],gB[2,:], #biased
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.1,
    color = palette(:default)[1],
    label = "",
)
mlD = mean(gB[1,:])
mA = mean(gB[2,:])
scatter!([mlD],[mA],marker=:xcross,color=palette(:default)[1],markersize = 5,markerstrokewidth=2,label="")

f(t) = cos(t)*sqrt(5.99) # 95% elipse
g1(t) = sin(t)*sqrt(5.99)
C = sqrt(cov(gB'))
 
plot!(t->C[1,1]*f(t)+C[1,2]*g1(t) + mlD,t->C[2,1]*f(t)+C[2,2]*g1(t)+ mA ,0,2pi,
    linewidth = 1,
    linestyle = :dash,
    color = palette(:default)[1],
    label = "",
)

scatter!(bB[1,:],bB[2,:],
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.1,
    color = palette(:default)[2],
    label = "",
)
mlD = mean(bB[1,:])
mA = mean(bB[2,:])
scatter!([mlD],[mA],marker=:xcross,color=palette(:default)[2],markerstrokewidth=2,markersize = 5,label="")

f(t) = cos(t)*sqrt(5.99) # 95% elipse
g1(t) = sin(t)*sqrt(5.99)
C = sqrt(cov(bB'))
 
plot!(t->C[1,1]*f(t)+C[1,2]*g1(t) + mlD,t->C[2,1]*f(t)+C[2,2]*g1(t)+ mA ,0,2pi,
    linewidth = 1,
    color = palette(:default)[2],
    linestyle = :dash,
    label = "",
)


## density

using KernelDensity, FFTW

#lds = LinRange(-5,-0.5,500)
#as = LinRange(-0.1,1.7,500)

den = kde((bB[1,:],bB[2,:]))

heatmap(den.x,den.y,den.density',
    fontfamily = "Computer Modern",
    title = "Kernel density estimate",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
    #xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    #xlim = (-5,-1),
    #ylim = (-0.1,1.4),
)


D = D0 # mean(bB[1,:])
mA = 2H0 #mean(bB[2,:])
K = (s,t) -> D*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:200
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

heatmap(den.x,den.y,den.density')

heatmap(den.x,den.y,res',
    fontfamily = "Computer Modern",
    title = "Deconvolved density estimate",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
    #xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    #xlim = (-5,-1),
    #ylim = (-0.1,1.4),
    #clim=(0,2)
)