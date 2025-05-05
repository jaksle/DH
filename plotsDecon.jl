using CairoMakie,  Distributions, LaTeXStrings, ProgressMeter, LinearAlgebra
using KernelDensity, FFTW

include("funs.jl")

## 1 population deconvolution

n = 10^4
ln = 100
dt = 1
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

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

##

H0, D0 = 0.35, 1

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ


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

# GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)

## num prep

K = (s,t) -> 2D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
eM = (Ts[1:l,:]'*Σ[1:l,1:l]^-1*Ts[1:l,:])^-1  # GLS
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1 # OLS

f(t) = cos(t)*sqrt(5.99) # 95% elipse
g1(t) = sin(t)*sqrt(5.99)


den = kde((bB[1,:],bB[2,:]),boundary=((-0.2,0.2),(0.4,1.0)),npoints=(500,500) )


nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))

zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:100
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end


## plot

darkRed = colorant"#cc3434"

with_theme(theme_latexfonts()) do
fig = Figure(size=(1200,400))
xlab = [L"10^{%$i}" for i in -0.15:0.05:0.15]
xlab[end÷2+1] = L"1"
ax = Axis(fig[1,1],
    xticks = (-0.15:0.05:0.15,xlab),
    limits = (-0.15,0.15, 0.4,1),
    title = "Step 1: points and predicted errors",
    titlesize = 20,
    xlabelsize= 20,
    ylabelsize = 20,
    xlabel = L"\hat{D}",
    ylabel = L"\hat{\alpha}",
    xgridvisible = false,
    ygridvisible = false,
)
gls= scatter!(ax,bB[1,:],bB[2,:],
    #markerstrokewidth=0,
    markersize=8,
    alpha = 0.15,
    color = darkRed,
    #label = "",

)
scatter!(ax, [log10(D0)],[2H0],
    marker='⨉',
    color=:black,
    markersize = 15,
)

C = sqrt(eM)
θs = LinRange(0,2pi,200)
xs =  @. C[1,1]*f(θs)+C[1,2]*g1(θs) + log10(D0)
ys = @. C[2,1]*f(θs)+C[2,2]*g1(θs)+ 2H0
conf = lines!(ax,xs,ys ,
    linewidth = 1.5,
    color = :black,
    linestyle = :dash,
    label = "",
)


ax2 = Axis(fig[1,2],
    xticks = (-0.15:0.05:0.15,xlab),
    xlabel = L"\hat{D}",
    limits = (-0.15,0.15, 0.4,1),
    title = "Step 2: density estimate",
    titlesize = 20,
    xlabelsize= 20,
    ylabelsize = 20,
)
heatmap!(ax2, den.x,den.y,den.density,
    colormap = :thermal,
)
scatter!(ax2, [log10(D0)],[2H0],
    marker='⨉',
    color=:black,
    markersize = 15,
)

ax3 = Axis(fig[1,3],
    xticks = (-0.15:0.05:0.15,xlab),
    xlabel = L"\hat{D}",
    limits = (-0.15,0.15, 0.4,1),
    title = "Step 3: deconvolution",
    titlesize = 20,
    xlabelsize= 20,
    ylabelsize = 20,
)
heatmap!(ax3, den.x,den.y,res)
cross = scatter!(ax3, [log10(D0)],[2H0],
    marker='⨉',
    color=:black,
    markersize = 15,
)

axislegend(ax,[MarkerElement(color = darkRed, marker=:circle, alpha = 0.5, markersize = 12),  conf, cross,],["GLS","error 95% ellipse",L"exact ($D, \alpha$)",])
fig
save("decEx.pdf",fig)
end

