using CairoMakie,  Distributions, LaTeXStrings, ProgressMeter, LinearAlgebra
using KernelDensity, FFTW

include("funs.jl")

## 1 population deconvolution

n = 10^4
ln = 100
dt = 0.0567
ts = dt*(1:ln)

lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

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


## GLS prep exact

n = 10^5
errCEx = Array{Float64}(undef,ln-1,ln-1,length(hs))
biasEx = Array{Float64}(undef,ln-1,length(hs))
msdTemp = Matrix{Float64}(undef,ln-1,n)

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) 
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    x = A'*ξ
    ξ = randn(length(ts), n)
    y = A'*ξ

    for i in 1:n
        msdTemp[:,i] .= estMSD(x[:,i],ln-1) .+ estMSD(y[:,i],ln-1)  # 2D
    end
    ltemp = log10.(msdTemp)
    errCEx[:,:,k] .= cov(ltemp')

    biasEx[:,k] .=  mean(ltemp,dims = 2) .- 2hs[k]*lts .- log10(4)  # 2D bias
end


## FBM sim

n = 10^4
H0, D0 = 0.35, 10^-3

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

l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)

# GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)
eB = Matrix{Float64}(undef, 2, n)
for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])

    gR = (Ts'*errCEx[:,:,j]^-1*Ts)^-1*Ts'*errCEx[:,:,j]^-1
    eB[:,i] .= gR*(lmsd[:,i] .- biasEx[:,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)
eB[1,:] .-= log10(4)

## num prep

K = (s,t) -> 2D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
eM = (Ts[1:l,:]'*Σ[1:l,1:l]^-1*Ts[1:l,:])^-1  # GLS
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1 # OLS

fn(t) = cos(t)*sqrt(5.99) # 95% elipse
gn(t) = sin(t)*sqrt(5.99)


den = kde((eB[1,:],eB[2,:]),boundary=((-3.4,-2.6),(0.4,1.0)),npoints=(500,500) )
den2 = kde((B[1,:],B[2,:]),boundary=((-3.4,-2.6),(0.4,1.0)),npoints=(500,500) )


nn = MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))

nn2 = MvNormal([den2.x[end÷2], den2.y[end÷2]], Symmetric(eM2))
ns2 = [pdf(nn2,[x,y]) for x in den2.x, y in den2.y]
ns2 = circshift(ns2,(length(den2.x)÷2,length(den2.y)÷2))

zs = den.density
ins = reverse(ns)
res = copy(zs)

zs2 = den2.density
ins2 = reverse(ns2)
res2 = copy(zs2)

for _ in 1:100
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))

    d2 = real.(ifft( fft(res2) .* fft(ns2)))
    d2[abs.(d2) .< 10^-12] .= 10^-12
    res2 .*= real.(ifft( fft(zs2 ./ d2) .* fft(ins2)))
end


## plot GLS

darkRed = colorant"#cc3434"

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200,400),figure_padding=(0,30,0,0),
        fontsize = 22,
    )
    # GLS

    xlab = [L"10^{%$i}" for i in -3.4:0.2:-2.6]
    xlab[end÷2+1] = L"10^{-3}"
    xname = L"$D$ [μm$^2$/s$^{\alpha}$]"
    ax = Axis(fig[1,1],
        xticks = (-3.4:0.2:-2.6, xlab),
        #xticklabelsvisible = false,
        limits = (-3.4,-2.6, 0.4,1),
        title = "Step 1: points and predicted errors",
        titlesize = 20,
        xlabelsize= 20,
        ylabelsize = 20,
        #xlabel = xname,
        ylabel = L"GLS ${\alpha}$",
        xgridvisible = false,
        ygridvisible = false,
    )
    gls = Makie.scatter!(ax,eB[1,:],eB[2,:],
        #markerstrokewidth=0,
        markersize=6,
        alpha = 0.15,
        color = :tomato, #darkRed,
        #label = "",

    )
    Makie.scatter!(ax, [log10(D0)],[2H0],
        marker='⨉',
        color=:black,
        markersize = 15,
    )

    C = sqrt(eM)
    θs = LinRange(0,2pi,200)
    xs =  @. C[1,1]*fn(θs)+C[1,2]*gn(θs) + log10(D0)
    ys = @. C[2,1]*fn(θs)+C[2,2]*gn(θs)+ 2H0
    conf = lines!(ax,xs,ys ,
        linewidth = 1.5,
        color = :black,
        linestyle = :dash,
        label = "",
    )


    ax2 = Axis(fig[1,2],
        xticks = (-3.4:0.2:-2.6, xlab),
        #yticklabelsvisible = false,
        xlabel = xname,
        limits = (-3.4,-2.6, 0.4,1),
        title = "Step 2: density estimate",
        titlesize = 20,
        xlabelsize= 20,
        ylabelsize = 20,
    )
    Makie.heatmap!(ax2, den.x,den.y,den.density,
        colormap = :thermal,
    )
    Makie.scatter!(ax2, [log10(D0)],[2H0],
        marker='⨉',
        color=:black,
        markersize = 15,
    )

    ax3 = Axis(fig[1,3],
        xticks = (-3.4:0.2:-2.6,xlab),
        #yticklabelsvisible = false,
        xlabel = xname,
        limits = (-3.4,-2.6, 0.4,1),
        title = "Step 3: deconvolution",
        titlesize = 20,
        xlabelsize= 20,
        ylabelsize = 20,
    )
    Makie.heatmap!(ax3, den.x,den.y,res)
    cross = Makie.scatter!(ax3, [log10(D0)],[2H0],
        marker='⨉',
        color=:black,
        markersize = 15,
    )
    #colgap!(fig.layout,50)
    save("decEx.pdf",fig)
    fig
end


## plot OLS

darkRed = colorant"#cc3434"

set_theme!(theme_latexfonts())

xlab = [L"10^{%$i}" for i in -3.4:0.2:-2.6]
xlab[end÷2+1] = L"10^{-3}"

xname = L"$D$ [μm$^2$/s$^{\alpha}$]"
fig = Figure(size=(1200,400),figure_padding=(0,30,0,0),
    fontsize = 22,
)
ax = Axis(fig[1,1],
    xticks = (-3.4:0.2:-2.6, xlab),
    limits = (-3.4,-2.6, 0.4,1),
    title = "Step 1: points and predicted errors",
    titlesize = 20,
    xlabelsize= 20,
    ylabelsize = 20,
    xlabel = xname,
    ylabel = L"OLS ${\alpha}$",
    xgridvisible = false,
    ygridvisible = false,
)
gls = Makie.scatter!(ax,B[1,:],B[2,:],
    #markerstrokewidth=0,
    markersize=6,
    alpha = 0.15,
    color = :dodgerblue2, #darkRed,
    label = "",

)
Makie.scatter!(ax, [log10(D0)],[2H0],
    marker='⨉',
    color=:black,
    markersize = 15,
)

C2 = sqrt(eM2)
θs = LinRange(0,2pi,200)
xs =  @. C2[1,1]*fn(θs)+C2[1,2]*gn(θs) + log10(D0)
ys = @. C2[2,1]*fn(θs)+C2[2,2]*gn(θs)+ 2H0
conf = lines!(ax,xs,ys ,
    linewidth = 1.5,
    color = :black,
    linestyle = :dash,
    label = "",
)


ax2 = Axis(fig[1,2],
    xticks = (-3.4:0.2:-2.6,xlab),
    #yticklabelsvisible = false,
    xlabel = xname,
    limits = (-3.4,-2.6, 0.4,1),
    title = "Step 2: density estimate",
    titlesize = 20,
    xlabelsize= 20,
    ylabelsize = 20,
)
Makie.heatmap!(ax2, den2.x,den2.y,den2.density,
    colormap = :thermal,
)
Makie.scatter!(ax2, [log10(D0)],[2H0],
    marker='⨉',
    color=:black,
    markersize = 15,
)

ax3 = Axis(fig[1,3],
    xticks = (-3.4:0.2:-2.6,xlab),
    #yticklabelsvisible = false,
    xlabel = xname,
    limits = (-3.4,-2.6, 0.4,1),
    titlesize = 20,
    xlabelsize= 20,
    ylabelsize = 20,
    title = "Step 3: deconvolution",
)
Makie.heatmap!(ax3, den2.x,den2.y,res2)
cross = Makie.scatter!(ax3, [log10(D0)],[2H0],
    marker='⨉',
    color=:black,
    markersize = 15,
)

colgap!(fig.layout,50)
rowgap!(fig.layout,30)

# axislegend(ax,[MarkerElement(color = :tomato, marker=:circle, alpha = 0.6, markersize = 12),  conf, cross,],["GLS","error 95% ellipse",L"exact ($D, \alpha$)"],
#     position = :lt,
# )
save("decEx2.pdf",fig)
fig
