using CairoMakie, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra
using KernelDensity, FFTW

include("funs.jl")


## sim data

ln = 100
n = 10^4
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

msd = Matrix{Float64}(undef,ln-1,n)

@showprogress for k in 1:n
    #al = rand(Uniform(0.4,1))
    al = (rand() < 1/2) ? rand(Uniform(0.4,0.6)) : rand(Uniform(0.8,1.0))
    lD = rand(Uniform(-1,1))

    f = (s,t) -> 10^lD*(t^(al)+s^(al)-abs(s-t)^(al)) 
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), 1)
    X = A'*ξ
    msd[:,k] .= estMSD(X,ln-1) 
end



## GLS prep

hs = 0.05:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(f(ts[i],ts[i])*f(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2 # 1 D
end


## msd fit

lmsd = log10.(msd)


l = 10 # window

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[1,:] .-= log10(4)


# GLS fit
w = 99 # window
bB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    #jmin, jmax = findfirst(hs .>= 0.2), findfirst(hs .>= 0.5)
    #j = max(j,jmin); j = min(j,jmax)
    gR = (Ts[1:w,:]'*errC[1:w,1:w,j]^-1*Ts[1:w,:])^-1*Ts[1:w,:]'*errC[1:w,1:w,j]^-1
    #gB[:,i] .= gR*lmsd[1:w,i]
    bB[:,i] .= gR*(lmsd[1:w,i] .- bias[1:w,j])
end

bB[1,:] .-= log10(2)


## simple deconv
nIter = 100

den = kde((bB[1,:],bB[2,:]),boundary = ((-1.5,1.5),(0.2,1.3)),npoints=(512,512))
#heatmap(den.x,den.y,den.density')

#surface(den.x,den.y,den.density')

D = 1 # mean(bB[1,:])
mA = 0.7 #mean(bB[2,:])
K = (s,t) -> D*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')


resStored = Array{Float64}(undef,length(den.x), length(den.y), 10)
kk = 1
#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
@showprogress for k in 1:nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
    if mod(k,10) == 0
        resStored[:,:,kk] .= res
        kk += 1
    end
end

##
j = 3
heatmap(den.x,den.y,resStored[:,:,j])
#surface(den.x,den.y,res)
#marg = vec(sum(res,dims=1))
#plot(marg)

#resO = copy(res)



#heatmap(den.x,den.y,res')

## long deconv main part
resIStored = Array{Float64}(undef,length(den.x), length(den.y), 10)

j = findfirst(den.y .>= 0.2)
j2 = findlast(den.y .< 1.2)
@showprogress for k in j:j2
    mA = den.y[k] 
    K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
    Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
    eM = (Ts'*Σ^-1*Ts)^-1
    nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

    ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
    ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
    #heatmap(den.x,den.y,ns')

    #dec = deconv(den.density,ns,-1)
    zs = den.density
    ins = reverse(ns)
    res = copy(zs)
    for i in 1:nIter
        d = real.(ifft( fft(res) .* fft(ns)))
        d[abs.(d) .< 10^-12] .= 10^-12
        res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
        if mod(i,10) == 0
            kk = i ÷ 10
            resIStored[:,k,kk] .= res[:,k]
        end
    end
    
end

mA = 1.2
K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
kk = 1
for i in 1:nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
    if mod(i,10) == 0
        resIStored[:,j2:end,kk] .= res[:,j2:end]
        resIStored[:,:,kk] ./= (sum(resIStored[:,:,kk])*step(den.x)*step(den.y))
        kk += 1
     end
end

#using JLD2
#@save "deconFull.jld2" B bB den resStored resIStored


## loading data
using CairoMakie, LaTeXStrings

using JLD2

#@load "deconFull.jld2"

resO = resStored[:,:,5]
resI = resIStored[:,:,5]

thDen = [( -1 <= x <= 1 && ( 0.4 <= y <= 0.6 || 0.8 <= y <= 1.0)) ? 1/(0.4*2) : 0. for x in den.x, y in den.y ]

sum((thDen .- den.density) .^2)*step(den.x)*step(den.y)
sum((thDen .- resO) .^2)*step(den.x)*step(den.y)
sum((thDen .- resI) .^2)*step(den.x)*step(den.y)

scatter(bB[1,:],bB[2,:],markersize=3)

heatmap(den.x,den.y,thDen)
heatmap(den.x,den.y,den.density)
heatmap(den.x,den.y,resO)
heatmap(den.x,den.y,resI)

surface(den.x,den.y,resI')

denMarg = vec(sum(den.density,dims=1))
denMarg .*= 1/(sum(denMarg)*step(den.y))

denMarg2 = vec(sum(resO,dims=1))
denMarg2 .*= 1/(sum(denMarg2)*step(den.y))

denMarg3 = vec(sum(resI,dims=1))
denMarg3 .*= 1/(sum(denMarg3)*step(den.y))


########################## figure
## top row


set_theme!(theme_latexfonts())

fig = Figure(size=(1000,800),
    fontsize = 16,
)
ga = GridLayout(fig[1, 1])
gb = GridLayout(fig[1, 2])
gc = GridLayout(fig[1:2,3])

xlab = L"$D$ [L$^2$/T$^\alpha$]"
ylab = L"{\alpha}"
xtickL = [L"10^{-1}",L"10^{-0.5}","1",L"10^{0.5}",L"10^{1}"]
xtick = (-1:0.5:1,xtickL)
lsize = 18

ax = Axis(ga[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Estimation results"
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
th = poly!(ax, Rect(-1,0.4,2,0.2),
    color = :silver,
    strokewidth = 1,
    strokecolor = :black,
)
poly!(ax, Rect(-1,0.8,2,0.2),
color = :silver,
    strokewidth = 1,
    strokecolor = :black,
)

gls = scatter!(ax,bB[1,:],bB[2,:],
    color = :red,
    alpha = 0.7,
    markersize = 2.5,
)
axislegend(ax,[th, MarkerElement(color = :red,alpha=0.7, marker=:circle, markersize = 8)],
    ["original distribution","GLS estimation"]
)
ax = Axis(ga[1,2],
    xlabel = L"histogram of $p_\alpha$",
    yticklabelsvisible = false,
    limits = (0,3.1,0.2,1.2)
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
hist!(ax,bB[2,:],normalization=:pdf,direction=:x,
    color = (:red,0.5),
    strokewidth = 1,
    strokecolor = :firebrick,
    #fillalpha = 0.5
)
lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
colsize!(ga, 1, Relative(4/5))
colgap!(ga,10)

ax2 = Axis(gb[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    title = "Density estimate"
    #ylabel = ylab,
)
ax2.xlabelsize = lsize
ax2.ylabelsize = lsize
ax2.titlesize = lsize
heatmap!(ax2,den.x,den.y,den.density,
    colormap = :thermal,
)

ax = Axis(gb[1,2],
    limits = (0,3.1,0.2,1.2),
    yticklabelsvisible = false,
    xlabel = L"density $p_\alpha$",
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax,denMarg,den.y,
    color = :orange,
)
colsize!(gb, 1, Relative(4/5))
colgap!(gb,10)


# bottom row


xlab = L"$D$ [L$^2$/T$^\alpha$]"
xtickL = [L"10^{-1}",L"10^{-0.5}","1",L"10^{0.5}",L"10^{1}"]
xtick = (-1:0.5:1,xtickL)
ylab = L"{\alpha}"

ga = fig[2, 1] = GridLayout()
gb = fig[2, 2] = GridLayout()


ax = Axis(ga[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    title = "Simple deconvolution",
    ylabel = ylab,
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
heatmap!(ax,den.x,den.y,resO,
    #colormap = :thermal,
    colorrange = (0,1.95),
)

ax = Axis(ga[1,2],
    limits = (0,nothing,0.2,1.2),
    xticks = [1,2,3],
    yticklabelsvisible = false,
    xlabel = L"density $p_\alpha$",
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax,denMarg2,den.y,
    color = :turquoise4,
)
colsize!(ga, 1, Relative(4/5))
colgap!(ga,10)


ax = Axis(gb[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    title = "Interpolated deconvolution",
    #ylabel = ylab,
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
heatmap!(ax,den.x,den.y,resI,
    #colormap = :thermal,
    colorrange = (0,1.95),
)

ax = Axis(gb[1,2],
    limits = (0,nothing,0.2,1.2),
    yticklabelsvisible = false,
    xlabel = L"density $p_\alpha$",
)
ax.xlabelsize = lsize
ax.ylabelsize = lsize
ax.titlesize = lsize
lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax,denMarg3,den.y,
    color = :turquoise4,
)
colsize!(gb, 1, Relative(4/5))
colgap!(gb,10)

# side

for (k,i) in enumerate([1,2,3,5,10])
    ax = Axis(gc[k,1],
        limits = (-1.2,1.2,0.2,1.2),
        aspect = 1,
    )
    hidedecorations!(ax)
    # save("deconTest.pdf",fig)
    heatmap!(ax,den.x,den.y,resIStored[:,:,i],
        #colormap = :thermal,
                colorrange = (0, 1.95),
    )
    text!(ax,0., 1.2, text = L"i = %$(i)0", color = :white, align = (:center,:top),fontsize = 20)
end
rowgap!(gc,7)
colsize!(fig.layout, 1, Relative(42/100))
colsize!(fig.layout, 2, Relative(42/100))
colsize!(fig.layout, 3, Relative(16/100))

#save("deconTest.pdf",fig)
fig


