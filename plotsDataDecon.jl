
using CairoMakie, FFTW, Distributions, LaTeXStrings
using KernelDensity

## data deConv



lds = LinRange(-5,-0.5,500)
as = LinRange(-0.1,1.7,500)

den = kde((bB[1,:],bB[2,:]))


D = 10 ^ -3 # mean(bB[1,:])
mA = 0.9 #mean(bB[2,:])
K = (s,t) -> D*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:30
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end



## margins

resPlot =res
# porównanie gęstości brzegowych
resPlot ./= (sum(resPlot)*step(den.x)*step(den.y))
denMarg = vec(sum(den.density,dims=1))
denMarg .*= 1/(sum(denMarg)*step(den.y))
denMarg2 = vec(sum(resPlot,dims=1))
denMarg2 .*= 1/(sum(denMarg2)*step(den.y))

##

set_theme!(theme_latexfonts())
fig = Figure(size = (900,400),
    fontsize = 22,
)
ax1 = Axis(fig[1,1],
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    ylabel = L"$α$",
    limits = (-5,-1, -0.1,1.2),
    xticks = [-4,-3,-2],
    yticks = 0:0.3:1.5,
    xtickformat = xs -> [L"10^{%$(Int(x))}" for x in xs],
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(3),
    #label = "OLS"
)

ax2 = Axis(fig[1,2],
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    limits = (-5,-1, -0.1,1.2),
    yticklabelsvisible = false,
    yticks = 0:0.3:1.5,
    xticks = [-4,-3,-2],
    xtickformat = xs -> [L"10^{%$(Int(x))}" for x in xs],
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(3),
    #ylabel = L"$α$",
    #label = "GLS"
)

ax3 = Axis(fig[1,3],
    xlabel = L"density $p{}_\alpha$",
    limits = (0, nothing, -0.1,1.2),
    yticks = 0:0.3:1.5,
    yticklabelsvisible = false,
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(3),
    #ylabel = L"$α$",
    #label = "GLS"
)

heatmap!(ax1,den.x,den.y,den.density, colormap = :magma,
    colorscale = sqrt,
)
hlines!(ax1,[0],linestyle=:dash,color=:white)

heatmap!(ax2, den.x,den.y,resPlot,
    colorscale = sqrt,
)
hlines!(ax2,[0],linestyle=:dash,color=:white)

lines!(ax3,denMarg,den.y, color = :purple,label = "original",
)
lines!(ax3,denMarg2,den.y, color = :teal, label = "deconvolved",
)
hlines!(ax3,[0],linestyle=:dash,color=:black)
axislegend(ax3)
colsize!(fig.layout, 3,Relative(0.22))
#save("dataDecon.pdf",fig)
fig

##




p1 = Plots.heatmap(den.x,den.y,den.density',
    fontfamily = "Computer Modern",
    #title = "Kernel density estimate",
    title = "original",
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    titlefont= 12,
    xguidefontsize = 10,
    yguidefontsize = 10,
    ylabel = L"α\ [1]",
    xticks = (-5:2:-1, [L"10^{%$s}" for s in -5:2:-1]),
    xlim = (-5,-1),
    ylim = (-0.1,1.4),
    yticks = [-0.1,0,0.2,0.4,0.6,0.8,1,1.2,1.4],
    colorbar = :none,
)
Plots.hline!(p1,[0],linestyle=:dash,color=:white,
    label = "",
)
#savefig("kde.pdf")


#heatmap(den.x,den.y,den.density')

p2 = Plots.heatmap(den.x,den.y,res',
    fontfamily = "Computer Modern",
    #title = "Deconvolved density estimate",
    title = "deconvolved",
    xlabel = L"$D$ [μm$^2$/s$^{\alpha}$]",
    #ylabel = L"α\ [1]",
    titlefont= 12,
    xguidefontsize = 10,
    yguidefontsize = 10,
    color = palette(:viridis),
    xticks = (-5:2:-1, [L"10^{%$s}" for s in -5:2:-1]),
    xlim = (-5,-1),
    ylim = (-0.1,1.4),
    yticks = [-0.1,0,0.2,0.4,0.6,0.8,1,1.2,1.4],
    clim=(0,2),
    colorbar = :none,
    #colorbar_ticks = 0:0.2:1.6,
)
Plots.hline!([0],linestyle=:dash,color=:white,
    label = "",
)

#savefig("kdeDeconv.pdf")

## joint plot




#plotly()
#surface(den.x,den.y,den.density')
#surface(den.x,den.y,res')






p3 = Plots.plot(den.y,denMarg,
    fontfamily = "Computer Modern",
    label = "original density",
    ylabel = L"density $p{}_\alpha$",
    linecolor = palette(:plasma)[100],
    #xlabel = "α",
    titlefont= 12,
    xguidefontsize = 10,
    yguidefontsize = 10,
    permute = (:x,:y),
    xlim = (-0.1,1.4),
    ylim = (0,3.1),
    linewidth = 1.5,
    xticks = [-0.1,0,0.2,0.4,0.6,0.8,1,1.2,1.4],
)
Plots.plot!(den.y,denMarg2,
    label = "deconvolved density",
    permute = (:x,:y),
    linewidth = 1,
    linecolor = palette(:viridis)[150],
)
Plots.hline!([0],linestyle=:dash,color=:black,alpha= 0.5,
    label = "",
)

#savefig("deconvMarg.pdf")

l = @layout [a{0.4w} b{0.4w} c{0.2w}]
#plot!(p1,legend=(0.5,0.3))
#plot!(p2,legend=(0.5,0.3))
#plot!(p3,legend=(0.5,0.3))
Plots.plot(p1,p2,p3,layout = l,
    titlefont= 12,
    xguidefontsize = 10,
    yguidefontsize = 10,
    size = (800,400),
)

savefig("kdeComp.pdf")



## 1D conv ex

xs = LinRange(-5,5,200)
ys = @. exp(-xs^2)
ns = circshift(ys,100)
ns ./= sum( ns )

zs = real.(ifft(fft(ys) .* fft(ns)))

plot(ys)
plot!(zs)

## 2D conv

xs = LinRange(-5,5,200)
im = @. max(abs(xs),abs(xs')) <= 1
ys = @. exp( -0.5*(xs-xs')^2 - (xs+xs')^2)
ns = circshift(ys,(100,100))
ns ./= sum(ns)

zs = real.(ifft(fft(im) .* fft(ns)))

heatmap(zs', 
    axisratio=1,
    xlim=(1,200),
)

#

## singular case ex

ws = real.(ifft(fft(ys) .* fft(ns)))

ins = reverse(circshift(ws,(100,100)))

res = copy(ys)

for _ in 1:100
    den = real.(ifft( fft(res) .* fft(ins)))
    res .*= real.(ifft( fft(ys ./ den) .* fft(ins)))
end

## ellipse

C =Σ^-1
A, B = 2C[1,1]-2C[1,2],2C[1,1]+2C[1,2] # tylko dla równych wariancji
f(t) = cos(t)/sqrt(A/5.99) # 95% elipse
g(t) = sin(t)/sqrt(B/5.99)



plot!(t->√2/2*(f(t)+g(t)) +0.31,t->√2/2*(-f(t)+g(t)),0,2pi,
    linewidth = 2,
    color = :black,
    linestyle = :dash,
    label = "noise 95% ellipse",
)


##


##################################

## interpolation
using KernelDensity,FFTW

den = kde((bB[1,:],bB[2,:]))

nIter = 100

heatmap(den.x,den.y,den.density')

# init run
mA = 0.1
K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

#heatmap(den.x,den.y,res')


resI = copy(res)
j = findfirst(den.y .> 0.1)
j2 = findlast(den.y .< 1.0)
@showprogress for k in j:j2
    mA = den.y[k] 
    K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
    Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
    eM = (Ts'*Σ^-1*Ts)^-1
    nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

    ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
    ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
    #heatmap(den.x,den.y,ns')

    #dec = deconv(den.density,ns,-1)
    zs = den.density
    ins = reverse(ns)
    res = copy(zs)
    for _ in 1:nIter
        d = real.(ifft( fft(res) .* fft(ns)))
        d[abs.(d) .< 10^-12] .= 10^-12
        res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
    end
    resI[:,k] .= res[:,k]
end

mA = 1.0 #mean(bB[2,:])
K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end
resI[:,j2:end] .= res[:,j2:end]

##

#linecolor = palette(:plasma)[100]
#linecolor = palette(:viridis)[150],


fig, ax, _ = heatmap(den.x,den.y,resI)



mDen = vec(sum(den.density,dims=1))
mDen ./= (sum(mDen) * step(den.y))
fig, ax, _ = lines(den.y, mDen, color = :purple, label = "original")
mRes = vec(sum(resI,dims=1))
mRes ./= (sum(mRes) * step(den.y))
lines!(ax, den.y, mRes, color = :teal, label = "deconvolved")
ax.title = "Interpolated deconvolution"
ax.xlabel = "α"
ax.ylabel = "density"
axislegend(ax)
save("dataDeconForRev.pdf",fig)
fig

## test better kde
dA = kde((bB[1,:],bB[2,:]), boundary = ((-5,-1),(-0.2,1.7)),npoints=(256,256))
dA2 = kde((bB[1,:],bB[2,:]),bandwidth = (0.1,0.03),boundary = ((-5,-1),(-0.2,1.7)),npoints=(256,256))

mask = bB[2,:] .<= 0.05
lm = count(mask)
dB = kde((bB[1,mask],bB[2,mask]),bandwidth = (0.1,0.01),boundary = ((-5,-1),(-0.2,1.7)),npoints=(256,256))
dC = kde((bB[1,.!mask],bB[2,.!mask]),boundary = ((-5,-1),(-0.2,1.7)),npoints=(256,256))
dF = @. dB.density * lm/n + dC.density * (1-lm/n)
heatmap(dA.x,dA.y,dA2.density',
    ylim = (-0.15,1.5)
)

heatmap(dB.x,dB.y,dF',
    ylim = (-0.15,1.5)
)

denMarg = vec(sum(dA2.density,dims=1))
denMarg .*= 1/(sum(denMarg)*step(dA2.y))

j = findlast(den.y .<=0)
sum(denMarg2[1:j])*step(den.y)

count(bB[2,:] .<= 0)/n

mask = bB[2,:] .< 0.2

denS = kde((bB[1,mask],bB[2,mask]))

heatmap(den.x,den.y,den.density')
heatmap!(denS.x,denS.y,denS.density')