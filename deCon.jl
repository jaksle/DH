
using Plots, FFTW, Distributions, LaTeXStrings


# Richardson-Lucy deconv

conv(zs,ns) = real.(ifft( fft(zs) .* fft(ns)))
function deconv(zs,ns,n)
    ins = reverse(ns)
    res = copy(zs)
    for _ in 1:n
        den = real.(ifft( fft(res) .* fft(ns)))
        res .*= real.(ifft( fft(zs ./ den) .* fft(ins)))
    end
    return res
end

ins = reverse(ns) # for Gauss no change

res = copy(zs)

for _ in 1:100
    den = real.(ifft( fft(res) .* fft(ns)))
    res .*= real.(ifft( fft(zs ./ den) .* fft(ins)))
end

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
Σ = 0.5*[1. 0.9;0.9 1.]
d = MvNormal(Σ)
ns = [pdf(d,[x,y]) for x in xs, y in xs]
ns = circshift(ns,(100,100))

p(x,y) = pdf(Gamma(2,1),x+1)*pdf(Normal(0,1),y)

zs = p.(xs',xs)
heatmap(zs)

ws = conv(zs,ns)

ys = deconv(ws,ns,100)
heatmap(ys)

## plots

p1 = heatmap(xs,xs,zs,
    fontfamily = "Computer Modern",
    axisratio = 1,
    xlim= (-2,5),
    ylim= (-3,3),
    cbar = :none,
    title = "original density",
    #xlabel = L"x",
    #ylabel = L"y",
)
contour!(xs,xs,zs, linecolor=:white,linewidth=0.5)
savefig("denOrg.pdf")

p2 = heatmap(xs,xs,ws,
    fontfamily = "Computer Modern",
    axisratio = 1,
    xlim= (-2,5),
    ylim= (-3,3),
    cbar = :none,
    title = "blurred density",
    #xlabel = L"x",
    #ylabel = L"y",
)
contour!(xs,xs,ws, linecolor=:white,linewidth=0.5)

savefig("denBlur.pdf")

p3 = heatmap(xs,xs,ys,
    fontfamily = "Computer Modern",
    axisratio = 1,
    xlim= (-2,5),
    ylim= (-3,3),
    cbar = :none,
    title = "reconstructed density",
    #xlabel = L"x",
    #ylabel = L"y",
)
contour!(xs,xs,ys, linecolor=:white,linewidth=0.5)
savefig("denRecon.pdf")

#plot([p1, p2, p3]...,layout = (1,3))


## k den test
using KernelDensity
n = 10^4
X, Y = randn(n),randn(n)

den = kde((X,Y))
pd = pdf(den)
heatmap(den.x,den.y,den.density)

den2 = deconv(den.density,ns,100)

heatmap(den2)


## parametric conv

X = randn(10^3) .+ 1
Y = X .+ .√abs.(X) .* randn(10^3)

ft = fit(Normal,Y)


## data deConv

using KernelDensity

lds = LinRange(-5,-0.5,500)
as = LinRange(-0.1,1.7,500)

den = kde((bB[1,:],bB[2,:]))

p1 = Plots.heatmap(den.x,den.y,den.density',
    fontfamily = "Computer Modern",
    #title = "Kernel density estimate",
    title = "original",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
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

#heatmap(den.x,den.y,den.density')

p2 = Plots.heatmap(den.x,den.y,res',
    fontfamily = "Computer Modern",
    #title = "Deconvolved density estimate",
    title = "deconvolved",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    #ylabel = L"α\ [1]",
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




plotly()
surface(den.x,den.y,den.density')
surface(den.x,den.y,res')

## porównanie gęstości brzegowych
res ./= (sum(res)*step(den.x)*step(den.y))
denMarg = vec(sum(den.density,dims=1))
denMarg .*= 1/(sum(denMarg)*step(den.y))
denMarg2 = vec(sum(res,dims=1))
denMarg2 .*= 1/(sum(denMarg2)*step(den.y))




p3 = Plots.plot(den.y,denMarg,
    fontfamily = "Computer Modern",
    label = "original density",
    ylabel = L"density $p{}_\alpha$",
    linecolor = palette(:plasma)[100],
    #xlabel = "α",
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
Plots.plot(p1,p2,p3,layout = l)

savefig("kdeComp.pdf")


##################################

## interpolation
using KernelDensity,FFTW

den = kde((bB[1,:],bB[2,:]))
den = dA2

heatmap(den.x,den.y,den.density')

# init run
mA = 0.15 #mean(bB[2,:])
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
for _ in 1:100
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

heatmap(den.x,den.y,res')


resI = copy(res)
j = findfirst(den.y .> 0.15)
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
    for _ in 1:100
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
for _ in 1:100
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end
resI[:,j2:end] .= res[:,j2:end]


heatmap(den.x,den.y,resI')



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