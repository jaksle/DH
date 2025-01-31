
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

## 1D conv

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

## singular case

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

X, Y = randn(200),randn(200)

den = kde((X,Y))
pd = pdf(den,xs,xs)
heatmap(den.density)

den2 = deconv(pd,ns,100)

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

heatmap(den.x,den.y,den.density')

D = 10 ^mean(bB[1,:])
mA = mean(bB[2,:])
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
for _ in 1:50
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

heatmap(den.x,den.y,den.density')
heatmap(den.x,den.y,res',
    clim=(0,2)
)

plotly()
surface(den.x,den.y,den.density')
surface(den.x,den.y,res')


# histogram - nie działa
histogram2d(bB[1,:],bB[2,:],bins=(50,50),normalize=:pdf)

nb = 50
lds = LinRange(-5,-0.5,nb+1)
As = LinRange(-0.1,1.7,nb+1)

hist = [count((lds[i] .< B[1,:] .< lds[i+1]) .& (As[j] .< B[2,:] .< As[j+1])) for i in 1:nb, j in 1:nb]

heatmap(hist')

ns2 = [pdf(g,[x+step(lds)/2,y+step(As)/2]) for x in lds[1:end-1], y in As[1:end-1]]
ns2 = circshift(ns2,(nb÷2,nb÷2))

zs = Float64.(hist)
ins = reverse(ns2)
res = copy(zs)

for _ in 1:20
    d = real.(ifft( fft(res) .* fft(ns2)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end