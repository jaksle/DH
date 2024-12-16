
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
A, B = 2C[1,1]-2C[1,2],2C[1,1]+2C[1,2]
f(t) = cos(t)/sqrt(A/5.99) # 95% elipse
g(t) = sin(t)/sqrt(B/5.99)

heatmap(xs,xs,ns)
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