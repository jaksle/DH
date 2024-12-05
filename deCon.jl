
using Plots, FFTW


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

## Richardson-Lucy deconv

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

## singular case

ws = real.(ifft(fft(ys) .* fft(ns)))

ins = reverse(circshift(ws,(100,100)))

res = copy(ys)

for _ in 1:100
    den = real.(ifft( fft(res) .* fft(ins)))
    res .*= real.(ifft( fft(ys ./ den) .* fft(ins)))
end

##

using Distributions

p(x,y) = pdf(Gamma(2,1),x+1)*pdf(Normal(0,1),y)

zs = p.(xs',xs)
heatmap(zs)

ws = conv(zs,ns)

xs = deconv(ws,ns,100)
heatmap(xs)