
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