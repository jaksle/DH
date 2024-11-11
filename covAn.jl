
using Plots

##
r(i,j,H) = 1/2*(abs(i)^(2H) + abs(j)^(2H) - abs(i-j)^(2H))
h(i,j,k,l,H) = r(i,j,H) + r(i-k,j-l,H) - r(i,j-l,H) - r(j,i-k,H)
g(n,k,H) = abs(n+k)^2H + abs(n-k)^2H - 2abs(n)^2H

tail(n,k,H) = 2H*(2H-1)*k^2*n^(2H-2)
apprG(n,k,H) = if n < 0
        apprG(-n,k,H)
    elseif n <= k
        g(0,k,H) + n/k*(g(k,k,H)-g(0,k,H))
    else
        tail(n,k,H)
    end

plot(n->g(n,5,0.35), 0, 10)
plot!(n->apprG(n,5,0.35), 0, 10)

plot(n->-g(n,50,0.35), 51, 500,
    xscale = :log10,
    yscale = :log10,
)
plot!(n -> -abs(n^(0.7-2))*g(51,50,0.35)/51^(0.7-2), 51,500)

plot!(n->2*0.35*25*(0.7-1)n^(0.7-2),5,100)

plot(cumsum(g(n,50,0.35)^2 for n in 1:500),
    #xscale = :log10,
    #yscale = :log10,
)
plot!(cumsum(apprG(n,50,0.35)^2 for n in 1:500),
)

plot!(cumsum(2*0.35*25*(0.7-1)n^(0.7-2) for n in 5:100))

plot(cumsum(g(n,5,0.35)-2*0.35*25*(0.7-1)n^(0.7-2) for n in 5:500))

plot!(n->2*0.35*25*(n)^(0.7-1)-2*0.35*25*5^(0.7-1),5,500)

##

plot(n->10^2H0*g(n/10,5,0.35), 1, 500)
plot(n->g(n,1,0.35), 0, 10)

sumCov = [sum(g(x,1,0.35)^2/k for x in LinRange(0,10,k)) for k in 10:500]
## 2D

heatmap(collect(g(i-j,50,0.35) for i in 51:500, j in 51:500))

plot(i->g(i, 50, 0.35)^2,1,500,
    #xscale = :log10,
    #yscale = :log10,
)
scatter!([50],[0])
plot(i->(g(0,50,0.35)-g(i, 50, 0.35)),1,500,
    xscale = :log10,
    yscale = :log10,
)
plot!(i->i*0.5, 1,50)
plot!(i->abs(i)^(0.7-1)*g(0,50,0.35), -50,50)