
using BenchmarkTools

function theorCovEff(k,l,ln,K)
    if k > l
        k, l = l, k
    end
    N1 = h-> ln-l-h+1
    N2 = h-> 
        if h <= l-k+1
            ln-l
        else
            ln-k-h+1
        end
    return 2/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov(1,h,k,l,K)^2 for h in 2:ln-l) + sum( N2(h)*incrCov(h,1,k,l,K)^2 for h in 1:ln-k ) )
end


H0 = 0.35
D0  = 1.
ln = 100
ts = 1:ln
n = 10000

K(s,t) = D0/2*(abs(t)^(2H0)+abs(s)^(2H0)-abs(s-t)^(2H0))



S1 = [theorCov(k,l,ln,K) for k in 1:50, l in 1:50]
S2 = [theorCovEff(k,l,ln,K) for k in 1:50, l in 1:50] 

## 
ln = 7
k = 1
l = 3

C = [incrCov(i,j,k,l,K)^2 for i in 1:ln-k, j in 1:ln-l]

f1 = h-> ln-l-h+1
f2 = h-> 
    if h <= l-k+1
        ln-l
    else
        ln-k-h+1
    end

scatter(f1.(2:ln-l))

scatter(f2.(1:ln-k))

sum( incrCov(i,j,k,l,K)^2 for j in l+1:ln, i in k+1:ln )

theorCov(k,l,ln,K)
theorCovEff(k,l,ln,K)
##

