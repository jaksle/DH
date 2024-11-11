using Plots, LinearAlgebra, LsqFit, Distributions


##
H0 = 0.35
D0  = 1.
ts = 1:5000
n = 10^4

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

X[:,1:5000] .*= rand(Exponential(),1,5000) .^ (H0)
X[:,5000:end] .= cumsum(randn(5000,5001),dims=1) .* rand(Exponential(),1,5001) .^ 0.5


##


n = 10^4

estH = zeros(n)
estD = zeros(n)
estK = zeros(n)

for k in 1:n
    m = estMSD(X[:,k],50)
    st = 10ts[1:50] # wybór skali
    f = curve_fit((t,p) -> p[1] .* t .^(2p[2]),
        st, m, [m[end]/st[end],0.5]
    )
    estD[k], estH[k] = f.param[1], f.param[2]
    estK[k] = abs(estD[k]) ^(1/2estH[k])
end

#mask = (3median(estD) .> estD .> 0) .& (0 .< estH .< 1)

mask = (estD .> 0) .& (0 .< estH .< 1)


cor(estD[mask],estH[mask])
cor(estK[mask],estH[mask])

cor(log.(estK[mask]), estH[mask])

##


n = 10^4

estH = zeros(n)
estK = zeros(n)

for k in 1:n
    m = estMSD(X[:,k],50)
    f = curve_fit((t,p) -> (abs(p[1]) .* t) .^(2p[2]),
        ts[1:50], m, [1.,0.5]
    )
    estK[k], estH[k] = abs(f.param[1]), f.param[2]
end

mask = (3median(estK) .> estK .> 0) .& (0 .< estH .< 1)

mask = (estK .> 0) .& (0 .< estH .< 1)

cor(estK[mask],estH[mask])

cor(log.(estK[mask]), estH[mask])


## msd err cov

H0 = 0.4
D0  = 1.
ts = 1:500
n = 10^4

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

M = Matrix{Float64}(undef,50, n)
L = similar(M)
for k in 1:n
    M[:,k] = estMSD(X[:,k],50)
    L[:,k] = log.(M[:,k])
    M[:,k] .-= K.(1:50,1:50)
    L[:,k] .-= log.(K.(1:50,1:50))
end

## msd err cov th

using ProgressMeter

incrCov(i,j,k,l) = begin 
    a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    K(a,b) + K(a-c,b-d) - K(a,b-d) - K(b,a-c)
end
theorCov(k,l) = begin # exact val
    tsN = length(ts)
    2/((tsN-k)*(tsN-l)) * sum( incrCov(i,j,k,l)^2 for j in l+1:tsN, i in k+1:tsN )
end
theorCov2(k) = begin
    tsN = length(ts)
    2/((tsN-k)*(tsN-k)) * sum(1/4* abs(abs(i-j-k)^2H0+abs(i-j+k)^2H0-2abs(i-j)^2H0)^2 for j in k+1:tsN, i in k+1:tsN )
end
theorCov3(k) = begin
    tsN = length(ts)
    2/((tsN-k)*(tsN-k)) * sum(abs(abs(i-j-k)^2-abs(i-j)^2H0)^2 for j in k+1:tsN, i in k+1:tsN )
end
theorCov4(k) = begin
    tsN = length(ts)
    2/((tsN-k)*(tsN-k)) * sum(1/4*abs(apprG(i-j,k,H0))^2 for j in k+1:tsN, i in k+1:tsN )
end
thC = @showprogress map(theorCov,1:50,1:50)
thC2 = theorCov2.(1:50)
thC3 = theorCov3.(1:50)
C = cov(M,dims=2)

plot(.√diag(C))
plot(.√diag(C) ./ (1:50))
plot(diag(C) ./ (1:50) .^(4H0+1) )
plot(thC2[1:50] ./ (1:50) .^(4H0+1) )

plot(thC2[1:10] ./ (1:10) .^2H0)

plot(thC[1:50] ./ (1:50) .^(2.65) )

plot(diag(C),xscale=:log10,yscale=:log10)
plot!((1:50) .^(2+4H0))