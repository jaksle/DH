using Plots, Statistics, LinearAlgebra, Distributions
using ProgressMeter

## FBM sim

H0 = 0.35
D0  = 1.
ln = 200
ts = 1:ln
n = 10000

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ


## msd 

m = 30

function estMSD(X,k)
    n = length(X)
    d = Vector{Float64}(undef,k)

    for j in 1:k
        d[j] = mean( (X[i]-X[i+j])^2 for i in 1:n-j  )
    end
    return d
end

msd = Matrix{Float64}(undef,m,n)

for i in 1:n
    msd[:,i] .= estMSD(X[:,i],m)
end

lmsd = log.(msd)
e = @. msd - D0*(ts[1:m])^(2H0)
le = @. log(msd) - log( D0*(ts[1:m])^(2H0) )

## covariance

ce = cov(e')
cle = cov(le')

incrCov(i,j,k,l) = begin 
    a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    K(a,b) + K(a-c,b-d) - K(a,b-d) - K(b,a-c)
end
theorCov(k,l,n) = begin
    n
    2/((n-k)*(n-l)) * sum( incrCov(i,j,k,l)^2 for j in l+1:n, i in k+1:n )
end

thC = theorCov.(1:m,1:m,ln)
thlC = @. thC ./ (D0*ts[1:m]^(2H0))^2

plot(diag(ce))
plot!(thC,linestyle=:dash)

plot(diag(cle))
plot!(thlC,linestyle=:dash)


## regression

Ts = [ones(m) log.(ts[1:m])]
B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts'*Ts)^-1*Ts'*lmsd[:,i]
end

cB = cov(B')

M = (Ts'*Ts)^-1*Ts'
thcM = @showprogress [theorCov(i,j,ln)/(D0^2 * ts[i]^(2H0)*ts[j]^(2H0)) for i in 1:m, j in 1:m]

thcB = M*thcM*M'

## gls regression

Ts = [ones(m) log.(ts[1:m])]
gB = Matrix{Float64}(undef, 2, n)

#thcM = @showprogress [theorCov(i,j,ln)/(D0^2 * ts[i]^(2H0)*ts[j]^(2H0)) for i in 1:m, j in 1:m]

for i in 1:n
    gB[:,i] .= (Ts'*thcM^-1*Ts)^-1*Ts'*thcM^-1*lmsd[:,i]
end

cgB = cov(gB')

thcgB = (Ts'*thcM^-1*Ts)^-1

## bootstrap gls

hs = 0.2:0.01:0.49
K(s,t,H) = 1/2*(t^(2H)+s^(2H)-abs(s-t)^(2H))
incrCov(i,j,k,l,H) = begin 
    a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    K(a,b,H) + K(a-c,b-d,H) - K(a,b-d,H) - K(b,a-c,H)
end
theorCov(k,l,n,H) = begin
    n
    2/((n-k)*(n-l)) * sum( incrCov(i,j,k,l,H)^2 for j in l+1:n, i in k+1:n )
end


errC = Array{Float64}(undef,m,m,length(hs))
@showprogress for k in eachindex(hs)
    errC[:,:,k] .= [theorCov(i,j,ln,hs[k])/(1^2 * ts[i]^(2hs[k])*ts[j]^(2hs[k])) for i in 1:m, j in 1:m]
end

##

gB2 = Matrix{Float64}(undef, 2, n)
for i in 1:n
    v = (Ts'*Ts)^-1*Ts'*lmsd[:,i]
    estD = exp(v[1])
    estH = v[2]/2
    j = findfirst(hs .>= estH)
    j === nothing && (j = length(hs))
    gB2[:,i] .= (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1*lmsd[:,i]
end

cgB2 = cov(gB2')