using Plots, LinearAlgebra, ProgressMeter


H0 = 0.35
D0  = 1.
n = 10^4
m = 200
ts = 1:ln

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

msd = Matrix{Float64}(undef,m-1,n)

for i in 1:n
    msd[:,i] .= estMSD(X[:,i],m-1)
end

lmsd = log.(msd)

##
Ts = [ones(m) log.(ts[1:m])]
vlD = Vector{Float64}(undef,90)
v2H = similar(vlD)


@showprogress for k in 1:90
    M = k + 4
    R = (Ts[1:M,:]'*Ts[1:M,:])^-1*Ts[1:M,:]'
    B = R * lmsd[1:M,:]
    vlD[k] = var(B[1,:])
    v2H[k] = var(B[2,:])
end


##
vlD = Vector{Float64}(undef,20)
v2H = similar(vlD)

for k in 1:20
    M = k+1
    R = (Ts[1:M,:]'*Ts[1:M,:])^-1*Ts[1:M,:]'
    B = R * lmsd[1:M,:]
    vlD[k] = var(B[1,:])
    v2H[k] = var(B[2,:])
end


## gls

thcM = @showprogress [theorCov(i,j,m,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) for i in 1:50, j in 1:50]

gvlD = Vector{Float64}(undef,50)
gv2H = similar(gvlD)
for k in 1:49
    M = k+1
    gR = (Ts[1:M,:]'*thcM[1:M,1:M]^-1*Ts[1:M,:])^-1*Ts[1:M,:]'*thcM[1:M,1:M]^-1
    gB = gR * lmsd[1:M,:]
    gvlD[k] = var(gB[1,:])
    gv2H[k] = var(gB[2,:])
end

