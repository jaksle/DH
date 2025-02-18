using Plots, MAT, ProgressMeter

##

file = matopen("interphase_traj_l100.mat")
matread("interphase_traj_l100.mat")

dat = read(file,"traj")

#X = read(file,"trajx")
#Y = read(file,"trajy")
X = dat[1:100,1:2:end]
Y = dat[1:100,2:2:end]

ln, n = size(X)
dt = 0.0567
##

msdX = Matrix{Float64}(undef,ln-1,n)
msdY = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msdX[:,i] .= estMSD(X[:,i],ln-1) 
    msdY[:,i] .= estMSD(Y[:,i],ln-1) 
end
lp(x) = x > 0 ? log10(x) : 10.
lmsdX = lp.(msdX)
lmsdY = lp.(msdY)
## OLS

ts = dt*(1:ln-1)
Ts = [ones(ln-1) log10.(ts)]
l = 10

Bx = Matrix{Float64}(undef, 2, n)
By = Matrix{Float64}(undef, 2, n)
for i in 1:n
    Bx[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsdX[1:l,i]
    By[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsdY[1:l,i]
end


## GLS prep

hs = 0.1:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
    errC[:,:,k] .= [theorCovEff(i,j,ln,K)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) for i in 1:ln-1, j in 1:ln-1]
end


## GLS fit

gBx = Matrix{Float64}(undef, 2, n)
gBy = Matrix{Float64}(undef, 2, n)
for i in 1:n
    jx = findfirst(hs .>= Bx[2,i]/2)
    jx === nothing && (jx = length(hs))
    gBx[:,i] .= (Ts'*errC[:,:,jx]^-1*Ts)^-1*Ts'*errC[:,:,jx]^-1*lmsdX[:,i]
    jy = findfirst(hs .>= By[2,i]/2)
    jy === nothing && (jy = length(hs))
    gBy[:,i] .= (Ts'*errC[:,:,jy]^-1*Ts)^-1*Ts'*errC[:,:,jy]^-1*lmsdY[:,i]
end

##

K = (s,t) -> 1*(t^(0.7)+s^(0.7)-abs(s-t)^(0.7))
Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1

##

scatter(gBx[1,:],gBy[1,:],
    markerstrokewidth=0,
    markersize=2.5,
    alpha = 0.3,
)


scatter(gBx[2,:],gBy[2,:],
    markerstrokewidth=0,
    markersize=2.5,
    alpha = 0.3,
)

## comparing variance, seems larger

dlD = gBx[1,:] .- gBy[1,:]

## correlation hist
dX = diff(X,dims = 1)
dY = diff(Y,dims = 1)

cs = [cor(dX[:,k],dY[:,k]) for k in 1:n]

H, D = 0.35, 1/2 # FBM parameters: Hurst index H and diffusivity D
n2 = 10^4 # number of trajectories
ln = 100 # trajectory length

K = (s,t) -> D*(t^(2H)+s^(2H)-abs(s-t)^(2H)) # depending on notation is D or D/2
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
η = randn(length(ts), n)
sX = A'*ξ
sY = A'*η
sdX = diff(sX,dims=1)
sdY = diff(sY,dims=1)

simcs = [cor(sdX[:,k],sdY[:,k]) for k in 1:n2]

## eigvals

emin = zeros(n)
emax = zeros(n)
semin = zeros(n)
semax = zeros(n)
for k in 1:n
    e = eigen([var(dX[:,k]) cov(dX[:,k],dY[:,k]); cov(dX[:,k],dY[:,k]) var(dY[:,k])])
    se = eigen([var(sdX[:,k]) cov(sdX[:,k],sdY[:,k]); cov(sdX[:,k],sdY[:,k]) var(sdY[:,k])])
    emin[k] = minimum(e.values)
    emax[k] = maximum(e.values)
    semin[k] = minimum(se.values)
    semax[k] = maximum(se.values)
end