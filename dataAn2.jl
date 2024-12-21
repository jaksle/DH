using Plots, MAT, ProgressMeter

## load

file = matopen("prl_trajectories_untreated.mat")

file = matopen("interphase_traj_l100.mat")
matread("interphase_traj_l100.mat")

dat = read(file,"traj")

#X = read(file,"trajx")
#Y = read(file,"trajy")
X = dat[1:100,1:2:end]
Y = dat[1:100,2:2:end]
dt = read(file,"t_step")

ln, n = size(X)


ts = dt*(1:ln-1)
Ts = [ones(ln-1) log10.(ts)]
l = 10


## msd

msd = Matrix{Float64}(undef,ln-1,n)
msdX = Matrix{Float64}(undef,ln-1,n)
msdY = Matrix{Float64}(undef,ln-1,n)
msdrX = Matrix{Float64}(undef,ln-1,n)
msdrY = Matrix{Float64}(undef,ln-1,n)

# rotated
rX = √2/2 * (X .- Y)
rY = √2/2 * (X .+ Y)


for i in 1:n
    msdX[:,i]  .= estMSD(X[:,i],ln-1) 
    msdY[:,i]  .= estMSD(Y[:,i],ln-1)
    msd[:,i]   .= msdX[:,i] .+ msdY[:,i]
    msdrX[:,i] .= estMSD(rX[:,i],ln-1) 
    msdrY[:,i] .= estMSD(rY[:,i],ln-1)
end

lmsd = log10.(msd)
lmsdX = log10.(msdX)
lmsdY = log10.(msdY)
lmsdrX = log10.(msdrX)
lmsdrY = log10.(msdrY)

## OLS

B = Matrix{Float64}(undef, 2, n)
BX = Matrix{Float64}(undef, 2, n)
BY = Matrix{Float64}(undef, 2, n)
BrX = Matrix{Float64}(undef, 2, n)
BrY = Matrix{Float64}(undef, 2, n)

R = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'

for i in 1:n
    B[:,i] .= R*lmsd[1:l,i]
    BX[:,i] .= R*lmsdX[1:l,i]
    BY[:,i] .= R*lmsdY[1:l,i]
    BrX[:,i] .= R*lmsdrX[1:l,i]
    BrY[:,i] .= R*lmsdrY[1:l,i]
end


## drift

mX = Matrix{Float64}(undef,10,n)
mY = Matrix{Float64}(undef,10,n)

for i in 1:n
    mX[:,i]  .= estMean(X[:,i],10) 
    mY[:,i]  .= estMean(Y[:,i],10)
end

##  drift plots

plot(mX[:,10])

plot(mY[:,1:100])

# drift 2D plot
scatter(mX[1,:], mY[1,:], markerstrokewidth=0,markersize=2, 
    label = "drift",
    xlabel = "x",
    ylabel = "y",
)

## variance

vX = Vector{Float64}(undef,n)
vY = Vector{Float64}(undef,n)

for i in 1:n
    vX[i]  = var(diff(X[:,i],dims=1))
    vY[i]  = var(diff(Y[:,i],dims=1))
end

scatter(vX, vY, markerstrokewidth=0,markersize=2, 
    label = "incr var",
    xlabel = "x",
    ylabel = "y",
)


# comp BM
cX = [sum(randn()^2 for _ in 1:ln-1) for k in 1:1000]
cY = [sum(randn()^2 for _ in 1:ln-1) for k in 1:1000]

scatter(cX, cY, markerstrokewidth=0,markersize=2, 
    label = "incr var",
    xlabel = "x",
    ylabel = "y",
)



## scatter plot

k = 202 # +
k = 222 # +
k = 218
k = 524
k = 235
plot(X[:,k],Y[:,k],
    axisratio=1,
    label = "traj no. 524"
)
scatter!([X[1,k]],[Y[1,k]],label="")
scatter!([X[100,k]],[Y[100,k]],label="")


##

scatter(BX[1,:],BY[1,:], markerstrokewidth=0,markersize=2, label = "",
    axisratio = 1,
    xlabel = "log10 D x",
    ylabel = "log10 D y",
    #color = cgrad([:red,:blue])[(1:n) ./ n]
)

scatter(B[1,:],B[2,:], markerstrokewidth=0,markersize=2, 
    label = "D along x axis",
    xlabel = "log10 D",
    ylabel = "H",
)





## rotated OLS


## rotated scatter

scatter(B[1,:],B[2,:], markerstrokewidth=0,markersize=2, 
    label = "D along 45 deg axis",
    ylabel = "α",
)

scatter(BrX[1,:],BrY[1,:], markerstrokewidth=0,markersize=2, label = "")

scatter(BrX[1,:],BrY[1,:], markerstrokewidth=0,markersize=2, label = "",
    axisratio = 1,
    xlabel = "D rx",
    ylabel = "D ry",
)


##

mX = mean(X,dims=2)
mY = mean(Y,dims=2)

tX = X .- mX
tY = Y .- mY

scatter(tX[:,1:500],tY[:,1:500],label="")


msdtX = Matrix{Float64}(undef,ln-1,n)
msdtY = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msdtX[:,i] .= estMSD(tX[:,i],ln-1) 
    msdtY[:,i] .= estMSD(tY[:,i],ln-1)
end

lmsdtX = log10.(msdtX)
lmsdtY = log10.(msdtY)

## rotated OLS


BtX = Matrix{Float64}(undef, 2, n)
BtY = Matrix{Float64}(undef, 2, n)
for i in 1:n
    BtX[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsdtX[1:l,i]
    BtY[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsdtY[1:l,i]
end

BtX[2,:] ./= 2
BtY[2,:] ./= 2

##

scatter(BtX[1,:],BtX[2,:], markerstrokewidth=0,markersize=2, label = "OLS")
scatter(BtY[1,:],BtY[2,:], markerstrokewidth=0,markersize=2, label = "OLS")

