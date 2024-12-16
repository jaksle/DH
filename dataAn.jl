using Plots, MAT, ProgressMeter

##

file = matopen("prl_trajectories_untreated.mat")
matread("prl_trajectories_untreated.mat")

X = read(file,"trajx")
Y = read(file,"trajy")
dt = read(file,"t_step")

ln, n = size(X)
##

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

## OLS

ts = dt*(1:ln-1)
Ts = [ones(ln-1) log10.(ts)]
l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end

B[2,:] ./= 2

## OLS an

scatter(B[1,:],B[2,:], markerstrokewidth=0,markersize=2, label = "OLS")

flag = B[2,:] .<= 0
sX = X[:,flag]

plot(sX .- sX[1,:]',label="") # uwięzione?

## GLS prep

hs = 0.1:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
    errC[:,:,k] .= [theorCovEff(i,j,ln,K)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) for i in 1:ln-1, j in 1:ln-1]
end


## GLS fit

gB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i])
    j === nothing && (j = length(hs))
    gB[:,i] .= (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1*lmsd[:,i]
end

gB[2,:] ./= 2


## GLS an

scatter!(gB[1,:],gB[2,:], markerstrokewidth=0,markersize=2, 
    xlabel = "log10 D",
    ylabel = "H",
    label = "GLS",
)

var(B[1,:]), var(gB[1,:])
var(B[2,:]), var(gB[2,:])
cov(B[1,:],B[2,:]), cov(gB[1,:],gB[2,:])

dH = gB[2,:] .- B[2,:]
dD = gB[1,:] .- B[1,:]

scatter(gB[2,:],dH, markerstrokewidth=0,markersize=2, label = "")
scatter(gB[1,:],dH, markerstrokewidth=0,markersize=2, label = "")

scatter(gB[2,:],dD, markerstrokewidth=0,markersize=2, label = "")
scatter(gB[1,:],dD, markerstrokewidth=0,markersize=2, label = "")


## density
using KernelDensity

d = kde((B[1,:],B[2,:]))
heatmap(d.x,d.y,d.density)


## linear plot

k = 70
plot(log10.(ts),lmsd[:,k], label = "")
plot!(lt->2B[2,k]*lt + B[1,k],log10(ts[1]),log10(ts[end]),
    label = "OLS"
)
plot!(lt->2gB[2,k]*lt + gB[1,k],log10(ts[1]),log10(ts[end]),
    label = "GLS"
)