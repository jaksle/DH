using Plots, MAT, ProgressMeter
using Statistics, HypothesisTests, Distributions

##

file = matopen("interphase_traj_l100.mat")
matread("interphase_traj_l100.mat")

dat = read(file,"traj")

#X = read(file,"trajx")
#Y = read(file,"trajy")
X = dat[1:100,1:2:end]
Y = dat[1:100,2:2:end]

ln, n = size(X)
nn = size(dat,2)
dt = 0.0567

## kurtosis

kurt = [kurtosis(diff(dat[1:100,k],dims=1)) for k in 1:nn]
kurtTh = [kurtosis(randn()^2*randn(99)) for k in 1:nn]

filterK = B[2,:] .> 0.3 
fK = vec([filterK filterK]')
histogram(kurt,normalize=true,linewidth=0)
histogram(kurtTh,normalize=true,linewidth=0)
## K-S

ks = zeros(nn)
for i in 1:nn
    v = diff(dat[1:100,i],dims=1)
    w = ExactOneSampleKSTest(v, Normal(mean(v),std(v)))
    ks[i] = w.δ
end

ksTh = zeros(nn)
for i in 1:nn
    v = randn(99)
    w = ExactOneSampleKSTest(v, Normal(mean(v),std(v)))
    ksTh[i] = w.δ
end

histogram(ks,normalize=true,linewidth=0)
histogram!(ksTh,normalize=true,linewidth=0)

## displ plot

histogram(dat[10,:] .- dat[1,:],normed=true)

## normed diffs

ddat = diff(dat[1:100,:],dims = 1)
stds = std(ddat,dims=1)
ndat = ddat ./ stds

histogram(ndat[:,1:4000][:], normed=true)

plot!(x->pdf(Normal(),x),-3,3,
    linewidth = 2
)
## msd main

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)


## msd full

lns = [findlast(!isnan,dat[:,k]) for k in 1:nn]

ts = dt*(1:2000-1)
Ts = [ones(2000-1) log10.(ts)]
lim = 20

fB = Matrix{Float64}(undef, 2, nn)
for i in 1:n
    l = lns[i]
    msd = estMSD(dat[1:l,i],lim)
    fB[:,i] .= (Ts[1:lim,:]'*Ts[1:lim,:])^-1*Ts[1:lim,:]' * log10.(msd)
end
fB[1,:] .-= log10(2)

## OLS

ts = dt*(1:ln-1)
Ts = [ones(ln-1) log10.(ts)]
l = 20

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end
B[1,:] .-= log10(4)

## OLS an

scatter(B[1,:],B[2,:],
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.5,
    color = :black,
    label = "OLS"
)


## GLS prep

hs = 0.05:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    f = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) for i in 1:ln-1, j in 1:ln-1]
end


## GLS fit

gB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gB[:,i] .= (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1*lmsd[:,i]
end
gB[1,:] .-= log10(4)


## GLS an

scatter(B[1,:],B[2,:],
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.5,
    color = :blue,
    label = "OLS",
    xlabel = "log10 D",
    ylabel = "α",
)
scatter!(gB[1,:],gB[2,:],
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.5,
    color = :red,
    label = "GLS"
)

var(B[1,:]), var(gB[1,:])
var(B[2,:]), var(gB[2,:])
cov(B[1,:],B[2,:]), cov(gB[1,:],gB[2,:])

da = gB[2,:] .- B[2,:]
dD = gB[1,:] .- B[1,:]

scatter(gB[2,:],da, markerstrokewidth=0,markersize=2, label = "")
scatter(gB[1,:],da, markerstrokewidth=0,markersize=2, label = "")

scatter(gB[2,:],dD, markerstrokewidth=0,markersize=2, label = "")
scatter(gB[1,:],dD, markerstrokewidth=0,markersize=2, label = "")

## piecewise analysis of alpha corrections

filter = B[2,:] .> 1
filter2 = gB[2,:] .> 1
scatter(B[1,filter],B[2,filter],
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.5,
    color = :blue,
    label = "OLS",
    xlabel = "log10 D",
    ylabel = "α",
)
scatter!(gB[1,filter],gB[2,filter],
    markerstrokewidth=0,
    markersize=1,
    alpha = 0.5,
    color = :red,
    label = "GLS"
)

sum(filter)
sum(gB[2,:] .> 1)
sum(gB[2,filter] .> 1)

scatter(gB[2,filter2])
mean(gB[2,filter2])
mean(B[2,filter])
## density
using KernelDensity

d = kde((B[1,:],B[2,:]))
heatmap(d.x,d.y,d.density')


## linear plot

k = 70
plot(log10.(ts),lmsd[:,k], label = "")
plot!(lt->2B[2,k]*lt + B[1,k],log10(ts[1]),log10(ts[end]),
    label = "OLS"
)
plot!(lt->2gB[2,k]*lt + gB[1,k],log10(ts[1]),log10(ts[end]),
    label = "GLS"
)