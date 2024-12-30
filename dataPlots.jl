using Plots, MAT, ProgressMeter, LaTeXStrings
using Statistics, HypothesisTests, Distributions, LinearAlgebra

##

file = matopen("interphase_traj_l100.mat")
matread("interphase_traj_l100.mat")

dat = read(file,"traj")

X = dat[1:100,1:2:end]
Y = dat[1:100,2:2:end]

ln, n = size(X)
nn = size(dat,2)
dt = 0.0567


## msd fit

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

K = (s,t) -> 4*10^(B[1,k])/2 * (t^(B[2,k])+s^(B[2,k])-abs(s-t)^(B[2,k]))
errM =[theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
errVar = diag(errM)

bias = @. -log(10)*errVar/2

ts = dt*(1:ln-1)
lts = log10.(ts)
Ts = [ones(ln-1) log10.(ts)]
l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end
B[1,:] .-= log10(4)

## msd traj plot
 k = 200

scatter(lts, lmsd[:,k] .-bias,
    yerrors = sqrt.(2errVar),
    marker = :square,
    markersize = 2,
)
scatter!(lts, lmsd[:,k],
    marker = :circle,
    markercolor = :white,
    markersize = 2,
    #yerrors = sqrt.(2errVar),
)
plot!(lt->B[2,k]*lt+B[1,k]+log10(4),minimum(lts),maximum(lts))

#vline!([10])

## scatter plot

scatter(B[1,:],B[2,:],
    fontfamily = "Computer Modern",
    markerstrokewidth=0,
    markersize=2,
    alpha = 0.3,
    color = :black,
    xticks = (-5:-1, [L"10^{%$s}" for s in -5:-1]),
    xlim = (-5,-0.5),
    label = "OLS",
    xlabel = L"D\ [\mu m^2/s^{\alpha}]",
    ylabel = L"α\ [1]",
)

d = kde((B[1,:],B[2,:]))
contour!(d.x,d.y,d.density', linecolor=:black)