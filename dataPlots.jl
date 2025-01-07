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
ts = dt*(1:100)

## GLS prep

hs = 0.05:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    f = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) # no ln(10)
    errC[:,:,k] .= [theorCovEff(i,j,ln,f)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
end

@showprogress for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

## msd fit

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsd = log10.(msd)

lts = log10.(ts)
Ts = [ones(ln-1) log10.(ts)]
l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end
B[1,:] .-= log10(4)

## GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)

## msd traj plot

 k = 960 # 960 220 550! 880!

xt = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5]
yt = [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1]
## making plots

j = findfirst(hs .>= B[2,k]/2)

 plot(
    fontfamily = "Computer Modern",
    xlabel = "t [s]",
    ylabel = L"MSD [μm$^2$]",
    xticks = (log10.(xt), string.(xt)),
    yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}" ]),
    
 )
 scatter!(lts, lmsd[:,k] .-bias[:,j],
    yerrors = sqrt.(2errVar),
    color = palette(:default)[2],
    marker = :square,
    markersize = 3,
)
scatter!(lts, lmsd[:,k],
    marker = :circle,
    markercolor = :white,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
)
scatter!(lts[1:10], lmsd[1:10,k],
    marker = :circle,
    color = palette(:default)[1],
    markerstrokewidth = 0,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
)
plot!(lt->B[2,k]*lt+B[1,k]+log10(4),minimum(lts),maximum(lts),
    color = palette(:default)[1],
    linestyle = :dash,
    linewidth = 2,
)
plot!(lt->bB[2,k]*lt+bB[1,k]+log10(4),minimum(lts),maximum(lts),
    color = palette(:default)[2],
    linestyle = :dash,
    linewidth = 2,
)
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