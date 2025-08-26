using Plots, MAT, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")
##

file = matopen("interphase_traj_l100.mat")
#matread("interphase_traj_l100.mat")

dat = read(file,"traj")

X = dat[1:100,1:2:end]
Y = dat[1:100,2:2:end]

ln, n = size(X)
nn = size(dat,2)
dt = 0.0567
ts = dt*(1:ln)


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

lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]
l = 10

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
end
B[1,:] .-= log10(4)


## GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)
#cB = Matrix{Float64}(undef, 2, n)

for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
    #cR = (Ts[l+1:end,:]'*errC[l+1:end,l+1:end,j]^-1*Ts[l+1:end,:])^-1*Ts[l+1:end,:]'*errC[l+1:end,l+1:end,j]^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .- bias[:,j])
    #cB[:,i] .= cR*(lmsd[l+1:end,i] .- bias[l+1:end,j])
end

gB[1,:] .-= log10(4)
bB[1,:] .-= log10(4)


## err plot log

xt = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5]
yt = [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1,1]
k = 158 #150 
j = findfirst(hs .>= B[2,k]/2)
errVar = diag(errC[:,:,j])


H0, D0 = bB[2,k]/2, 10^bB[1,k] # sim
n = 10^4
ln = 100
dt = 0.0567
ts = dt*(1:ln)

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ


msdS = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msdS[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
end

lmsdS = log10.(msdS)



p = plot(
    fontfamily = "Computer Modern",
    xlabel = "t [s]",
    ylabel = L"MSD [μm$^2$]",
    #xticks = (log10.(xt), string.(xt)),
    #yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
    label = "",
    ylim = (0,0.05),
    xlim = (0,5.5),

 )
 plot!(ts[1:end-1],msdS[:,1:15],
    linecolor=palette(:default)[1],
    #marker = :circle,
    #markercolor = palette(:default)[1],
    #markersize = 3,
    #markerstrokewidth=0,
    linewidth = 2,
    alpha=0.2,
    label = "",

)
scatter!(ts[1:end-1], msd[:,k],
    marker = :circle,
    markercolor = :white,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
    label = "experimental TA-MSD",
)

plot!(t->4*10^bB[1,k]*t^bB[2,k],minimum(ts),maximum(ts),
    color = :black,#palette(:default)[3],
    linestyle = :solid,
    linewidth = 2.5,
    label = "MSD fit"
)

plot!([],[],
    linecolor=palette(:default)[1],
    linewidth = 2,
    alpha=0.2,
    label = "simulated TA-MSDs"
)

display(p)
savefig("err.pdf")

## err plot non-log


p = plot(
    fontfamily = "Computer Modern",
    xlabel = "t [s]",
    ylabel = L"MSD [μm$^2$]",
    xticks = (log10.(xt), string.(xt)),
    yticks = (log10.(yt), [L"10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}", L"5\!\cdot\! 10^{-2}", L"10^{-1}",L"10^0" ]),
    label = "",

 )
 plot!(lts,lmsdS[:,1:15],
    linecolor=palette(:default)[1],
    #marker = :circle,
    #markercolor = palette(:default)[1],
    #markersize = 3,
    #markerstrokewidth=0,
    linewidth = 2,
    alpha=0.2,
    label = "",

)
scatter!(lts, lmsd[:,k],
    marker = :circle,
    markercolor = :white,
    markersize = 3,
    #yerrors = sqrt.(2errVar),
    label = "experimental TA-MSD",
)

plot!(lt->bB[2,k]*lt+bB[1,k]+log10(4),minimum(lts),maximum(lts),
    color = :black,#palette(:default)[3],
    linestyle = :solid,
    linewidth = 2.5,
    label = "MSD fit"
)

plot!([],[],
    linecolor=palette(:default)[1],
    linewidth = 2,
    alpha=0.2,
    label = "simulated TA-MSDs"
)

display(p)
savefig("err.pdf")


## err heteroscedasticity, cor

plot(lts,lmsd[:,k])

ff = @.  bB[2,k]*lts+bB[1,k]+log10(4) 

plot!(lts,ff)

empErr = zeros(99)
empCov = zeros(99,50)
c = 0
for j in 1:size(lmsd,2)
    if bB[2,j] > 0.2
        err = @. lmsd[:,j] - (bB[2,j]*lts+bB[1,j]+log10(4))
        for l in 1:50
            empCov[:,l] .+= err[l] .* err
        end
        empErr .+= err .^2
        c += 1
    end
end

empErr ./= c
empCov ./= c
empCor = empCov ./ (sqrt.(empErr .* empErr[1:50]'))



scatter(lts[1:end-1],empErr,
    marker = :square,
    markercolor = :white,
    markersize = 3,
    yscale= :log10,
    #xscale= :log10,
    fontfamily = "Computer Modern",
    xlabel = "t [s]",
    ylabel = "variance",
    label = "sample variance of log TA-MD errors",
    xticks = (log10.(xt), string.(xt)),
    legend = :topleft,
)
#savefig("errVar.pdf")

p = Plots.plot([],[],
    fontfamily = "Computer Modern",
    xlabel = L"error index $j$",
    ylabel = "correlation",
    label = "",
    yticks = [-0.4,-0.2,0,0.2,0.4,0.6,0.8,1],
    #xlim= (0,100),
    #legend = :centerright,
)
#hline!(p,[0],linestyle = :dash,label = "",color=:black,linewidth=0.5)
lst = [3,5,10,15,20,50]
pal = :rainbow_bgyr_35_85_c72_n256 #:Set1_3 # :RdYlGn_4
cls = palette(pal,length(lst)) #palette([:red,:blue],length(lst))
mrks = [:circle,:square,:utriangle,:diamond,:hexagon,:star]
for (k,l) in enumerate(lst)
    Plots.plot!(1:99,empCor[:,l],
        label = L"corr($e_j$,$\.$ $e_k$), $k = %$l$",
        linewidth = 0.5,
        #linealpha = 0.5,
        color = cls[k],
        marker = mrks[k],
        markerstrokewidth=0,
        markersize=3,
        xlim = (0,100),
        framestyle = :zerolines,
    )
end

display(p)

#savefig("errCorr.pdf")
