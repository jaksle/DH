using Plots, LinearAlgebra, ProgressMeter, LaTeXStrings


##

H0 = 0.25
D0  = 1.
n = 10^5
ln = 100
ts = 1:ln

K(s,t) = D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ

msd = Matrix{Float64}(undef,ln-1,n)

for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end

lmsd = log10.(msd)

##
lns = 2:50
Ts = [ones(ln) log10.(ts)]
thcM = @showprogress [theorCovEff(i,j,ln,K)/(K(ts[i],ts[i])*K(ts[j],ts[j])) for i in 1:lns[end], j in 1:lns[end]]



vlD = Vector{Float64}(undef,length(lns))
v2H = similar(vlD)
vC = similar(vlD)
gvlD = Vector{Float64}(undef,length(lns))
gv2H = similar(vlD)
gvC = similar(vlD)

@showprogress for (k,m) in enumerate(lns)
    R = (Ts[1:m,:]'*Ts[1:m,:])^-1*Ts[1:m,:]'
    B = R * lmsd[1:m,:]
    vlD[k] = var(B[1,:])
    v2H[k] = var(B[2,:])
    vC[k] = cov(B[1,:],B[2,:])

    gR = (Ts[1:m,:]'*thcM[1:m,1:m]^-1*Ts[1:m,:])^-1*Ts[1:m,:]'*thcM[1:m,1:m]^-1
    gB = gR * lmsd[1:m,:]
    gvlD[k] = var(gB[1,:])
    gv2H[k] = var(gB[2,:])
    gvC[k] = cov(gB[1,:],gB[2,:])
end

## plots

p = plot(layout = (2,1))
scatter!(p[1],lns, vlD,
    fontfamily = "Computer Modern",
    markerstrokewidth = 0,
    markersize = 3,
    color =palette(:default)[1],
    title = L"true $α = 0.5$", 
    xlim = (lns[1],lns[end]),
    xticks = vcat(2,5,vec(10:10:50)),
    label = L"OLS estimation of $D$",
    ylabel = L"${}_\mathrm{tw}\mathrm{var}(\log{}_{10}\hat D)$",
    linewidth = 2,
)
plot!(p[1],lns,vlD,
    color =palette(:default)[1],
    alpha = 0.2,
    label = "",

)
scatter!(p[1],lns, gvlD,
    fontfamily = "Computer Modern",
    color =palette(:default)[1],
    markerstrokewidth = 0,
    markersize = 3,
    marker = :utriangle,
    linestyle = :dot,
    label = L"GLS estimation of $D$",
    #ylabel = L"${}_\mathrm{tw}\mathrm{var}(\log{}_\mathrm{tw}\hat D)$",
    linewidth = 2,
)
plot!(p[1],lns,gvlD,
    color =palette(:default)[1],
    alpha = 0.2,
    label = "",

)
scatter!(p[2],lns, vlD,
    fontfamily = "Computer Modern",
    color =palette(:default)[1],
    markerstrokewidth = 0,
    markersize = 3,
    xlim = (lns[1],lns[end]),
    ylim = (0.0044,0.0046),
    xticks = vcat(2,5,vec(10:10:50)),
    label = "",
    ylabel = L"${}_\mathrm{tw}\mathrm{var}(\log{}_{10}\hat D)$",
    xlabel = "length of TA-MSD used",
    linewidth = 2,
)
plot!(p[2],lns,vlD,
    color =palette(:default)[1],
    alpha = 0.2,
    label = "",

)

scatter!(p[2],lns, gvlD,
    fontfamily = "Computer Modern",
    color =palette(:default)[1],
    markerstrokewidth = 0,
    markersize = 3,
    marker = :utriangle,
    label = "",
    #ylabel = L"${}_\mathrm{tw}\mathrm{var}(\log{}_\mathrm{tw}\hat D)$",
    linewidth = 2,
)
plot!(p[2],lns,gvlD,
    color =palette(:default)[1],
    alpha = 0.2,
    label = "",

)

#savefig("inputLenD.pdf")
##

#p = plot(layout = (2,1))
scatter(lns, v2H,
    fontfamily = "Computer Modern",
    markerstrokewidth = 0,
    markersize = 3,
    color =palette(:default)[2],
    title = L"true $H = 0.25$", 
    xlim = (lns[1],lns[end]),
    xticks = vcat(2,5,vec(10:10:50)),
    label = L"OLS estimation of $α$",
    ylabel = L"${}_\mathrm{tw}\mathrm{var}(\hat α)$",
    ylim = (0.01,0.045),
    yticks = 0.01:0.005:0.045,
    linewidth = 2,
)
plot!(lns,v2H,
    color =palette(:default)[2],
    alpha = 0.2,
    label = "",

)
scatter!(lns, gv2H,
    fontfamily = "Computer Modern",
    color =palette(:default)[2],
    markerstrokewidth = 0,
    markersize = 3,
    marker = :utriangle,
    linestyle = :dot,
    label = L"GLS estimation of $α$",
    #ylabel = L"${}_\mathrm{tw}\mathrm{var}(2H)$",
    xlabel = "length of TA-MSD used",
)
plot!(lns,gv2H,
    color =palette(:default)[2],
    alpha = 0.2,
    label = "",

)


#savefig("inputLenH.pdf")