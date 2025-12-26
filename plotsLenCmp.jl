using Plots, LinearAlgebra, ProgressMeter, LaTeXStrings

include("funs.jl")
##

H0 = 0.25
D0  = 1.
n = 10^5
ln = 100
dt = 0.0567 # 1 default
ts = dt*(1:ln)

K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
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
bias =  -1/log(10) .* diag(thcM) ./2

#plot(log10.(2:50),mean(lmsd,dims=2)[2:50])
#plot!(log10.(2:50),log10.(K.(ts,ts))[2:50] .+ bias[2:50])

vlD = Vector{Float64}(undef,length(lns))
v2H = similar(vlD)
vC = similar(vlD)
gvlD = Vector{Float64}(undef,length(lns))
gv2H = similar(vlD)
gvC = similar(vlD)
mlD = similar(vlD)
m2H = similar(vlD)
gmlD = similar(vlD)
gm2H = similar(vlD)

@showprogress for (k,m) in enumerate(lns)
    R = (Ts[1:m,:]'*Ts[1:m,:])^-1*Ts[1:m,:]'
    B = R * lmsd[1:m,:]
    vlD[k] = var(B[1,:])
    v2H[k] = var(B[2,:])
    vC[k] = cov(B[1,:],B[2,:])

    gR = (Ts[1:m,:]'*thcM[1:m,1:m]^-1*Ts[1:m,:])^-1*Ts[1:m,:]'*thcM[1:m,1:m]^-1
    gB = gR * (lmsd[1:m,:] .- bias[1:m])
    gvlD[k] = var(gB[1,:])
    gv2H[k] = var(gB[2,:])
    gvC[k] = cov(gB[1,:],gB[2,:])

    mlD[k] = mean(B[1,:])
    gmlD[k] = mean(gB[1,:])
    m2H[k] = mean(B[2,:])
    gm2H[k] = mean(gB[2,:])
end

#plot(0.5 .- bias1)
#plot!(0.5 .- bias2)


##


## 
using CairoMakie

with_theme(theme_latexfonts()) do
    fig = Figure(size = (1200,400),
        fontsize = 22,
    )
    darkRed = colorant"#cc3434"
    ax = Axis(fig[1, 1],
        #title = L"true $\alpha = 0.5$",
        titlesize = 22, 
        ylabel = L"var($\hat{\alpha}$)",
        xlabelsize = 22,
        #xlabel = "length of TA-MSD used",
        ylabelsize = 22,
        xticks = vcat(2,5,vec(10:10:50)),
        yticks = [0,0.02,.05]
    )
    CairoMakie.xlims!(ax,1,51)
    CairoMakie.ylims!(ax,0.0,0.05)
    CairoMakie.scatter!(ax,lns,v2H,
        label = L"OLS estimation of $α$",
        color = :dodgerblue2,
    )
    CairoMakie.scatter!(ax,lns,gv2H,
        label = L"GLS estimation of $α$",
        color = :tomato,
        marker = :utriangle,
    )

    ax2 = Axis(fig[2, 1],
        #title = L"true $\alpha = 0.5$",
        titlesize = 22, 
        ylabel = L"$\langle\hat{\alpha}\rangle - \alpha$",
        xlabelsize = 22,
        xlabel = L"length $m$ of TA-MSD used ",
        ylabelsize = 22,
        xticks = vcat(2,5,vec(10:10:50)),
        yticks = -0.06:0.02:0.00,
        #yticks = 0.00:0.005:0.045,
    )
    CairoMakie.xlims!(ax2,1,51)
    #CairoMakie.ylims!(ax2,0.0,0.045)
    CairoMakie.scatter!(ax2,lns,m2H .- 0.5,
        label = L"OLS estimation of $α$",
        color = :dodgerblue2,
    )
    CairoMakie.scatter!(ax2,lns,gm2H .-0.5,
        label = L"GLS estimation of $α$",
        color = :tomato,
        marker = :utriangle,
    )


    ax3 = Axis(fig[1, 2],
        #title = L"true $\alpha = 0.5$",
        titlesize = 22, 
        ylabel = L"var($\log_{10} \hat{D}$)",
        xlabelsize = 22,
        #xlabel = "length of TA-MSD used",
        ylabelsize = 22,
        xticks = vcat(2,5,vec(10:10:50)),
        yticks = 0.01:0.01:0.03,
    )
    CairoMakie.xlims!(ax3,1,51)
    CairoMakie.ylims!(ax3,0.01,0.03)
    CairoMakie.scatter!(ax3,lns,vlD,
        label = L"OLS estimation of $α$",
        color = :dodgerblue2,
    )
    CairoMakie.scatter!(ax3,lns,gvlD,
        label = L"GLS estimation of $α$",
        color = :tomato,
        marker = :utriangle,
    )

    ax4 = Axis(fig[2, 2],
        #title = L"true $\alpha = 0.5$",
        titlesize = 22, 
        ylabel = L"$\langle\log_{10}\,\hat{D}\rangle - \log_{10}\,D$",
        xlabelsize = 22,
        xlabel = L"length $m$ of TA-MSD used ",
        ylabelsize = 22,
        xticks = vcat(2,5,vec(10:10:50)),
        yticks = [-.04,-0.02,0]
    )
    CairoMakie.xlims!(ax4,1,51)
    #CairoMakie.ylims!(ax3,0.0,0.03)
    s1 = CairoMakie.scatter!(ax4,lns,mlD,
        label = "OLS",
        color = :dodgerblue2,
    )
    s2 = CairoMakie.scatter!(ax4,lns,gmlD,
        label = "GLS",
        color = :tomato,
        marker = :utriangle,
    )
    axislegend(ax4,position=:rc)

    save("lenComp.pdf",fig)
    fig
end