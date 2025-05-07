
using Plots, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")


n = 10^5
ln = 10
dt =  0.0567
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]


##

using DataFrames

simH = 0.15:0.05:0.7

data2 = DataFrame(
    α = 2simH,
    OLS_bias = fill((0.,0.),12),
    GLS_bias = fill((0.,0.),12),
    GLS_bgain = fill((0.,0.),12),
    OLS_var = fill((0.,0.),12),
    GLS_var = fill((0.,0.),12),
    GLS_vgain = fill((0.,0.),12),
)
data5 = DataFrame(
    α = 2simH,
    OLS_bias = fill((0.,0.),12),
    GLS_bias = fill((0.,0.),12),
    GLS_bgain = fill((0.,0.),12),
    OLS_var = fill((0.,0.),12),
    GLS_var = fill((0.,0.),12),
    GLS_vgain = fill((0.,0.),12),
)

## GLS prep exact

hs = 0.05:0.01:0.95
errCEx = Array{Float64}(undef,ln-1,ln-1,length(hs))
biasEx = Array{Float64}(undef,ln-1,length(hs))
msdTemp = Matrix{Float64}(undef,ln-1,n)

@showprogress for k in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[k])+s^(2hs[k])-abs(s-t)^(2hs[k])) 
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    x = A'*ξ
    ξ = randn(length(ts), n)
    y = A'*ξ

    for i in 1:n
        msdTemp[:,i] .= estMSD(x[:,i],ln-1) .+ estMSD(y[:,i],ln-1)  # 2D
    end
    ltemp = log10.(msdTemp)
    errCEx[:,:,k] .= cov(ltemp')

    biasEx[:,k] .=  mean(ltemp,dims = 2) .- 2hs[k]*lts .- log10(4)  # 2D bias
end


##
n = 10^5
l = 5
data = data5 
ols = Array{Float64}(undef,2,n,length(simH))
gls = similar(ols)

@showprogress for k in eachindex(simH)
    H0, D0 = simH[k], 1.
    #n = 10^6
    ln = 10
    dt =  0.0567
    ts = dt*(1:ln)
    lts = log10.(ts[1:ln-1])
    Ts = [ones(ln-1) lts]
    
    K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
    S = [K(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    X = A'*ξ
    ξ = randn(length(ts), n)
    Y = A'*ξ

    msd = Matrix{Float64}(undef,ln-1,n)
    for i in 1:n
        msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1) 
    end

    lmsd = log10.(msd)
    B = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        B[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd[1:l,i]
    end

    B[1,:] .-= log10(4)

    eB = Matrix{Float64}(undef, 2, n)

    for i in 1:n
        j = findfirst(hs .>= B[2,i]/2) # H not α
        j === nothing && (j = length(hs))
        #jmin, jmax = findfirst(hs .>= 0.2), findfirst(hs .>= 0.5)
        #j = max(j,jmin); j = min(j,jmax)

        gR = (Ts'*errCEx[:,:,j]^-1*Ts)^-1*Ts'*errCEx[:,:,j]^-1
        eB[:,i] .= gR*(lmsd[:,i] .- biasEx[:,j])
    end
    eB[1,:] .-= log10(4)
    ols[:,:,k] .= B
    gls[:,:,k] .= eB
    data.OLS_bias[k] = mean(B[1,:]),  mean(B[2,:]) - 2H0
    data.OLS_var[k] = var(B[1,:]),  var(B[2,:])
    data.GLS_bias[k] = mean(eB[1,:]),  mean(eB[2,:]) - 2H0
    data.GLS_var[k] = var(eB[1,:]),  var(eB[2,:])

    data.GLS_bgain[k] = 100 .*(1 .- abs.(data.GLS_bias[k] ./data.OLS_bias[k]))
    data.GLS_vgain[k] = 100 .* (1 .- data.GLS_var[k] ./ data.OLS_var[k])
end


## round

r(x) = round.(x, sigdigits=2)

r.(data)

for row in  1:12
    R = Float64[]
    for k in 1:7
        x = data[row,k]
        if x isa Tuple
            push!(R,x[1])
            push!(R,x[2])
        else
            push!(R,x)
        end
    end

    r.(R)' |> display
end



## violin plots

using CairoMakie


fig = Figure()
ax = Axis(fig[1,1],
    ylabel = L"estimated $\alpha$",
    xlabel = L"sample $\alpha$",
    xticks = (1:1:12, string.(2simH[1:1:12])),
    limits= (0,13,-1,2.5),
)
for k in 1:1:12
    Makie.violin!(ax,k*ones(n),ols[2,:,k],side=:left,
        color = :dodgerblue2,
        show_median = true
    )
    # CairoMakie.lines!(ax, [k-1/2,k], fill(mean(ols[2,:,k]),2),
    #     #marker='⨉',
    #     color=:blue,
    #     #markersize = 10,
    # )
    Makie.violin!(ax,k*ones(n),gls[2,:,k],side=:right,
        color = :tomato,
        show_median = true
    )
    # CairoMakie.lines!(ax, [k,k+1/2], fill(mean(gls[2,:,k]),2),
    #     #marker='⨉',
    #     color=:red,
    #     #markersize = 10,
    # )
end
CairoMakie.scatter!(ax, 1:1:12, 2simH[1:1:12],
    marker='⨉',
    color=:black,
    markersize = 10,
)

fig
save("violinA.png",fig)




fig = Figure()
ax = Axis(fig[1,1],
    ylabel = L"estimated $D$",
    xlabel = L"sample $\alpha$",
    xticks = (1:1:12, string.(2simH[1:1:12])),
    limits= (0,13,-2,2),
)
for k in 1:1:12
    Makie.violin!(ax,k*ones(n),ols[1,:,k],side=:left,
        color = :dodgerblue2,
        show_median = true
    )
    # CairoMakie.lines!(ax, [k-1/2,k], fill(mean(ols[2,:,k]),2),
    #     #marker='⨉',
    #     color=:blue,
    #     #markersize = 10,
    # )
    Makie.violin!(ax,k*ones(n),gls[1,:,k],side=:right,
        color = :tomato,
        show_median = true
    )
    # CairoMakie.lines!(ax, [k,k+1/2], fill(mean(gls[2,:,k]),2),
    #     #marker='⨉',
    #     color=:red,
    #     #markersize = 10,
    # )
end
# CairoMakie.scatter!(ax, 1:1:12, 2simH[1:1:12],
#     marker='⨉',
#     color=:black,
#     markersize = 10,
# )

fig
save("violinD.png",fig)



## box plot

fig = Figure()
ax = Axis(fig[1,1],
    ylabel = L"estimated $\alpha$",
    xlabel = L"sample $\alpha$",
    limits= (0,13,-1,3),
)
for k in 1:2:12
    Makie.boxplot!(ax,k*ones(n),ols[2,:,k],
        color = :dodgerblue2,
    )
    Makie.boxplot!(ax,k*ones(n) .+ 1,gls[2,:,k],
    color = :tomato
    )
end


fig