

using LaTeXStrings
using LinearAlgebra

using CairoMakie

H0, D0 = 0.35, 10^-3 
n = 10^4
ln = 100
dt = 0.0567
ts = dt*(1:ln)
##
 # 0.35, 10^-3    0.49, 10^-2  0.25, 10^-4  0.6, 10^-4    0.15, 10^-2
 H0, D0 = 0.35, 10^-3 
 n = 10^4
 ln = 100
 dt = 0.0567
 ts = dt*(1:ln)
 
 ##
 K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
 S = [K(s,t) for s in ts, t in ts]
 A = cholesky(Symmetric(S)).U
 ξ = randn(length(ts), n)
 X = A'*ξ
 ξ = randn(length(ts), n)
 Y = A'*ξ

## GLS prep

hs = 0.1:0.01:0.8
errC = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for k in eachindex(hs)
    D0, H0 = 1, hs[k]
    K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
    errC[:,:,k] .= [theorCovEff(i,j,ln,K)/(ts[i]^(2hs[k])*ts[j]^(2hs[k])) for i in 1:ln-1, j in 1:ln-1]
end

for k in eachindex(hs)
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
 
B2 = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B2[:,i] .= (Ts[l+1:end,:]'*Ts[l+1:end,:])^-1*Ts[l+1:end,:]'*lmsd[l+1:end,i]
end
B2[1,:] .-= log10(4)


 # GLS fit
 
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
 
## znajdź odchylenia
using JLD2

#@save "plotSimpleComp.jld2" msd B bB
@load "plotSimpleComp.jld2" msd B bB # na PC w dom

ols = abs.(B[2,:] .- 0.7)
#ols2 = abs.(B2[2,:] .- 0.7)
gls = abs.(bB[2,:] .- 0.7)

js0 = sortperm(ols)
js = sortperm(abs.(ols .- gls))
js2 = sortperm(abs.(ols) .- abs.(gls))
#js3 = sortperm(abs.(ols) .- abs.(ols2))
#js4 = sortperm(ols2)
##
 # lap data long time

#j = 2451 # # lap data short time gls impr

#j = 779

## Makie ver

with_theme(theme_latexfonts()) do
fig = Figure(size=1 .* (1200,400),
    fontsize = 22,
)
fs = 22
xt = ([0.1,0.2,0.5,1,2,5], string.([0.1,0.2,0.5,1,2,5]) )
yt = ([10^-3,2*10^-3,5*10^-3, 10^-2], [L"10^{-3}",L"2\!\cdot\! 10^{-3}",L"5\!\cdot\! 10^{-3}", L"10^{-2}"])

j1, j2 = 9193, 6414 #2413 # 6414 553
mm, M = 0.9min(msd[1,j1],msd[1,j2]), 1.1max(1.1maximum(msd[:,j1]), 1.1maximum(msd[:,j2]))
j = j1 # 7530 #7213 

ax = Axis(fig[1,1],
    xlabel = L"$t$ [s]",
    ylabel = L"MSD [μm$^2$]",
    ylabelpadding = 5,
    #title = "Trajectory with misleading short time TA-MSD",
    #title = "Trajectory with misleading long time TA-MSD",
    xscale = log10,
    yscale = log10,
    xticks = xt,
    yticks = yt,
    limits = (nothing,nothing, mm, M)
)
ax.xticklabelsize = fs
ax.yticklabelsize = fs
ax.xlabelsize = fs
ax.ylabelsize = fs
# Makie.lines!(ax,[ts[end÷10], ts[end÷10]],[0.9msd[1,j], 1.1maximum(msd[:,j])],
#     linestyle = :dash,
#     label = "10% of the data",
#     color = :black,
#     alpha = 0.5,
# )
Makie.scatter!(ax,ts[1:99],msd[:,j],
    color = :white,#palette(:default)[3],
    marker = :circle,
    strokewidth = 1,
    #markersize = 3,
    label = "measured TA-MSD",
)
Makie.scatter!(ax,ts[1:10],msd[1:10,j],
    color = :grey,#palette(:default)[3],
    marker = :circle,
    strokewidth = 1,
    #markersize = 3,
    label = "OLS range",
)
lines!(ax,ts, 4D0*ts .^(2H0),
    label = "exact MSD",
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
#axislegend(position = :lt)
j = j2 #7530
ax = Axis(fig[1,2],
    xlabel = L"$t$ [s]",
    #ylabel = L"MSD [μm$^2$]",
    ylabelpadding = 0,
    #title = "Trajectory with misleading short time TA-MSD",
    #title = "Trajectory with misleading long time TA-MSD",
    xscale = log10,
    yscale = log10,
    xticks = xt,
    yticks = (yt[1], ["", "", "", ""]),
    limits = (nothing,nothing, mm, M)
)
ax.xticklabelsize = fs
ax.yticklabelsize = fs
ax.xlabelsize = fs
ax.ylabelsize = fs
# Makie.lines!(ax,[ts[end÷10], ts[end÷10]],[0.9msd[1,j], 1.1maximum(msd[:,j])],
#     linestyle = :dash,
#     label = "10% of the data",
#     color = :black,
#     alpha = 0.5,
# )
Makie.scatter!(ax,ts[1:99],msd[:,j],
    color = :white,#palette(:default)[3],
    marker = :circle,
    strokewidth = 1,
    #markersize = 3,
    label = "measured TA-MSD",
)
Makie.scatter!(ax,ts[1:10],msd[1:10,j],
    color = :grey,#palette(:default)[3],
    marker = :circle,
    strokewidth = 1,
    #markersize = 3,
    label = "OLS range",
)
lines!(ax,ts, 4D0*ts .^(2H0),
    label = "exact MSD",
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
#axislegend(position = :lt)
colgap!(fig.layout,40)

save("simpleComp.pdf",fig)
fig
end
