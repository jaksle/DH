

using Plots, ProgressMeter, LaTeXStrings

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
@load "plotSimpleComp.jld2" msd B bB

ols = abs.(B[2,:] .- 0.7)
gls = abs.(bB[2,:] .- 0.7)

findmax(ols .- gls)
j =  findmax(ols .- gls)[2]

j = 6 # long time TA-MSD

vline([ts[end÷10]],linestyle= :dash,
   label = "",
   color = :black,
   alpha = 0.5,
)
plot!(ts[1:99],msd[:,j],
    fontfamily = "Computer Modern",
    xlabel = "t [s]",
    ylabel = L"MSD [μm$^2$]",
    color = :white,#palette(:default)[3],
    marker = :circle,
    markersize = 3,
    label = "measured TA-MSD",
    title = "Trajectory with misleading long time TA-MSD",
    #title = "Trajectory with misleading short time TA-MSD",
    xlim = (ts[1],ts[100]+0.1),
    xscale = :log10,
    yscale = :log10,
    xticks = (vcat(vec(0.1:0.1:0.5),vec(1:5)), vcat(string.(vec(0.1:0.1:0.5)),string.(vec(1:5))) ),
    yticks = (vcat(vec(10^-3 * (1:5)), 10^-2), [L"10^{-3}","","","","", L"10^{-2}"]),
    ylim = (msd[1,j],0.015),
    markerstrokewidth = 0.5,
 )
 plot!(t->4D0*t^(2H0),ts[1],ts[99],
    label = "exact MSD",
    color = :black,
    linestyle = :dash,
    linewidth = 2,
 )

plot!([],[],linestyle= :dash,
    label = "10% of the data",
    color = :black,
    alpha = 0.5,
 )



 plot!(t->4*10^B[1,j]*t^(B[2,j]),0,ts[99],
 label = "OLS estimate",
 linewidth = 2,
)
plot!(t->4*10^bB[1,j]*t^(bB[2,j]),0,ts[99],
 label = "GLS estimate",
 linewidth = 2,
)
 savefig("simpleComp1.pdf")