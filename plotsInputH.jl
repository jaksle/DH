
#using Plots
using  LinearAlgebra, ProgressMeter, LaTeXStrings
include("funs.jl")

##

hs = 0.2:0.01:0.6
lenH = length(hs)
b = 10
ln = 100
m = 99
n = 10^4
ts = 1:ln


##

M = Array{Float64}(undef,lenH,m,m)

@showprogress for (a,h) in enumerate(hs)
    f = (s,t) -> 1/2*(t^(2h)+s^(2h)-abs(s-t)^(2h))
    M[a,:,:] .= [1/(log(b))^2*theorCovEff(i,j,ln,f)/(f(ts[i],ts[i])*f(ts[j],ts[j])) for i in 1:m, j in 1:m]
end


Ts = [ones(m) log10.(ts[1:m])]
gR = [(Ts'*M[a,:,:]^-1*Ts)^-1*Ts'*M[a,:,:]^-1 for a in 1:lenH]


#R = (Ts'*Ts)^-1*Ts'


Res1 = Matrix{Float64}(undef,lenH,lenH)
Res2 = similar(Res1)
Res3 = similar(Res1)


n = 10^5

@showprogress for (a,h) in enumerate(hs)
    f = (s,t) -> 1/2*(t^(2h)+s^(2h)-abs(s-t)^(2h))
    S = [f(s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(length(ts), n)
    X = A'*ξ
    msd = Matrix{Float64}(undef,m,n)
    for i in 1:n
        msd[:,i] .= estMSD(X[:,i],m)
    end
    lmsd = log10.(msd)
    for b in 1:lenH
        B = gR[b] * lmsd
        Res1[a,b] = var(B[1,:])
        Res2[a,b] = var(B[2,:])
        Res3[a,b] = cov(B[1,:],B[2,:])
    end
end

Xg = [hs[i] for i in 1:lenH, j in 1:lenH]
Yg = [hs[j] for i in 1:lenH, j in 1:lenH]

## check for one H

h = 0.3

f = (s,t) -> 1/2*(t^(2h)+s^(2h)-abs(s-t)^(2h))
S = [f(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
msd = Matrix{Float64}(undef,m,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],m)
end
lmsd = log10.(msd)

res1 = zeros(lenH)
res2 = zeros(lenH)
res3 = zeros(lenH)
for b in 1:lenH
    B = gR[b] * lmsd
    res1[b] = var(B[1,:])
    res2[b] = var(B[2,:])
    res3[b] = cov(B[1,:],B[2,:])
end


## plot optimum

plot(2hs,res1,
    fontfamily = "Computer Modern",
    ribbon=  (zeros(lenH),-res1),
    ylim = (minimum(res1),maximum(res1)),
    xlabel = L"input $α$",
    title = L"true $α=0.6$",
    ylabel = L"{}_\mathrm{tw}\mathrm{var}(\log{}_{10}\hat D)",
    label = "",
    xlim = (0.4,1.2),
    xticks = 0.4:0.2:1.2,
    linewidth = 2,
)
plot!(twinx(),2hs,res2,
    color =palette(:default)[3],
    xlim = (0.4,1.2),
    ylabel = L"{}_\mathrm{tw}\mathrm{var}(\hat α\,)",
    label = "",
    ribbon=  (zeros(lenH),-res2),
    ylim = (minimum(res2),maximum(res2)),
    linewidth = 2,
)
plot!([],[],
    linewidth = 2,
    color = palette(:default)[1],
    label = L"estimation of $D$",
)
plot!([],[],
    linewidth = 2,
    color = palette(:default)[3],
    label = L"estimation of $α$",
    legend = :topleft,
)

#savefig("inputHslice.pdf")

## heatmap plots

p1 = heatmap(2hs,2hs,Res1'*10^3,
    title = L"\mathrm{var}(\log\!{}_{10}\hat D)", 
    xlabel = L"true $α$",
    ylabel = L"GLS input $α$",
    fontfamily = "Computer Modern",
    xlim = (0.4,1.2),
    ylim = (0.4,1.2),
    xticks = 0.2:0.2:1.4,
    yticks = 0.2:0.2:1.4,
    clim = (3.8,4.7),
    axisratio = 1,
    framestyle = :box,
    annotatation
)
#savefig("inputH_D.pdf")

p2 = heatmap(2hs,2hs,Res2'*10^3,
    title = L"{}\mathrm{var}(\hat α)", 
    xlabel = L"true $α$",
    #ylabel = L"GLS input $α$",
    fontfamily = "Computer Modern",
    #xlim = (0.4,1.2),
    #ylim = (0.4,1.2),
    xticks = 0.4:0.2:1.2,
    yticks = 0.4:0.2:1.2,
    clims = (0.01,0.0275) .* 10^3,
    axisratio = 1
)
#savefig("inputH_H.pdf")

p3 = heatmap(2hs,2hs,Res3' ./ sqrt.(Res2' .* Res1'),
    title = L"\mathrm{corr}(\log\!{}_{10}\hat D,\hat α)", 
    xlabel = L"true $α$",
    #ylabel = L"GLS input $α$",
    fontfamily = "Computer Modern",
    color = :turbo,
    xlim = (0.4,1.2),
    ylim = (0.4,1.2),
    yticks = 0.4:0.2:1.2,
    xticks = 0.4:0.2:1.2,
    axisratio = 1
)
#savefig("inputH_DH.pdf")
using Plots.PlotMeasures
l = @layout [a{0.33w} b{0.33w} c{0.33w}]
plot!(p1,left_margin=6mm,top_margin=1mm)
plot!(p2,left_margin=0mm,top_margin=1mm)
plot!(p3,top_margin=1mm)
plot(p1,p2,p3,layout = l,size=(1200,300))

## with Makie 

using CairoMakie

set_theme!(theme_latexfonts())
fig = Figure(size = (1200, 400))
ga = GridLayout(fig[1, 1])
gb = GridLayout(fig[1, 2])
gc = GridLayout(fig[1, 3])
ax = Axis(ga[1,1])
hm = CairoMakie.heatmap!(ax,2hs,2hs,Res1*10^3,
    colormap = :thermal,
    colorrange  = (3.8,4.7),
)
#fig.fontsize = 12
ax.title = L"var($\log_{10}\,\hat{D}$)"
ax.titlesize = 20
ax.xlabel = L"true $\alpha$"
ax.ylabel = L"GLS input $\alpha$"
ax.ylabelsize= 20
ax.xticks = 0.4:0.2:1.2
ax.yticks = 0.4:0.2:1.2
ax.xlabelsize= 20
Colorbar(ga[:, end+1],hm,
    ticks = 3.8:0.1:4.7,
    ) 
colsize!(ga, 1, Aspect(1, 1.0))
Label(ga[1, 2, Top()], L"\cdot 10^{-3}",
    fontsize = 16,
    halign = :left,
)
#text!(fig,0.,0.,L"\cdot 10^{-3}")

ax2 = Axis(gb[1,1])
hm2 = CairoMakie.heatmap!(ax2,2hs,2hs,Res2'*10^3,
    colormap = :thermal,
    colorrange  = (10,28),
)
ax2.title = L"var($\hat{\alpha}$)"
ax2.titlesize = 20
ax2.xlabel = L"true $\alpha$"
ax2.xticks = 0.4:0.2:1.2
ax2.yticks = 0.4:0.2:1.2
ax2.xlabelsize= 20
Colorbar(gb[:, end+1],hm2,
    ticks = 10:2:28,
) 
colsize!(gb, 1, Aspect(1, 1.0))
Label(gb[1, 2, Top()], L"\ \ \ \cdot 10^{-3}",
    fontsize = 16,
    halign = :left,
)
ax3 = Axis(gc[1,1])
hm3 = CairoMakie.heatmap!(ax3,2hs,2hs,Res3 ./ sqrt.(Res2 .* Res1),
    colormap = :turbo,
    #colorrange  = (3.8,4.7),
)
CairoMakie.contour!(ax3,2hs,2hs,Res3,levels=[0.0],
    linestyle = :dash,
    color = :white,
    linewidth = 2,
)
ax3.title = L"corr($\log_{10}\,\hat{D},\ \hat{\alpha}$)"
ax3.titlesize = 20
ax3.xlabel = L"true $\alpha$"
ax3.xlabelsize= 20
ax3.yticks = 0.4:0.2:1.2
ax3.xticks = 0.4:0.2:1.2
Colorbar(gc[:, end+1],hm3,
    ticks = (-0.4:0.1:0.2,vcat(string.(-0.4:0.1:-0.1), " 0.0", " 0.1", " 0.2")),
)
colsize!(gc, 1, Aspect(1, 1.0))


#save("inputHeat.pdf",fig)


fig