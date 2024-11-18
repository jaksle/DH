
using Plots, ProgressMeter, LaTeXStrings

##

hs = 0.2:0.01:0.6
lenH = length(hs)
b = 10
ln = 100
m = 50
ts = 1:ln

M = Array{Float64}(undef,lenH,m,m)

@showprogress for (a,h) in enumerate(hs)
    f = (s,t) -> 1/2*(t^(2h)+s^(2h)-abs(s-t)^(2h))
    M[a,:,:] .= [1/(log(b))^2*theorCovEff(i,j,ln,f)/(f(ts[i],ts[i])*f(ts[j],ts[j])) for i in 1:m, j in 1:m]
end

n = 10^5
Ts = [ones(m) log10.(ts[1:m])]
gR = [(Ts'*M[a,:,:]^-1*Ts)^-1*Ts'*M[a,:,:]^-1 for a in 1:lenH]
R = (Ts'*Ts)^-1*Ts'
vlD[k] = var(B[1,:])
v2H[k] = var(B[2,:])

Res1 = Matrix{Float64}(undef,lenH,lenH)
Res2 = similar(Res1)
Res3 = similar(Res1)



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

##

heatmap(hs,hs,Res1',
    title = L"{}_\mathrm{tw}\mathrm{var}(\log\!{}_{10} D)", 
    xlabel = L"true $H$",
    ylabel = L"GLS input $H$",
    fontfamily = "Computer Modern",
    xlim = (0.2,0.6),
    xticks = 0.2:0.1:0.6,
    axisratio = 1,
    color = :magma,
)
savefig("inputH_D.pdf")

heatmap(hs,hs,Res2',
    title = L"{}_\mathrm{tw}\mathrm{var}(2H)", 
    xlabel = L"true $H$",
    ylabel = L"GLS input $H$",
    fontfamily = "Computer Modern",
    xlim = (0.2,0.6),
    xticks = 0.2:0.1:0.6,
    clims = (0.01,0.0225),
    axisratio = 1
)
savefig("inputH_H.pdf")

heatmap(hs,hs,Res3',
    title = L"{}_\mathrm{tw}\mathrm{cov}(\log\!{}_{10} D, 2H)", 
    xlabel = L"true $H$",
    ylabel = L"GLS input $H$",
    fontfamily = "Computer Modern",
    color = :turbo,
    xlim = (0.2,0.6),
    xticks = 0.2:0.1:0.6,
    #clims = (0.01,0.0225),
    axisratio = 1
)
savefig("inputH_DH.pdf")