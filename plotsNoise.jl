
using CairoMakie, ProgressMeter, LaTeXStrings
using Statistics, Distributions, LinearAlgebra

include("funs.jl")





## noise cov test

ln = 6
n = 10^6
X = randn(ln,n)

thC = noiseC.(ln,1:ln-1,(1:ln-1)')

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end



lmsd = log.(msd)

## test sim

H0, D0 = 0.3, 10.
σ = 1.5
n = 10^6
ln = 100
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
lts2 = log10.(ts[2:ln])
Ts = [ones(ln-1) lts]

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [K(s,t) for s in ts, t in ts] 
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ σ .* randn(ln,n) # 1D

msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end
msd .-= 2σ^2

lg(x) = (x > 0) ? log10(x) : 0.
lmsd = lg.(msd)
lmsd[isnan.(lmsd)] .= 0.

p =plot(ts[1:end-1],mean(msd,dims=2)[:])
lines!(ts[1:end-1],K.(ts[1:end-1],ts[1:end-1]), color=:red)
p

p =plot(lts,mean(lmsd,dims=2)[:])
lines!(lts,2H0 .* lts .+ log10(2D0), color=:red)
p

p =plot(lts,mean(lmsd,dims=2)[:] .+ log(10) .*diag(eMErr)/2' )
lines!(lts,2H0 .* lts .+ log10(2D0), color=:red)
p
## test of err cov, brute force

ffn = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) + ( (s==t) ? σ^2 : 0. )
eM = [theorCovEff(i,j,ln,ffn) for i in 1:ln-1, j in 1:ln-1]

eMErr =  1/(log(10)^2) * [ (eM[k,l])/((2D0*ts[k]^(2H0))*(2D0*ts[l]^(2H0))) for k in 1:ln-1, l in 1:ln-1]



cv = cov(msd')
(cv .- eM) ./ cv

lcv = cov(lmsd')
(lcv .- eMErr) ./ lcv



## test of err cov, equation

f = (s,t) -> 1*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))



crossTh = [4/((ln-k)*(ln-l1))*sum(incrCov(i,j,k,l1,f)*(==(i,j) + ==(i+k,j+l1) - ==(i,j+l1) - ==(i+k,j)) for i in 1:(ln-k), j in 1:(ln-l1)) for k in 1:ln-1,l1 in 1:ln-1]

crossTh2 = fill(NaN,ln-1,ln-1)
for k in 1:ln-1, l1 in k:ln-1
    crossTh2[k,l1] = 4/((ln-k)*(ln-l1))*sum(
    incrCov(i,j,k,l1,f)*(==(i,j) + ==(i+k,j+l1) - ==(i,j+l1) - ==(i+k,j)) for i in 1:(ln-k), j in 1:(ln-l1)) 
end

thC = noiseCov.(ln,1:ln-1,(1:ln-1)')

e1 = [theorCovEff(i,j,ln,f) for i in 1:ln-1, j in 1:ln-1]

crossTh3 = [crossCovEff(k,l,ln,f) for k in 1:ln-1,l in 1:ln-1]

eMTh = D0^2*e1 .+ σ^2*D0*crossTh3 .+ σ^4*thC # zgadza się z eM i cov(msd')


#eTh =  1/(log(10)^2) * [(D0^2*e1[k,l] + σ^4*thC[k,l])/((2D0*ts[k]^(2H0))*(2D0*ts[l]^(2H0))) for k in 1:ln-1, l in 1:ln-1]

## GLS prep
ln = 100
dt =  1
ts = dt*(1:ln)

hs = 0.05:0.01:0.8
errTh = Array{Float64}(undef,ln-1,ln-1,length(hs))
crossTh = Array{Float64}(undef,ln-1,ln-1,length(hs))
bias = Array{Float64}(undef,ln-1,length(hs))

@showprogress for kH in eachindex(hs) # uwaga 1D czy 2D!
    f = (s,t) -> 1*(t^(2hs[kH])+s^(2hs[kH])-abs(s-t)^(2hs[kH]))
    for i in 1:ln-1, j in i:ln-1
        errTh[i,j,kH] = theorCovEff(i,j,ln,f)
        errTh[j,i,kH] = errTh[i,j,kH]
    end
    for i in 1:ln-1, j in i:ln-1
        crossTh[i,j,kH] = crossCovEff(i,j,ln,f)
        crossTh[j,i,kH] = crossTh[i,j,kH]
    end
end

thN = noiseCov.(ln,1:ln-1,(1:ln-1)')

## sim FBM

H0, D0 = 0.35, 10.
σ = 7
n = 10^4
ln = 100
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [K(s,t) for s in ts, t in ts] 
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ .+ σ .* randn(ln,n) # 1D



## msd fit
 
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1)
end

msd .-= 2σ^2
lg(x) = (x> 0) ? log10(x) : 0.
lmsd = lg.(msd)

w = 10 # window

B = Matrix{Float64}(undef, 2, n)
for i in 1:n
    B[:,i] .= (Ts[1:w,:]'*Ts[1:w,:])^-1*Ts[1:w,:]'*lmsd[1:w,i]
end

B[1,:] .-= log10(2)


## GLS fit

gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)
errC = Matrix{Float64}(undef,ln-1,ln-1)
@showprogress for i in 1:n
    j = findfirst(hs .>= B[2,i]/2) # H not α
    j === nothing && (j = length(hs))
    D = 10^B[1,i]
    #D = 10
    for k in 1:ln-1,l in k:ln-1
        errC[k,l] = 1/(log(10)^2) * (D^2*errTh[k,l,j] + σ^2*D*crossTh[k,l,j] + σ^4*thN[k,l])/((2D*ts[k]^(B[2,i]))*(2D*ts[l]^(B[2,i])))
        errC[l,k] = errC[k,l]
    end
    gR = (Ts'*errC^-1*Ts)^-1*Ts'*errC^-1
    gB[:,i] .= gR*lmsd[:,i]
    bB[:,i] .= gR*(lmsd[:,i] .+ log(10)*diag(errC) ./ 2)
end

gB[1,:] .-= log10(2)
bB[1,:] .-= log10(2)


##

mean(B[1,:])
mean(gB[1,:])
mean(bB[1,:])


mean(B[2,:])
mean(gB[2,:])
mean(bB[2,:])

var(bB[1,:])/var(B[1,:])

var(bB[2,:])/var(B[2,:])

mean(B[2,B[2,:] .> 0])
mean(gB[2,gB[2,:] .> 0])
mean(bB[2,bB[2,:] .> 0])

count(B[2,:] .<= 0 )
count(bB[2,:] .<= 0 )
count(pB[2,:] .<= 0 )

count(B[2,:] .>= 1 )
count(pB[2,:] .>= 1 )

count(0 .< B[2,:] .< 2)
count(0 .< bB[2,:] .< 2)

###########################################
## main sim loop

vs = LinRange(0,50,100)
mB = Matrix{Float64}(undef,2,100)
mbB = Matrix{Float64}(undef,2,100)

vB = Matrix{Float64}(undef,2,100)
vbB = Matrix{Float64}(undef,2,100)

H0, D0 = 0.35, 10.
n = 10^4
ln = 100
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]
lg(x) = (x > 0) ? log10(x) : NaN

errC = Matrix{Float64}(undef,ln-1,ln-1)
msd = Matrix{Float64}(undef,ln-1,n)
gB = Matrix{Float64}(undef, 2, n)
bB = Matrix{Float64}(undef, 2, n)
B = Matrix{Float64}(undef, 2, n)

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) 
S = [K(s,t) for s in ts, t in ts] 
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
orgX = A'*ξ

for (iv,v) in enumerate(vs)
    σ = sqrt(v)

    X = orgX + σ .* randn(ln,n) # 1D

    # msd fit
    
    for i in 1:n
        msd[:,i] .= estMSD(X[:,i],ln-1)
    end
    msd .-= 2σ^2

    lmsd = lg.(msd)

    w = 10 # window
    for i in 1:n
        mask = falses(ln-1)
        mask[1:w] .= .!isnan.(lmsd[1:w,i])
        if count(mask) > 2 # pechowe trajektorie
            B[:,i] .= (Ts[mask,:]'*Ts[mask,:])^-1*Ts[mask,:]'*lmsd[mask,i]
        else
            B[1,i] = log10(20)
            B[2,i] = 0.7
        end
    end

    B[1,:] .-= log10(2)

    # GLS fit

    @showprogress for i in 1:n
        j = findfirst(hs .>= B[2,i]/2) # H not α
        j === nothing && (j = length(hs))
        D = 10^B[1,i]

        for k in 1:ln-1,l in k:ln-1
            errC[k,l] = 1/(log(10)^2) * (D^2*errTh[k,l,j] + σ^2*D*crossTh[k,l,j] + σ^4*thN[k,l])/((2D*ts[k]^(B[2,i]))*(2D*ts[l]^(B[2,i])))
            errC[l,k] = errC[k,l]
        end
        
        mask = .!isnan.(lmsd[:,i])
        iC = inv(errC[mask,mask])
        gR = (Ts[mask,:]'*iC*Ts[mask,:])^-1*Ts[mask,:]'*iC
        gB[:,i] .= gR*lmsd[mask,i]
        bB[:,i] .= gR*(lmsd[mask,i] .+ log(10)*diag(errC)[mask] ./ 2)
    end

    gB[1,:] .-= log10(2)
    bB[1,:] .-= log10(2)

    m1 = 0 .< B[2,:] .< 2.
    m2 = 0 .< bB[2,:] .< 2.

    mB[1,iv] = mean(B[1,m1])
    mB[2,iv] = mean(B[2,m1])
    mbB[1,iv] = mean(bB[1,m2])
    mbB[2,iv] = mean(bB[2,m2])

    vB[1,iv] = var(B[1,m1])
    vB[2,iv] = var(B[2,m1])
    vbB[1,iv] = var(bB[1,m2])
    vbB[2,iv] = var(bB[2,m2])

    # mB[1,iv] = mean(B[1,:])
    # mB[2,iv] = mean(B[2,:])
    # mbB[1,iv] = mean(bB[1,:])
    # mbB[2,iv] = mean(bB[2,:])

    # vB[1,iv] = var(B[1,:])
    # vB[2,iv] = var(B[2,:])
    # vbB[1,iv] = var(bB[1,:])
    # vbB[2,iv] = var(bB[2,:])
end

##

#using JLD2
#@load "varComp.jld2" mB mbB vB vbB
#@save "varComp.jld2" mB mbB vB vbB

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800,400))

    ax = Axis(fig[1,1],
        limits = ((0,50),nothing),
        ylabel = L"\langle \log_{10}\, \hat{D} \rangle",
        title = L"Results for $\log_{10}\, \hat{D}$"
    )

    scatter!(ax,vs,mB[1,:],
        color = :dodgerblue2,
    )
    scatter!(ax,vs,mbB[1,:],
        color = :tomato,
        marker = :utriangle,
    )
    hlines!(ax,[1],color = :black,linestyle=:dash,alpha=0.5)

    ax = Axis(fig[1,2],
        limits = ((0,50),nothing),
        ylabel = L"\langle\hat{\alpha}\rangle",
        title = L"Results for $\hat{\alpha}$"
    )
    scatter!(ax,vs,mB[2,:],
        color = :dodgerblue2,
    )
    scatter!(ax,vs,mbB[2,:],
        color = :tomato,
        marker = :utriangle,
    )
    hlines!(ax,[0.7],color = :black,linestyle=:dash,alpha=0.5)

    ax = Axis(fig[2,1],
        limits = ((0,50),(0,nothing)),
        xlabel = L"$\sigma^2$ [L$^2$]",
        ylabel = ylabel = L"\text{var}(\log_{10}\, \hat{D})",
    )
    scatter!(ax,vs,vB[1,:],
        color = color = :dodgerblue2,
    )
    scatter!(ax,vs,vbB[1,:],
        color = :tomato,
        marker = :utriangle,
    )

    ax = Axis(fig[2,2],
        limits = ((0,50),(0,nothing)),
        xlabel = L"$\sigma^2$ [L$^2$]",
        ylabel = ylabel = L"\text{var}(\hat{\alpha})",
    )
    ols = scatter!(ax,vs,vB[2,:],
        color = :dodgerblue2,    
    )
    gls = scatter!(ax,vs,vbB[2,:],
        color = :tomato,
        marker = :utriangle,
    )
    axislegend(ax, [ols,gls],["OLS","GLS"],
        position=:lt,
    )

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)

    #save("noise.pdf",fig)
    fig
end