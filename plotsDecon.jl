## 1 population deconvolution

n = 10^4
ln = 100
dt = 1
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

for k in eachindex(hs)
    bias[:,k] .=  -log(10) .* diag(errC[:,:,k]) ./2
end

##

H0, D0 = 0.35, 1

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

##


p1 = scatter(B[1,:],B[2,:],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 0.3,
    color = palette(:default)[1],
    label = "",
)

p2 = scatter(bB[1,:],bB[2,:],
    markerstrokewidth=0,
    markersize=0.5,
    alpha = 1,
    color = palette(:default)[3],
    label = "",
)

scatter!(p1, [log10(D0)],[2H0],marker=:x,color=:red, label = "")
scatter!(p2, [log10(D0)],[2H0],marker=:x,color=:red, label = "")


K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
eM = (Ts'*Σ^-1*Ts)^-1
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1

f(t) = cos(t)*sqrt(5.99) # 95% elipse
g1(t) = sin(t)*sqrt(5.99)

C = sqrt(eM2)

plot!(p1, t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "",
)
C = sqrt(eM)

plot!(p2, t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
    linewidth = 1,
    color = :black,
    linestyle = :dash,
    label = "",
)


den = kde((bB[1,:],bB[2,:]),boundary=((-0.2,-0.2),(0.4,1.0)),npoints=(500,500), )

heatmap(den.x,den.y,den.density',xlim=(den.x[1],den.x[end]),ylim=(den.y[1],den.y[end]))
hline!([0.7],color=:white, linestyle = :dash, label = "")
vline!([0.0],color=:white, linestyle = :dash, label = "")

savefig("oryg.png")

nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))

zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:100
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

heatmap(den.x,den.y,res',xlim=(den.x[1],den.x[end]),ylim=(den.y[1],den.y[end]))
hline!([0.7],color=:white, linestyle = :dash, label = "")
vline!([0.0],color=:white, linestyle = :dash, label = "")

savefig("rozplot.png")