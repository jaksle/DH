

ln = 50

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



## trapped particles



H0, D0 = 0., 10^-3
n = 10^4
ln = 50
dt = 0.0567
ts = dt*(1:ln)

X = sqrt(D0)*randn(2ln, n) # czynnik 2 inny
Y = sqrt(D0)*randn(2ln, n)

K = (s,t) -> D0*(abs(t-s)< 1e-12)
Σ = [2theorCovEff(i,i2,ln,K)/(4K(ts[i],ts[i])*4K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!

eM = (Ts'*errC[:,:,1]^-1*Ts)^-1*Ts'*errC[:,:,1]^-1*Σ*errC[:,:,1]^-1*Ts*(Ts'*errC[:,:,1]^-1*Ts)^-1

eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1

eMF = copy(eM)
eMF[1,2], eMF[2,1] = 0., 0.

eM2F = copy(eM2)
eM2F[1,2], eM2F[2,1] = 0., 0.

