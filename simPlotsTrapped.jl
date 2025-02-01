

H0, D0 = 0., 10^-3
n = 10^4
ln = 100
dt = 0.0567
ts = dt*(1:ln)

X = sqrt(D0)*randn(length(ts), n) # czynnik 2 inny
Y = sqrt(D0)*randn(length(ts), n)

K = (s,t) -> D0*(abs(t-s)< 1e-12)
Σ = [2theorCovEff(i,i2,ln,K)/(4K(ts[i],ts[i])*4K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!

eM = (Ts'*errC[:,:,1]^-1*Ts)^-1*Ts'*errC[:,:,1]^-1*Σ*errC[:,:,1]^-1*Ts*(Ts'*errC[:,:,1]^-1*Ts)^-1

eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1

