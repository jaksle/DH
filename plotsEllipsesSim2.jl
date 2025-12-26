
l = 10

K = (s,t) -> D0*(abs(t-s)< 1e-12)
Σ = [2theorCovEff(i,i2,ln,K)/(4K(ts[i],ts[i])*4K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!

eM = (Ts'*errC[:,:,1]^-1*Ts)^-1*Ts'*errC[:,:,1]^-1*Σ*errC[:,:,1]^-1*Ts*(Ts'*errC[:,:,1]^-1*Ts)^-1
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1

f1 = t-> cos(t)*sqrt(5.99) # 95% elipse
g1 = t-> sin(t)*sqrt(5.99)

C12 = sqrt(eM)
C11 = sqrt(eM2)

##
D0, H0 = 10^-3, 0.35

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
eM = (Ts'*Σ^-1*Ts)^-1 # GLS
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1 # OLS


C22 = sqrt(eM)
C21 = sqrt(eM2)

##
D0, H0 = 10^-3, 0.6

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
eM = (Ts'*Σ^-1*Ts)^-1 # GLS
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1 # OLS

C32 = sqrt(eM)
C31 = sqrt(eM2)