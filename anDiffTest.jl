

cr = errCov_tamsd(ts, 0.1, 2)[2]

errC[:,:,1] ./ cr # ok

# load plotSData

init_A = B[2,:]

gls = fit_internal(msd, 2, dt, init_A)


## sim

n = 10^4
h = 0.3

f = (s,t) -> 1*(t^(2h)+s^(2h)-abs(s-t)^(2h)) # D = 1
S = [f(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U
ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ
msd = Matrix{Float64}(undef,ln-1,n)
for i in 1:n
    msd[:,i] .= estMSD(X[:,i],ln-1) .+ estMSD(Y[:,i],ln-1)
end

gls, c = fit_gls(msd, 2, dt, fill(0.6,n))

cov(gls')

exErr = (Ts'*errCov_tamsd(ts, 2, 2h)[2]^(-1)*Ts)^-1

gls2, c2 = fit_gls(msd[:,1:200], 2, dt, fill(0.6,n), precompute=false)