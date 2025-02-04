
##

 # 0.35, 10^-3    0.49, 10^-2  0.25, 10^-4  0.6, 10^-4    0.15, 10^-2
 H0, D0 = 0.15, 10^-2
 n = 10^4
 ln = 50
 dt = 0.0567
 ts = dt*(1:2ln)
 
 K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
 S = [K(s,t) for s in ts, t in ts]
 A = cholesky(Symmetric(S)).U
 ξ = randn(length(ts), 2n)
 X = A'*ξ
 ξ = randn(length(ts), 2n)
 Y = A'*ξ

## msd fit
 
 msd1 = Matrix{Float64}(undef,ln-1,n)
 msd2 = Matrix{Float64}(undef,ln-1,n)
 for i in 1:n
     msd1[:,i] .= estMSD(X[1:ln,i],ln-1) .+ estMSD(Y[1:ln,i],ln-1) 
     msd2[:,i] .= estMSD(X[ln+1:end,i],ln-1) .+ estMSD(Y[ln+1:end,i],ln-1) 
 end


 
 lmsd1 = log10.(msd1)
 lmsd2 = log10.(msd2)
 
 lts = log10.(ts[1:ln-1])
 Ts = [ones(ln-1) lts]
 l = 5
 
 B1 = Matrix{Float64}(undef, 2, n)
 B2 = Matrix{Float64}(undef, 2, n)
 for i in 1:n
     B1[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd1[1:l,i]
     B2[:,i] .= (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*lmsd2[1:l,i]
 end
 
 B1[1,:] .-= log10(4)
 B1[2,:] .-= log10(4)
 
 # GLS fit
 
 #tB = Matrix{Float64}(undef, 2, n)
 #tbB = Matrix{Float64}(undef, 2, n)
 gB1 = Matrix{Float64}(undef, 2, n)
 bB1 = Matrix{Float64}(undef, 2, n)
 gB2 = Matrix{Float64}(undef, 2, n)
 bB2 = Matrix{Float64}(undef, 2, n)
 
#K = (s,t) -> D0/2*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0)) # no ln(10)
#tC = [2theorCovEff(i,j,ln,K)/(2K(ts[i],ts[i])*2K(ts[j],ts[j])) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1]
#tg = (Ts'*tC^-1*Ts)^-1*Ts'*tC^-1

 for i in 1:n
     j = findfirst(hs .>= B1[2,i]/2) # H not α
     j === nothing && (j = length(hs))
     gR = (Ts'*errC[:,:,j]^-1*Ts)^-1*Ts'*errC[:,:,j]^-1
     #tB[:,i] = tg*lmsd1[:,i]
     #tbB[:,i] = tg*(lmsd1[:,i] .+ log(10) .* diag(tC) ./2 )
     gB1[:,i] .= gR*lmsd1[:,i]
     bB1[:,i] .= gR*(lmsd1[:,i] .- bias[1:ln-1,j])
     gB2[:,i] .= gR*lmsd2[:,i]
     bB2[:,i] .= gR*(lmsd2[:,i] .- bias[1:ln-1,j])
 end
 
 #tB[1,:] .-= log10(4)
 #tBb[1,:] .-= log10(4)
 gB1[1,:] .-= log10(4)
 bB1[1,:] .-= log10(4)
 gB2[1,:] .-= log10(4)
 bB2[1,:] .-= log10(4)
 
##
 
 
 scatter!(p1,B1[1,:],B2[2,:],
     markerstrokewidth=0,
     markersize=0.5,
     alpha = 0.3,
     #color = palette(:default)[1],
     color = palette(:default)[2],
     label = "",
 )
 
 scatter!(p2,bB1[1,:],bB2[2,:],
     markerstrokewidth=0,
     markersize=0.5,
     alpha = 0.3,
     #color = palette(:default)[3],
     color = palette(:default)[7],
     label = "",
 )
 
 scatter!(p1, [log10(D0)],[2H0],marker=:x,color=:red, label = "")
 scatter!(p2, [log10(D0)],[2H0],marker=:x,color=:red, label = "")
 
##

K = (s,t) -> D0*(t^(2H0)+s^(2H0)-abs(s-t)^(2H0))
theorCovSplit(k,l,ln,s,K) = begin
    2/((s-k)*(ln-s-l)) * sum( incrCov(i,j,k,l,K)^2 for j in s+1:ln-l, i in 1:s-k )
end

Σ = [2theorCovEff(i,i2,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!
cΣ = [2theorCovSplit(i,i2,2ln,ln,K)/(2K(ts[i],ts[i])*2K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] # factor 2!

eM = (Ts'*Σ^-1*Ts)^-1
ceM = (Ts'*Σ^-1*Ts)^-1*Ts'*Σ^-1*cΣ*Σ^-1*Ts*(Ts'*Σ^-1*Ts)^-1
eMF = copy(eM)
eMF[1,2], eMF[2,1] = ceM[1,2], ceM[1,2]

ceM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*cΣ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1
eM2 = (Ts[1:l,:]'*Ts[1:l,:])^-1*Ts[1:l,:]'*Σ[1:l,1:l]*Ts[1:l,:]*(Ts[1:l,:]'*Ts[1:l,:])^-1
eM2F = copy(eM2)
eM2F[1,2], eM2F[2,1] = ceM2[1,2], ceM2[1,2]
 
 f(t) = cos(t)*sqrt(5.99) # 95% elipse
 g1(t) = sin(t)*sqrt(5.99)
 
 C = sqrt(eM2F)
 
 plot!(p1,t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
     linewidth = 1,
     color = :black,
     linestyle = :dash,
     label = "",
 )

 C = sqrt(eMF)

 plot!(p2,t->C[1,1]*f(t)+C[1,2]*g1(t) + log10(D0),t->C[2,1]*f(t)+C[2,2]*g1(t)+ 2H0,0,2pi,
     linewidth = 1,
     color = :black,
     linestyle = :dash,
     label = "",
 )
 