# GLS 

n2 = 10^3
nt = 0
l1, l2 = 10, 199
gls = Matrix{Float64}(undef, 2, n2)
mH = 1 - mean(ols[2,:])/2
mZ = sinpi(2mH)/(pi*mH*(1-2mH)*(2-2mH)) / (2*10^mean(ols[1,:]))
@threads for i in 1:n2
    h = 1 - ols[2,nt+i]/2
    if h < 1/2 || h > 1
        h = mH
    end 
    z = sinpi(2h)/(pi*h*(1-2h)*(2-2h)) / (2*10^ols[1,nt+i])
    if z < 1 || z > 100
        z = mZ
    end 
    C = theorCovEff2(ln,(s,t) -> covGLE(s,t,h,z), l1, l2)
    thMSD = covGLE.(ts,ts,h,z) # dim = 1
    #C = theorCovEff2(ln,(s,t) -> ζ^-1*sinpi(2H)/(2pi*H*(1-2H)*(2-2H)) *(t^(2-2H)+s^(2-2H)-abs(s-t)^(2-2H))) # FBM
    #thMSD = ζ^-1*sinpi(2H)/(pi*H*(1-2H)*(2-2H)) .* ts .^ (2-2H) # FBM
    #errC = [theorCovEff(i,j,ln,(s,t) -> ζ^-1*sinpi(2H)/(2pi*H*(1-2H)*(2-2H))*(t^(2-2H)+s^(2-2H)-abs(s-t)^(2-2H)))/(2thMSD[i]*thMSD[j]) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1] # FBM
    
    errC = [ C[i,j]/(2*thMSD[i] * thMSD[j]) * 1/(log(10)^2) for i in 1:ln-1, j in 1:ln-1 ] # dim = 2
    bias =  -log(10) .* diag(errC[l1:l2,l1:l2]) ./2
    gR = (Ts[l1:l2,:]'*errC[l1:l2,l1:l2]^-1*Ts[l1:l2,:])^-1*Ts[l1:l2,:]'*errC[l1:l2,l1:l2]^-1
    gls[:,i] .= gR*(lmsd[l1:l2,nt+i] .- bias)
    println(i)
end

gls[1,:] .-= log10(4)

gls08 = gls

#@save "simGLE2" gls08