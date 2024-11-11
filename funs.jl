using Statistics

function estMSD(X,k)
    n = length(X)
    d = Vector{Float64}(undef,k)

    for j in 1:k
        d[j] = mean( (X[i]-X[i+j])^2 for i in 1:n-j  )
    end
    return d
end

##

incrCov(i,j,k,l,K) = begin 
    a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    K(a,b) + K(a-c,b-d) - K(a,b-d) - K(b,a-c)
end
theorCov(k,l,n,K) = begin
    n
    2/((n-k)*(n-l)) * sum( incrCov(i,j,k,l,K)^2 for j in l+1:n, i in k+1:n )
end

errV(k,n,K) = theorCov(k,k,n,K)
lerrV(k,n, K,b = exp(1) ) = theorCov(k,k,n,K)/(log(b)*K(k,k))^2

