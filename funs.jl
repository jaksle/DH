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

function estMean(X,k)
    n = length(X)
    d = Vector{Float64}(undef,k)

    for j in 1:k
        d[j] = mean( (X[i]-X[i+j]) for i in 1:n-j  )
    end
    return d
end


##

incrCov(i,j,k,l,K) = begin 
    a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    K(a,b) + K(a+c,b+d) - K(a,b+d) - K(a+c,b)
end
theorCov(k,l,ln,K) = begin
    2/((ln-k)*(ln-l)) * sum( incrCov(i,j,k,l,K)^2 for j in 1:ln-l, i in 1:ln-k )
end

function theorCovEff(k,l,ln,K) # zakres h?
    if k > l
        k, l = l, k
    end
    N1 = h-> ln-l-h+1
    N2 = h-> 
        if h <= l-k+1
            ln-l
        else
            ln-k-h+1
        end
    return 2/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov(1,h,k,l,K)^2 for h in 2:ln-l; init=0) + sum( N2(h)*incrCov(h,1,k,l,K)^2 for h in 1:ln-k ) )
end

function theorCovEff2(ln,K, i1=1,i2=ln-1) # tablicuje wartości kowariancji
    C = Matrix{Float64}(undef, ln, ln)
    for i in 1:ln, j in i:ln
        C[i,j] = K(ts[i],ts[j])
        C[j,i] = C[i,j]
    end
    incrCov2(i,j,k,l) = begin 
        a, b, c, d = i, j, k, l
        C[a,b] + C[a+c,b+d] - C[a,b+d] - C[a+c,b]
    end
    M = Matrix{Float64}(undef,ln-1,ln-1)
    M .= NaN
    for k1 in i1:i2, l1 in i1:i2
        k, l = k1, l1
        if k > l
            k, l = l, k
        end
        N1 = h-> ln-l-h+1
        N2 = h-> 
            if h <= l-k+1
                ln-l
            else
                ln-k-h+1
            end
        M[k1,l1] = 2/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov2(1,h,k,l)^2 for h in 2:ln-l; init=0) + sum( N2(h)*incrCov2(h,1,k,l)^2 for h in 1:ln-k ) )
    end
    return M
end

function crossCovEff(k,l,ln,K)
    if k > l
        k, l = l, k
    end
    N1 = h-> ln-l-h+1
    N2 = h-> 
        if h <= l-k+1
            ln-l
        else
            ln-k-h+1
        end
    return 4/((ln-k)*(ln-l)) *( sum(N1(h)*incrCov(1,h,k,l,K)*(==(1,h) + ==(1+k,h+l) - ==(1,h+l) - ==(1+k,h)) for h in 2:ln-l; init=0) + sum( N2(h)*incrCov(h,1,k,l,K)*(==(h,1) + ==(h+k,1+l) - ==(h,1+l) - ==(h+k,1)) for h in 1:ln-k ) )
end

"""
Covariance of 1D iid noise TA-MSD
"""
function noiseCov(ln,k,l)
    if k > l
        l, k = k, l
    end
    if k == l 
        return 4/(ln-k)^2 * ( (ln >= 2k) ? (3ln-4k) : (2ln-2k) )
    else
        return 4/((ln-k)*(ln-l)) * ( (ln >= k+l) ? ( 2ln-k-2l) : (ln-l) )
    end
end



errV(k,ln,K) = theorCov(k,k,ln,K)
lerrV(k,ln,K,b = ℯ) = theorCov(k,k,ln,K)/(log(b)*K(k,k))^2

