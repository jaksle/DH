

# Theil-Sen
theil = Vector{Float64}(undef,n)
theilT = fill(NaN,9,9)
for k in 1:n
    for i in 1:5, j in i+1:6
        theilT[i,j] = (lmsd[j,k]-lmsd[i,k])/(lts[j] - lts[i] )
    end
    theil[k] = median(filter(!isnan,theilT))
end


## box fit
cB = Matrix{Float64}(undef, 2, n)

l, u = [0.,-Inf], [2.,Inf]


@showprogress for k in 1:n
    l, u = [-Inf,0.], [Inf,2.]
    sA = max(0.01,B[2,k])
    sA = min(2.,sA)
    opt = optimize((par) -> sum( (lmsd[1:5,k] .- par[2] .* lts[1:5] .- par[1]) .^2), l, u, [B[1,k],sA])
    cB[:,k] = Optim.minimizer(opt)
end
