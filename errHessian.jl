using LsqFit, LinearAlgebra, Statistics

N = 100
ts = 1:N
D, H = 1, 0.4
σ = 0.1

ys = D .* ts .^ (2H)

M = 1000
estD = Vector{Float64}(undef, M)
estH = Vector{Float64}(undef, M)

for k in 1:M
    f = curve_fit((t,p) -> p[1] .* t .^(2p[2]), ts, ys .+ σ*randn(N), [0.9, 0.42])
    estD[k] = f.param[1]
    estH[k] = f.param[2]
end

v = ts .^ (2H)
w = @. 2D*log(ts)*ts^(2H)

vvA = N^(4H+1)/(4H+1)
vwA = 2D*N^(4H+1)/(4H+1)*(log(N) - 1/(4H+1))
wwA = 4D^2*N^(4H+1)/(4H+1)*(log(N)^2 + 2/(4H+1)^2 - 2log(N)/(4H+1))

detA = 4D^2*(N^(4H+1)/(4H+1))^2 * 1/(4H+1)^2

I = [ v⋅v v⋅w; v⋅w w⋅w]

Err = σ^2 * I^-1
