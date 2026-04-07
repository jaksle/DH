using QuadGK, SpecialFunctions
using LinearAlgebra
using CairoMakie
##

function spectrum(ω,s, t, H, ζ) # ω > 0, m = 1, kBT = 1
    g = gamma(2H+1)
    return (cos((t-s)*ω) - cos(t*ω) - cos(s*ω) + 1  ) * ω^(-2) *  ζ*2g*sinpi(H)*ω^(1-2H) / abs(ζ*g*ω^(1-2H)*(sinpi(H) - im*cospi(H)) - im*ω)^2
end

function covGLS(s, t, H, ζ) # ω > 0, m = 1, kBT = 1
    quadgk(ω->spectrum(ω,s, t, H, ζ),0, Inf)[1]/π
end

##

ts = 1:100
H, ζ = 0.6,  0.5
C = Symmetric([covGLS(s,t, H, ζ) for s in ts, t in ts])

##
fig = Figure()
ax = Axis(fig[1,1],
    xscale = log10,
    yscale = log10,
)

scatter!(ax, ts,diag(C))
lines!(ax,ts, ζ^-1*sinpi(2H)/(pi*H*(1-2H)*(2-2H)) .* ts .^ (2-2H))
#lines!(ax,ts, ts .^ 2)
fig

##