

using LinearAlgebra, Statistics, Plots, SpecialFunctions 


##

function sim(H,D,ts,n) 
    m = length(ts)
    c(x,y) = sqrt(gamma(2x+1)*gamma(2y+1)*sin(pi*x)*sin(pi*y))/(2gamma(x+y+1)*sin(pi*(x+y)/2))
    K(H,dt,s,t) = c(H(s),H(t))*sqrt(D(s)*D(t))*(abs(t-s+dt)^(H(s)+H(t))-2abs(t-s)^(H(s)+H(t))+abs(t-s-dt)^(H(s)+H(t)))
    S = [K(H,step(ts),s,t) for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    c(x,y) = sqrt(gamma(2x+1)*gamma(2y+1)*sin(pi*x)*sin(pi*y))/(2gamma(x+y+1)*sin(pi*(x+y)/2))
    K(H,dt,s,t) = c(H(s),H(t))*sqrt(D(s)*D(t))*(abs(t-s+dt)^(H(s)+H(t))-2abs(t-s)^(H(s)+H(t))+abs(t-s-dt)^(H(s)+H(t)))
    X = randn(m, n)
    Y = A'*X
    return Y
end

##

c(x,y) = sqrt(gamma(2x+1)*gamma(2y+1)*sin(pi*x)*sin(pi*y))/(2gamma(x+y+1)*sin(pi*(x+y)/2))
K(H,dt,s,t) = c(H(s),H(t))*sqrt(D(s)*D(t))*(abs(t-s+dt)^(H(s)+H(t))-2abs(t-s)^(H(s)+H(t))+abs(t-s-dt)^(H(s)+H(t)))


##
D(t) = 1 #+ t/2

H(t) = t < 1 ? 0.2 : 0.4
#H(t) = 0.2
H(t) = 0.2*atan(100*(t-1))/pi +1/2 + 0.2
H3(t) = if t < 10
    0.2
elseif t < 20
    0.6
else
    0.2
end
H(t) = 0.8 - 0.6t
#H(t) = (cos(t)+1)/8
#H(t) = 0.2

m = 10000 # no of points
ts = LinRange(0.01,1,m)
S = [K(H,step(ts),s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U


##


n = 10^2 # sample

X = randn(m, n)
Y = A'*X

##


msd = mean(cumsum(Y,dims=1) .^ 2,dims = 2)

plot(ts[2:end],msd[2:end],
    #xlim = (9,11),
    xscale=:log10,
    yscale=:log10,
    label="",
)
vline!([1],linestyle=:dash, label = "")

#plot!(t->t^(2*0.2),step(ts),50)

##

##

n = 10^2 # sample

X = randn(m, n)
Y = A'*X


##

h1, h2 = 0.2, 0.4
c1, c2 = 1, 2
τ = 3
ta, tb = 0.001, 10



