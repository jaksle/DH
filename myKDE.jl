
function myKDE(X,Y,x,y)
    n = length(X)
    s(y) = 0.1 + y
    pdf = 0.
    for i in eachindex(X)
        pdf += 1/s(Y[i])*exp( -((x-X[i])^2+(y-Y[i])^2)/(2s(Y[i])))
    end
    pdf *= 1/(2pi*n)
    return pdf
end


xs = LinRange(-5,-1,100)
ys = LinRange(-0.2,1.4,100)