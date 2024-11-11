using MAT

file = matopen("prl_trajectories_untreated.mat")

var = keys(file)

X = read(file,"trajx")
Y = read(file,"trajy")

##