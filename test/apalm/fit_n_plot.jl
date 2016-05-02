# fit w/o apalm
params = APALMParams(nb_tries=1, delay=1)
history = LowRankModels.History()
@time X,Y,ch = fit!(glrm, params, history=history);
XYh = X'*Y;
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE\n")

# fit w/ apalm
params_apalm = APALMParams(nb_tries=5, delay=5)
history_apalm = LowRankModels.History()
@time Xa,Ya,cha = fit!(glrm_apalm, params_apalm, history=history_apalm);
XYha = Xa'*Ya;
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE\n")

using PyPlot

# step size
figure()
plot(history_apalm.FPR, label="APALM")
plot(history.FPR, label="PALM")
ylabel("Step size")
xlabel("Iteration")
legend()
semilogy()
savefig(file_prefix*"_stepsize.pdf")

# tries
figure()
plot(history_apalm.tries, label="APALM")
ylabel("Tries")
xlabel("Iteration")
savefig(file_prefix*"_tries.pdf")

# objective
figure()
plot(cha.objective, label="APALM")
plot(ch.objective, label="PALM")
ylabel("Objective")
xlabel("Iteration")
legend()
semilogy()
savefig(file_prefix*"_objective.pdf")

figure()
plot(cha.times, cha.objective, label="APALM")
plot(ch.times, ch.objective, label="PALM")
ylabel("Objective")
xlabel("Time (s)")
legend()
semilogy()
savefig(file_prefix*"_objectivebytime.pdf")