using PGFPlots

plots = Array(PGFPlots.Linear, ntrials)
for i=1:ntrials
    plots[i] = Plots.Linear(chs[i].times, chs[i].objective - chs[i].optval)
end
a = Axis(plots, xlabel="time (s)", ylabel="objective suboptimality", width="5in")



using Gadfly
p = plot(df,x=:cum_train_time,y=train_error,Geom.line)
draw(PDF("regpath.pdf", 6inch, 3inch), p)