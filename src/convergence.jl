export ConvergenceHistory, update_ch!

type ConvergenceHistory
    name::AbstractString
    objective::Array
    dual_objective::Array
    primal_residual::Array
    dual_residual::Array
    times::Array
    stepsizes::Array
    optval
end
ConvergenceHistory(name::AbstractString,optval=0) = ConvergenceHistory(name,Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],optval)
ConvergenceHistory() = ConvergenceHistory("unnamed_convergence_history")

function update_ch!(ch::ConvergenceHistory, dt::Number, obj::Number,
                    stepsize::Number=0, pr::Number=0, dr::Number=0)
    push!(ch.objective,obj)
    push!(ch.primal_residual,pr)
    push!(ch.dual_residual,dr)
    push!(ch.stepsizes,stepsize)
    if isempty(ch.times)
        push!(ch.times,dt)
    else
        push!(ch.times,ch.times[end]+dt)
    end
end

function update_ch!(ch::ConvergenceHistory, dt; obj=0, stepsize=0, pr=0, dr=0, dual_obj=0)
    push!(ch.objective,obj)
    push!(ch.dual_objective,dual_obj)
    push!(ch.primal_residual,pr)
    push!(ch.dual_residual,dr)
    push!(ch.stepsizes,stepsize)
    if isempty(ch.times)
        push!(ch.times,dt)
    else
        push!(ch.times,ch.times[end]+dt)
    end
end

function show(ch::ConvergenceHistory)
    print("Convergence History for $(ch.name)\n\n")
    @printf "%16s%16s\n" "time (s)" "objective"
    for i=1:length(ch.objective)
        @printf "%16.2e%16.4e\n" ch.times[i] ch.objective[i]
    end
end
