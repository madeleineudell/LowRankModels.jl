export ConvergenceHistory, update!

type ConvergenceHistory
    name::String
    objective::Array
    primal_residual::Array
    dual_residual::Array
    times::Array
    optval
end
ConvergenceHistory(name::String,optval=0) = ConvergenceHistory(name,Float64[],Float64[],Float64[],Float64[],optval)

function update!(ch::ConvergenceHistory, dt, 
                 obj=0, pr=0, dr=0)
    push!(ch.objective,obj)
    push!(ch.primal_residual,pr)
    push!(ch.dual_residual,dr)
    if isempty(ch.times)
        push!(ch.times,dt)
    else
        push!(ch.times,ch.times[end]+dt)
    end
end