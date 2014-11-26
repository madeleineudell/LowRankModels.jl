#module Plot

import Gadfly
import DataFrames: DataFrame

export plot

function plot(df::DataFrame, xs::Symbol, ys::Array{Symbol, 1}; scale = :linear, filename=None, height=3, width=6)
    dflong = vcat(map(l->stack(df,l,xs),ys)...)
    if scale ==:log
        p = Gadfly.plot(dflong,x=xs,y=:value,color=:variable,Gadfly.Scale.y_log10)
    else
        p = Gadfly.plot(dflong,x=xs,y=:value,color=:variable)
    end     
    if !(filename==None)
        println("saving figure in $filename")
        Gadfly.draw(Gadfly.PDF(filename, width*Gadfly.inch, height*Gadfly.inch), p) 
    end
    return p
end

#end # module