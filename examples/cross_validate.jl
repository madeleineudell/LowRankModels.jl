using LowRankModels, DataFrames, Gadfly

do_cv = false
do_cv_by_iter = false
do_reg_path = true
do_plot = true

m,n,k = 200,200,3
A = randn(m,k)*randn(k,n) + k*sprandn(m,n,.05)
losses = fill(huber(),n)
r = quadreg(.1)
glrm = GLRM(A,losses,r,r,k+2)

function plot(df, x::Symbol, y::Array{Symbol}, scale = :linear, filename=None, height=3, width=6)
    dflong = vcat(map(l->stack(df,l,x),y)...)
    if scale ==:log
        p = plot(dflong,x=:times,y=:value,color=:variable,Geom.line,Scale.y_log10)
    else
        p = plot(dflong,x=:times,y=:value,color=:variable,Geom.line)
    end 	
	if filename
		println("saving figure in $filename")
		draw(PDF(filename, width*inch, height*inch), p) 
	end
	return p
end

if do_cv
    println("Computing cross validation error for each fold")
    train_error, test_error, train_glrms, test_glrms = cross_validate(glrm,5,Params(1,100,0,.001))
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_cv_by_iter
    println("Computing training and testing error as a function of iteration number")
    train_error, test_error = cv_by_iter(glrm)
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_reg_path
    println("Computing regularization path")
    train_error, test_error, train_time, model_onenorm, reg_params = 
    regularization_path(glrm, params=Params(1,50,.00001,.01), reg_params=logspace(2,-2,15))
    df = DataFrame(train_error = train_error, test_error = test_error,
                   train_time = train_time, model_onenorm=model_onenorm, reg_param = reg_params)
	if do_plot 
		p = plot(df, :reg_param, [:train_error, :test_error], scale = :log, "reg_path.pdf")
	end
end

println(df)