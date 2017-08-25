# Calling LowRankModels from python

This workflow relies on two major tools.

* [PyCall](https://github.com/JuliaPy/PyCall.jl): first make sure the `PYTHON` environment variable is set:
at the shell, `export PYTHON=\`which python\``.
Then start julia and run
```
Pkg.add("PyCall")
Pkg.build("PyCall")
using PyCall
```

* [pyjulia](https://github.com/JuliaPy/pyjulia): install this manually via git:
```
git clone https://github.com/JuliaPy/pyjulia
cd pyjulia
sudo python setup.py install
```

* And of course make sure LowRankModels is installed

Now try out LowRankModels using the syntax [in this test file](https://github.com/madeleineudell/LowRankModels.jl/blob/python/python/hello_world.py)!
