using LowRankModels
using FactCheck

TOL = 1e-3

facts("Classification Losses") do

  context("logistic loss") do
    l = LogisticLoss()
    l1 = 1.31326168
    l0 = 0.3132616875
    # 1 is true
    # -1 and 0 are both false
    # anything else is an error
    @fact evaluate(l, 1, true) --> roughly(l0, TOL)
    @fact evaluate(l, 1, false) --> roughly(l1, TOL)
    @fact evaluate(l, -1, true) --> roughly(l1, TOL)
    @fact evaluate(l, -1, false) --> roughly(l0, TOL)
    @fact evaluate(l, 1, 1) --> roughly(l0, TOL)
    @fact evaluate(l, 1, -1) --> roughly(l1, TOL)
    @fact evaluate(l, 1, 0) --> roughly(l1, TOL)
    @fact evaluate(l, -1, 1) --> roughly(l1, TOL)
    @fact evaluate(l, -1, -1) --> roughly(l0, TOL)
    @fact evaluate(l, -1, 0) --> roughly(l0, TOL)

    @fact evaluate(3*l, 1, false) --> roughly(3*l1, TOL)
  end

  context("hinge loss") do
    l = HingeLoss()
    # 1 is true
    # -1 and 0 are both false
    # anything else is an error
    @fact evaluate(l, 1, true) --> roughly(0, TOL)
    @fact evaluate(l, 1, false) --> roughly(2, TOL)
    @fact evaluate(l, -1, true) --> roughly(2, TOL)
    @fact evaluate(l, -1, false) --> roughly(0, TOL)
    @fact evaluate(l, 1, 1) --> roughly(0, TOL)
    @fact evaluate(l, 1, -1) --> roughly(2, TOL)
    @fact evaluate(l, 1, 0) --> roughly(2, TOL)
    @fact evaluate(l, -1, 1) --> roughly(2, TOL)
    @fact evaluate(l, -1, -1) --> roughly(0, TOL)
    @fact evaluate(l, -1, 0) --> roughly(0, TOL)

    @fact evaluate(3*l, 1, false) --> roughly(3*2, TOL)

    @fact grad(l, -1, true) --> roughly(-1, TOL)
    @fact grad(l, 2, true) --> roughly(0, TOL)
    @fact grad(l, -2, false) --> roughly(0, TOL)
    @fact grad(l, 2, false) --> roughly(1, TOL)
  end

end
