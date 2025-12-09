def test_argmin_reducer():
    from mneme.tuning.reducers import ArgMinReducer

    r = ArgMinReducer()
    cfgs = ["A", "B", "C"]
    vals = [3.0, 1.5, 2.0]
    result = r.reduce(cfgs, vals)
    assert result == "B"


def test_argmax_reducer():
    from mneme.tuning.reducers import ArgMaxReducer

    r = ArgMaxReducer()
    cfgs = ["A", "B", "C"]
    vals = [3.0, 1.5, 5.0]
    result = r.reduce(cfgs, vals)
    assert result == "C"
