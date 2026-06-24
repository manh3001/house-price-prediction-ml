# tests/test_predict.py
import predict


def test_predict_all_defaults_is_realistic():
    price = predict.predict({})
    assert 50_000 < price < 800_000


def test_higher_quality_raises_price():
    low = predict.predict({"OverallQual": 3})
    high = predict.predict({"OverallQual": 10})
    assert high > low


def test_build_input_has_all_columns_one_row():
    df = predict.build_input({"GrLivArea": 2000})
    assert df.shape[0] == 1
    assert df.loc[0, "GrLivArea"] == 2000
    assert df.shape[1] == len(predict._columns)
