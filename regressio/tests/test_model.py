import numpy as np
from regressio.model import RegressionModel

def test_train_predict_linear():
    X = np.array([0, 1, 2, 3, 4], dtype=float)
    y = 2.0 * X + 1.0  
    rm = RegressionModel()
    rm.train(X, y)
    preds = rm.predict([5.0])
    assert pytest.approx(preds[0], rel=1e-6) == 11.0  
    assert pytest.approx(rm.r2, rel=1e-6) == 1.0