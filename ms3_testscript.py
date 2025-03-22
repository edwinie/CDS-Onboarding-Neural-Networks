import os
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from ms3 import split_data

def test_split_data():
    test_df = pd.DataFrame({
        'feature1': range(100),
        'feature2': [i*2 for i in range(100)],
        'target': [i*3 for i in range(100)]
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(test_df, 'target')
    
    assert abs(len(X_train) - 70) <= 1, f"Train size too far from 70: {len(X_train)}"
    assert abs(len(X_val) - 15) <= 1, f"Val size too far from 15: {len(X_val)}"
    assert abs(len(X_test) - 15) <= 1, f"Test size too far from 15: {len(X_test)}"
    
    assert len(set(X_train.index) & set(X_val.index)) == 0, "Train/val overlap"
    assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test overlap"
    assert len(set(X_val.index) & set(X_test.index)) == 0, "Val/test overlap"
    
    assert len(X_train) + len(X_val) + len(X_test) == len(test_df)
    
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(
        test_df, 'target', train_size=0.8, val_size=0.1, test_size=0.1
    )
    assert len(X_train2) == 80, f"Expected train size: 80, got: {len(X_train2)}"
    
    return True

test_split_data()
print("Split data tests have passed")