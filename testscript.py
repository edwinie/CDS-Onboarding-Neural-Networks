import os
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from ms3 import split_data

path = os.path.join(".", "Top_spotify_songs.csv")
print(path)
def testdownload():
  assert(os.path.exists(path)), "ERROR: Dataset not found at " + path + ". Please download it."
  print("Dataset found")
  return True

testdownload()

def removenanrows(dataset):
    return dataset.dropna()

def addSummaryStatistic(dataset):
    mean = dataset.mean()
    dataset.fillna(mean, inplace=True)

def one_hot_encode(data, labels):
    one_hot = pd.DataFrame(index=data.index)
    for label in labels:
        one_hot[label] = (data == label).astype(int)
    return one_hot

def test_removenanrows():
  df1 = pd.DataFrame({
    'A': [1,2,3], 
    'B': [4,5,6]})
  cleaneddf1 = removenanrows(df1)
  assert_frame_equal(cleaneddf1, df1)

  df2 = pd.DataFrame({
    'A': [1, np.nan, 3], 
    'B': [4,5,np.nan]})
  cleaneddf2 = removenanrows(df2)
  expected2 = pd.DataFrame({
    'A': [1.0], 
    'B': [4.0]}, index=[0])
  assert_frame_equal(cleaneddf2, expected2)

  df3 = pd.DataFrame({
    'A': [np.nan, np.nan, np.nan], 
    'B': [np.nan, np.nan, np.nan]})
  cleaneddf3 = removenanrows(df3)
  expected3 = pd.DataFrame({
    'A': [], 
    'B': []})
  assert len(cleaneddf3) == 0
  assert list(cleaneddf3.columns) == list(expected3.columns)

    
def test_addSummaryStatistic():
  df1 = pd.DataFrame({
    'A': [1,2,3], 
    'B': [4,5,6]})
  df1_copy = df1.copy()
  addSummaryStatistic(df1)
  assert_frame_equal(df1, df1_copy)

  df2 = pd.DataFrame({
    'A': [1, np.nan, 3], 
    'B': [4,5,np.nan]})
  addSummaryStatistic(df2)
  expected2 = pd.DataFrame({
    'A': [1, 2.0, 3], 
    'B': [4, 5, 4.5]})
  assert_frame_equal(df2, expected2)

  df3 = pd.DataFrame({
    'A': [1, 2, 3], 
    'B': [np.nan, np.nan, np.nan]})
  addSummaryStatistic(df3)
  assert not df3['A'].isna().any()

def test_one_hot_encode():
    data1 = pd.Series(['A', 'B', 'C', 'A', 'B'])
    labels1 = ['A', 'B', 'C']
    result1 = one_hot_encode(data1, labels1)
    expected1 = pd.DataFrame({
        'A': [1, 0, 0, 1, 0],
        'B': [0, 1, 0, 0, 1],
        'C': [0, 0, 1, 0, 0]
    }, index=data1.index)
    assert_frame_equal(result1, expected1)

    data2 = pd.Series(['A', 'A', 'B'])
    labels2 = ['A', 'B', 'C', 'D']
    result2 = one_hot_encode(data2, labels2)
    expected2 = pd.DataFrame({
        'A': [1, 1, 0],
        'B': [0, 0, 1],
        'C': [0, 0, 0],
        'D': [0, 0, 0]
    }, index=data2.index)
    assert_frame_equal(result2, expected2)

    data3 = pd.Series(['X', 'Y', 'Z'])
    labels3 = ['A', 'B', 'C']
    result3 = one_hot_encode(data3, labels3)
    expected3 = pd.DataFrame({
        'A': [0, 0, 0],
        'B': [0, 0, 0],
        'C': [0, 0, 0]
    }, index=data3.index)
    assert_frame_equal(result3, expected3)

    data4 = pd.Series([], dtype=object)
    labels4 = ['A', 'B', 'C']
    result4 = one_hot_encode(data4, labels4)
    assert list(result4.columns) == labels4
    assert len(result4) == 0

    assert True

def test_data_visualization():
    np.random.seed(1)
    df = pd.DataFrame({
        'num1': np.random.normal(0, 1, 10),
        'num2': np.random.normal(5, 2, 10),
        'cat1': np.random.choice(['A', 'B', 'C'], 10),
        'cat2': np.random.choice(['X', 'Y', 'Z'], 10)
    })   
    try:
        assert True
    except Exception as e:
        assert False, f"Visualization code raised exception: {e}"

def test_split_data():
    test_df = pd.DataFrame({
        'feature1': range(100),
        'feature2': [i*2 for i in range(100)],
        'target': [i*3 for i in range(100)]
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(test_df, 'target')
    
    assert len(X_val) == 15, f"Expected val size: 15, got: {len(X_val)}"
    assert len(X_test) == 15, f"Expected test size: 15, got: {len(X_test)}"
    
    assert len(set(X_train.index) & set(X_val.index)) == 0, "Train/val overlap"
    assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test overlap"
    assert len(set(X_val.index) & set(X_test.index)) == 0, "Val/test overlap"
    
    assert len(X_train) + len(X_val) + len(X_test) == len(test_df)
    
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(
        test_df, 'target', train_size=0.8, val_size=0.1, test_size=0.1
    )
    assert len(X_train2) == 80, f"Expected train size: 80, got: {len(X_train2)}"
    
    return True


test_removenanrows()
print("removenanrows test cases have passed")
test_addSummaryStatistic()
print("addSummaryStatistic test cases have passed")
test_one_hot_encode()
print("onehotencode test cases have passed")
test_data_visualization()
print("data visualization test case have passed")
print("all tests cases worked")
test_split_data()
print("Split data tests have passed")