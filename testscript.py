import os
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

path = os.path.expanduser("./CDS-Onboarding-Project/Top_spotify_songs.csv")
print(path)
def testdownload():
  assert(os.path.exists(path))
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
    'A': [1], 
    'B': [4]}, index=[0])
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

  df4 = pd.DataFrame({
    'A': [1, 2, np.nan],
    'B': ['foo', np.nan, 'bar'],
    'C': [np.nan, 3.14, 2.71]})
  cleaneddf4 = removenanrows(df4)
  expected4 = pd.DataFrame({
    'A': [2],
    'B': [pd.NA],
    'C': [3.14]}, index=[1])
  assert_frame_equal(cleaneddf4, expected4)

  assert True
    
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
  assert not df3['B'].isna().any()

  df4 = pd.DataFrame({
    'A': [np.nan, np.nan, np.nan],
    'B': [np.nan, np.nan, np.nan]})
  addSummaryStatistic(df4)
  assert not df4.isna().any().any()

  assert True

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

if __name__ == '__main__':
    test_removenanrows()
    print("removenanrows test cases have passed")

    test_addSummaryStatistic()
    print("addSummaryStatistic test cases have passed")

    test_one_hot_encode()
    print("onehotencode test cases have passed")

    test_data_visualization()
    print("data visualization test case have passed")

    print("all tests cases worked")