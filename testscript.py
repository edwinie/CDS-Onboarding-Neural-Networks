import os
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

path = os.path.expanduser("~/CDS/Top_spotify_songs.csv")
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

class TestData(unittest.TestCase):
    
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

if __name__ == '__main__':
    unittest.main()