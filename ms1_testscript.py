import os
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

path = os.path.join(".", "Top_spotify_songs.csv")
print(path)
def testdownload():
  assert(os.path.exists(path)), "ERROR: Dataset not found at " + path + ". Please download it."
  print("Dataset found")
  return True

testdownload()