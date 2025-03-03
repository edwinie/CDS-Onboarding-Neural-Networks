import os
import unittest

path = os.path.expanduser("~/CDS/Tio_spotify_songs.csv")
def testdownload():
  assertTrue(os.path.exists(path))
  return "You have downloaded the dataset correctly!"


testdownload()
