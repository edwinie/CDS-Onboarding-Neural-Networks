import os
import unittest

path = os.path.expanduser("~/CDS/Top_spotify_songs.csv")
def testdownload():
  assert(os.path.exists(path))
  return True


testdownload()
