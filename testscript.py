import os
def testdownload():
  path = input("What is the path to the dataset? ")
  assert os.path.exists(path)
  return "You have downloaded the dataset!" 


testdownload()
