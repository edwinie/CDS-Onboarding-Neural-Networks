import os
def testdownload():
  path = input("What is the path to the dataset? ")
  assert os.path.exists(path)
  print("You have downloaded the dataset!")


testdownload()
