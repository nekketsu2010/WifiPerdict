import sys
import os

import TestWaveToImage
import WavToImage
import PictureToNumpy
import Test_PictureToNumpy
import PictureTraining

number = sys.argv[1]
#一通りフォルダ作成
top = str(number) + "回目"
# os.mkdir(top)
# os.mkdir(str(top) + "/Image")
# os.mkdir(str(top) + "/Model")
# os.mkdir(str(top) + "/TestImage")

WavToImage.main(number)
# TestWaveToImage.main(number)
PictureToNumpy.main(number)
# Test_PictureToNumpy.main(number)
PictureTraining.main(number)