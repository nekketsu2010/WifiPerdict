import sys
import os
import WavToImage
import PictureToNumpy
import PictureTraining

number = sys.argv[1]
#一通りフォルダ作成
# top = str(number) + "回目"
# os.mkdir(top)
# os.mkdir(str(top) + "/Image")
# os.mkdir(str(top) + "/Model")
# os.mkdir(str(top) + "/TestImage")

# WavToImage.main(number, train=True)
# WavToImage.main(number, train=False)
PictureToNumpy.main(number, train=True)
PictureToNumpy.main(number, train=False)
# PictureTraining.main(number)