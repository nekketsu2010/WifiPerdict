import sys
import os
import WavToImage
import PictureToNumpy
import PictureTraining
import NewPictureTraining

number = sys.argv[1]
#一通りフォルダ作成
top = str(number) + "回目"
imageFolders = ['Image', 'Image_wn', 'Image_ss', 'Image_st', 'Image_com']
if not os.path.exists(top):
    os.mkdir(top)

for imageFolder in imageFolders:
    if not os.path.exists(str(top) + "/" + imageFolder):
        os.mkdir(str(top) + "/" + imageFolder)

if not os.path.exists(str(top) + "/TestImage"):
    os.mkdir(str(top) + "/TestImage")
if not os.path.exists(str(top) + "/Model"):
    os.mkdir(str(top) + "/Model")

WavToImage.main(number, train=True)
# WavToImage.main(number, train=False)
# PictureToNumpy.main(number, train=True)
# PictureToNumpy.main(number, train=False)
# PictureTraining.main(number)
# NewPictureTraining.main(number)