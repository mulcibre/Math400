import os,sys,glob,Image
#import pdb; pdb.set_trace()

lCrop = 150
tCrop = 35
rCrop = 125
bCrop = 40

#   load all files in current directory with .png extension
imageFiles = []
for file in glob.glob("*.png"):
    imageFiles.append(file)

#   debug print    
print imageFiles
print

#   trim margins on each image
for image in imageFiles:
    imageToResize = Image.open(image)
    w, h = imageToResize.size
    imageToResize.crop((lCrop, tCrop, w-rCrop, h-bCrop)).save(image)