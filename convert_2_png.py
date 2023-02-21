from PIL import Image
import os
import glob
import numpy as np


im1 = Image.open('/home/mrblack/Hydromea/images_synchronized/test_image_7/image_left/00000.png')
print(im1.size) # Get the width and hight of the image for iterating over
print(im1.format) # Get the format of the image
print(im1.mode) # Get the mode of the image
im1 = np.array(im1)
print(im1.shape)


# print(im1[0])

# im1.save('/home/mrblack/Hydromea/images_synchronized/test_image_7/image_left/00000.png', 'PNG')
# im1 = im1.convert('RGB') # Convert to RGB
# im1.save('/home/mrblack/Hydromea/images_synchronized/test_image_7/image_left/000000.png', 'PNG')   

# print(im1.size) # Get the width and hight of the image for iterating over
# print(im1.format) # Get the format of the image
# print(im1.mode) # Get the mode of the image


#Do a loop through the whole directory and convert all the images to png:
#Directory where the images are: /home/mrblack/Projects_DL/DeepVO-pytorch/Unreal/test_image_7/image_left
#Directory where the images will be saved: /home/mrblack/Projects_DL/DeepVO-pytorch/Unreal/test_image_7/image_left

# for infile in glob.glob("/home/mrblack/Hydromea/images_synchronized/test_image_0/image_left/*.png"):
#     file, ext = os.path.splitext(infile)
#     im = Image.open(infile)
#     im.save(file + ".png", "PNG")
