## CRDIR_22-23.py
# Jeff Pelz 9/25/22
## Example main program to demonstrate assorted functions


import CRDIR_basic_funcs as cbf  # All functions are called as cbf.*** [e.g. cbf.select_raw_image(...)]

import os
import pathlib
from sys import platform  # Mac/PC/Linux?
import cv2

if __name__ == '__main__':  # Execute the following code when run (not called by another program)

    verbose = True  # Flag to inhibit (False) or enable (True) diagnostic print statements

    #########################  Select image directory and image list:  #######################
    rtPimgDir = "Images"  # Directory with images
    rtPreadDefaultImgNumList = [8]  # Which image[s] should we read from that directory?

    #########################  Set up environment  #######################
    # Set up the local environment; which computer are we running on?
    # user = ?  platform = ?
    computerOS = platform  # Mac, PC, Linux?
    homeDir = os.path.expanduser("~")  # *Home directory* for this computer
    subDir = 'CRDIR'  # Direct program to this subdirectory immediately off the home directory
    baseDir = pathlib.PurePath(homeDir, subDir)  # Define the base directory for this computer

    if verbose:
        print(f'Running exampleCRDIRmain.py.  Platform = {platform}, homeDir = {homeDir}, baseDir = {baseDir}')

    #########################  Get list of all images in image directory  #######################
    print(f"\nTo get a listing of all images in the specified directory, call:\n"
          f"    cbf.get_img_fnames(imgDir, extension, verbose=False)  <extension & verbose are optional>")

    absImgDir = pathlib.PurePath(baseDir, rtPimgDir)  # combine base directory and image subdirectory
    rawImgListWpath = cbf.get_img_fnames(absImgDir)  # Create list of all *raw* images in directory (default = 'nef')

    if verbose:
        print(f'\nImage directory = {absImgDir}\n')
        for idx, imgFname in enumerate(rawImgListWpath):
            print(f'{idx:3d}: {imgFname}')

    #########################  Load image[s] in image list read from JSON file  #######################

    # #####  Step through each image number in the JSON file
    for idx, imgNum in enumerate(rtPreadDefaultImgNumList):  # Cycle through each image number in the list

        if verbose:
            print(f'Reading image {idx} of specified files (imgNum = {imgNum})')

        # Read in raw image; return the raw image array, the CFA array, base filename, ISO and exposure time (sec)
        rawRender, rawImgArr, rawCFA, confidenceArr, rawImgFname, ISO, expSec, BPS = \
            cbf.select_raw_image(rtPimgDir, baseDir, defaultImgNumChoice=imgNum, verbose=False)

        if verbose:
            print(f'\nRead: raw image fname = {rawImgFname}  ISO = {ISO}  exposure = {expSec:0.4f} seconds')

        #########################  Display image[s] in image list  #######################
        scale = 0.25  # Image scale to display (1 = full size, 0.5 = half size)
        showForDuration = 2  # Display the image to the screen for this duration

        renderFig = cbf.show_for(rawRender, scaleTo255=True,
                                 title=f'rawRender of {rawImgFname}',
                                 scale=scale, interMethod=cv2.INTER_NEAREST, durationSec=showForDuration)

        #########################  Extract Bayer channels from image  #######################

        # Recode the Bayer channel codes from 0, 1, 1, 2 to 0,1,2,3
        rawCFA = cbf.recode_rggb_to_rg1g2b(rawCFA, RGrGbB=(0, 1, 2, 3), verbose=False)

        # Extract the four Bayer channel subImages [0 - 3], representing the R, G1, G2, & B channels
        # Note that the scale of each Bayer channel is 1/2 the full-image scale.
        bayerChannels = cbf.extract_4_color_channels(rawImgArr, rawCFA, verbose=verbose)
        bayerChanH, bayerChanW = bayerChannels[0].shape  # By definition, the Bayer channel is 2D

        # Look at each image (also scaled in size)
        for chanNum, channel in enumerate(bayerChannels):
            chanH, chanW = channel.shape[:2]  # Get height and width of each layer
            print(f'chanH, chanW = {chanH}, {chanW}')
            if chanNum == 0:  # display in upper left position
                winX, winY = 0 * scale, 0 * scale
            elif chanNum == 1:  # display in upper right position
                winX, winY = chanW * scale, 0 * scale
            if chanNum == 2:  # display in lower left position
                winX, winY = 0 * scale, chanH * scale
            if chanNum == 3:  # display in lower right position
                winX, winY = chanW * scale, chanH * scale

            cbf.show_for(channel, scaleTo255=True, winX=int(winX), winY=int(winY),
                         title=f'Bayer channel {chanNum} of {rawImgFname}',
                         scale=scale, interMethod=cv2.INTER_NEAREST, durationSec=showForDuration)

    print('Destroying all windows')
    cv2.destroyAllWindows()  # Uncheck "Run with Python console" Pycharm Run/Edit Configurations

    if verbose:
        print('\n\nEnd of Program')