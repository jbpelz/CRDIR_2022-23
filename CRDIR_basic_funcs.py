### basic functions for CRDIR project:
# Jeff Pelz 9/25/22

# Import required modules
import copy
import inspect
import os
import pathlib
from pathlib import Path
import time

import cv2  # Note: install module "opencv-python" to get cv2
import exifread
import numpy as np
import rawpy  # Note - installs correctly for python <= 3.7

# ###################################################################################################
# Read a specified raw image and return the rawImg object
def read_raw_image(absImgFnameWpath, verbose=False):

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')

    try:
        rawImg = rawpy.imread(absImgFnameWpath)  # Read raw image class
    except:
        print(f" ****************** |{absImgFnameWpath}| doesn't exist or is not raw image.  Returning None  *********")
        rawImg = None

    if verbose:
        print(f'np.unique(rawImg.raw_colors) = {np.unique(rawImg.raw_colors)}   ', end='')
        print(f'[Number of colors = {len(np.unique(rawImg.raw_colors))}]')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<bayerChan3Dimg.shape<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return rawImg


# ###################################################################################################
#
def get_ISO_exposuretime_and_bitdepth(imgFnameWpath, verbose=False):

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')
        print(f'Fetching data for file {imgFnameWpath}:  ')

    imgFile = open(imgFnameWpath, 'rb')  # Open the image file for reading (binary)
    allTags = exifread.process_file(imgFile)  # Get all the EXIF tags for this image file
    imgFile.close()  # Close the image file

    ISOval, exposureTimeSec, bitsPerSample = 0, None, None  # Initialize in case we can't find them in the EXIF data

    # Look for ISO value:
    if "EXIF ISOSpeedRatings" in allTags:
        ISOval = allTags["EXIF ISOSpeedRatings"]
        ISOval = float(ISOval.values[0])  # Extract ISO value from IdfTag object
    else:  # NOT FOUND: First - see whether the name is on the list of 'known ISO' files:
        if Path(imgFnameWpath).resolve().stem == 'iss031e168171':  # On list if 'known ISO files"
            ISOval = 102400
        elif Path(imgFnameWpath).resolve().stem == 'iss031e152638':  # On list if 'known ISO files"
            ISOval = 102400

        else:  # I don't know this one - ask for it manually:
            print(f'\n', '!'*100, '\n', '          NO EXIF ISO value found.\n', '!'*100, '\n')
            ISOtxt = input(f'   Manually enter an ISO value to be stored with the files (0) to ignore  ---> ')
            ISOval = int(ISOtxt)

    # Look for exposure time value:
    if "EXIF ExposureTime" in allTags:  # Note that the value returned is a set, whose first value is Ratio (num,den)
        exposureTimeIdfTag = allTags["EXIF ExposureTime"]

        # Extract numerator and denominator from IdfTag object and convert to float after division
        exposureTimeSec = float(exposureTimeIdfTag.values[0].num) / float(exposureTimeIdfTag.values[0].den)

    # Look for bits per sample (bps) value:
    if "Image BitsPerSample" in allTags:
        BPSval = allTags["Image BitsPerSample"]
        BPSval = float(BPSval.values[0])  # Extract ISO value from IdfTag object

    if verbose:
        if ISOval:
            print(f'ISO = {ISOval:,.0f}  ', end="")
        else:
            print(f'ISO not found   ', end="")

        if exposureTimeSec:
            print(f'  exposureTimeSec = {exposureTimeSec} seconds  ', end="")
        else:
            print(f'  exposureTimeSec not found  ', end="")

        if BPSval:
            print(f'BitsPerSample = {BPSval:,.0f}  <NOTE: value returned is for thumbnail - IGNORED>')
        else:
            print(f'BitsPerSample not found   ')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return ISOval, exposureTimeSec, BPSval


# ##########################################################################################
def in_func_called_by():
    return inspect.stack()[1][3], inspect.stack()[2][3]

# ###################################################################################################
# # Retrieve image files ending with a specified extension in a named directory
# # Return a list of strings
def get_img_fnames(imgDir, extension='nef', verbose=False):

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')

    imgList = []

    if not os.path.isdir(imgDir):
        print(f" ********************* {imgDir} doesn't exist or is not a directory.  *********************")
        return None

    for fname in os.listdir(imgDir):
        if fname.endswith(extension.lower()) or fname.endswith(extension.upper()):
            imgList.append(os.path.join(imgDir, fname))

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return imgList

# ###################################################################################################
# ................
# Aug 11, 2021 JBP: Add a 'confidence' image the same size as the raw image, initialized to 1's (presumed good)
def select_raw_image(imgDir, baseDir, imgType='nef', defaultImgNumChoice=-1, verbose=False):

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')
        print(f'imgDir = {imgDir}   baseDir = {baseDir}   imgType = {imgType}   '
              f'defaultImgChoice = {defaultImgNumChoice}')

    absImgDir = pathlib.PurePath(baseDir, imgDir)
    rawImgListWpath = get_img_fnames(absImgDir, imgType)  # Create list of all raw images in directory

    # Sort the image list in alphabetical order:
    rawImgListWpath.sort()

    for idx, rawImgFnamepath in enumerate(rawImgListWpath):
        # Extract just the filename (with extension) from the full path [Use .resolve().stem to get name w/o ext]
        rawImgFnameExt = Path(rawImgFnamepath).resolve().name
        if (defaultImgNumChoice >= 0) & (idx == defaultImgNumChoice):  # Indicate the default choice
            print(f'{idx:3}:  {rawImgFnameExt} <--- DEFAULT SELECTED ({defaultImgNumChoice})')
        else:
            print(f'{idx:3}:  {rawImgFnameExt}')

    # Use a default, or allow user to select one of the images:
    if defaultImgNumChoice >= 0:  # Using a default image number
        imgListNumber = defaultImgNumChoice

        if verbose:
            print(f'Using default image #{defaultImgNumChoice}')
            print(f'Image {rawImgListWpath[defaultImgNumChoice]} selected for reading.')
    else:
        imgListNumber = input('\n - - - - - - -> Enter the number of the image to load from the above list: ')

    rawImgFnameWpath = rawImgListWpath[int(imgListNumber)]
    rawImgFname = Path(rawImgFnameWpath).resolve().stem  # Extract the filename (no .ext) from the full path

    rawImgObj = read_raw_image(rawImgFnameWpath, verbose=False)

    # Create a BGR render from the raw object (we have to swap B <-> R channels to stay with cv2's BGR)
    rawRender = cv2.cvtColor(rawImgObj.postprocess(), cv2.COLOR_BGR2RGB)

    # Extract the visible portion of the raw image so that the mosiaced image will not include the borders around image
    rawArr = rawImgObj.raw_image_visible  # Extract raw arrays from the raw image class. Note: MOSAICED image
    rawCFAarr = rawImgObj.raw_colors_visible  # Extract the color-filter array mask; 0,1,2,3 for R, G_r, G_b, B

    # Create confidence image the same size as rawArr, initialized to 1.0 at each location
    # confidenceArr = np.ones_like(rawArr).astype(np.float64)  #  h x w x 1, initialized to 1.0
    confidenceArr = None  #  h x w x 1, initialized to 1.0

    # if verbose:
    #     print(f'confidenceArr:   shape = {confidenceArr.shape}    '
    #           f'type = {type(confidenceArr)} {(type(confidenceArr.flatten()[0]))} '
    #           f'{confidenceArr.flatten()[0]:0.2f} initialized to 1.0s')

    ISOval, expSecVal, BPSval = get_ISO_exposuretime_and_bitdepth(rawImgFnameWpath, verbose=verbose)


    if verbose:
        print(f'-'*120, '\n[Back to FUNC select_raw_image()]:  '
              f'For image file {rawImgFname}:  '
                        f'ISO = {ISOval}   Exposure (sec) = {expSecVal}   Bits Per Sample = {BPSval}')
        print(f'rawCFA order: \n{rawCFAarr[0:2, 0:2]}')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return rawRender, rawArr, rawCFAarr, confidenceArr, rawImgFname, ISOval, expSecVal, BPSval


# ###################################################################################################
# Open a cv2 window and display img for durationSec seconds (default = 1 sec)
# Optionally: place window at xinX,winY, and show only image section starting at r1,c1 of size h,w
#   showfor
def show_for(img, durationSec=1.0, title="show_for",
             txtStr=None, txtPos=(100, 100), txtSize=1, fontColor=(0,0,0), fontStroke=1,
             winX=0, winY=0,  # Position of window on screen
             scale=1.0, interMethod=cv2.INTER_LINEAR, # 'magnification' and interpolation method of image to display
             r1=0, c1=0, h=0, w=0,  # Crop image to img[r1:r2, c1:c2]
             normalize=False,  # use cv2.normalize() for > 8 bits, e.g.
             histEq=False,  # histEq = True, False, or [2; show both orig and clahe EQ in one frame]
             scaleTo255=False,  # histEq = True, False, or [2; show both orig and clahe EQ in one frame]
             # (x,y) points array, w/'pointColor' squares of pointSize and pointThickness
             pointArr=None, pointColor=(0, 0, 255), pointSize=9, pointThickness=1,  # pointColor=(50, 50, 200),
             saveFigure=False, saveDir=None, imgFolder=None,
             iso=0, expSec=0,
             verbose=False):

    # ?? setting waitKey to 0 should make it stay until a keypress, but it is causing a problem.  So make it 10 min
    if durationSec == 0:
        durationSec = 600
    # Check and clean r,c  h,w  and image size
    imgShape = img.shape
    imgH, imgW = imgShape[0], imgShape[1]

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')
        print(f'[r1,c1 = {r1, c1}  h,w = {h, w}]  img.shape = {img.shape}    imgH,imgW = {imgH, imgW}')

    if imgH == h & imgW == w:  # We are displaying the whole image
        dispImg = img

    else:  # Extract a section of the image to display
        r1,c1,  r2,c2, h,w = clean_section_defs(r1, c1, h, w, imgH, imgW, verbose=verbose)

        if len(imgShape) == 3:  # If it is a 3-dimensional array
            imgChans = imgShape[2]
        else:
            imgChans = 1

        if imgChans == 3:  # Color (multi-channel) image
            dispImg = img[r1:r2, c1:c2, :]

        elif imgChans == 1:  # Monochrome (1-channel) image
            dispImg = img[r1:r2, c1:c2]

        else:
            print(f'ERROR: show_for() can only display 1 or 3 channel images.  [imgShape = {imgShape}]')
            exit(' ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Bailed in show_for')

    if verbose:
        print(f'imgShape = {imgShape}   dispImg.shape = {dispImg.shape}')

    # Check for boolean mask & display float equiv if necessary
    # imgType = type(dispImg.flatten()[0])  # Get the first element, no matter the dimensions
    if isinstance(dispImg.flatten()[0],np.bool_):  # If it is a numpy boolean
        print(f'show_for() received a Boolean; converting ...')
        dispImg = (dispImg).astype(np.single)
        dispImg = cv2.normalize(dispImg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Optionally draw a number of squares on the image.
    if pointArr is not None:  # If there are any points to draw a square on the image
        for idx, point in enumerate(pointArr):
            pointArr[idx] = point[1], point[0]

        for idx, point in enumerate(pointArr):
            dispImg = cv2.rectangle(dispImg,
                                    (point[0]-pointSize//2, point[1]-pointSize//2),
                                    (point[0]+pointSize//2 + 1, point[1]+pointSize//2+1),
                                    color=pointColor, thickness=pointThickness)

    if normalize:  # Normalize image to fit in 8-bits
        print(f'DIAG - before:  \n{dispImg}')
        print(f'DIAG - before:  np.min(dispImg), np.max(dispImg) = {np.min(dispImg)}, {np.max(dispImg)}')
        dispImg = cv2.normalize(dispImg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        print(f'DIAG - after:  np.min(dispImg), np.max(dispImg) = {np.min(dispImg)}, {np.max(dispImg)}')
        print(f'DIAG - after :  \n{dispImg}')

    if scaleTo255:  # scale so max value is 255
        # dispImg = (dispImg/16).astype(np.uint8)
        maxVal = np.max(dispImg)
        scaleVal = 255/maxVal
        dispImg = (dispImg*scaleVal).astype(np.uint8)

    # Histogram equalize the image if histEq == True [if histEq = 2, equalize, and return BOTH]
    if histEq:  # perform contrast limited equalization:

        if len(dispImg.shape) == 3:  # If it is a color (3-chan) image
            dispImgEQ = clahe_color_img(dispImg)
        else:
            dispImgEQ = clahe_monochrome_img(dispImg)

        if histEq == 2:  # vstack the original and the histEq versions
            dispImg = np.vstack((dispImg, dispImgEQ))

            if isinstance(title, list):  # maintain the 2+ lines as sep list
                title[0] = f'{title[0]} [orig (top) & clahe EQ (bottom)]'
            else:
                title = f'{title} [orig (top) & clahe EQ (bottom)]'
        else:
            dispImg = dispImgEQ
            if isinstance(title, list):  # maintain the 2+ lines as sep list
                title[0] = f'{title[0]} [clahe EQ]'
            else:
                title = f'{title} [clahe EQ]'

        print(f'DIAG - after histEQ:  np.min(dispImg), np.max(dispImg) = {np.min(dispImg)}, {np.max(dispImg)}')
        print(f'DIAG - after histEQ :  \n{dispImg}')

    # Resize the image (larger or smaller) if scale != 1.0
    if scale != 1.0:
        dispImg = cv2.resize(dispImg, (0, 0), fx=scale, fy=scale, interpolation=interMethod)

        if isinstance(title, list):
            title[0] = f'{title[0]} (scale: {scale:0.2f}X)  [{r1}:{c1}, {r2}:{c2}]'  # Append scale & img range
        else:
            title = f'{title} (scale: {scale:0.2f}X)  [{r1}:{c1}, {r2}:{c2}]'

    if isinstance(title, list):  # If there are more lines, add the second one to the end for img display
        winTitle = f'{title[0]}'  # Just the first line for the window title
    else:  # Just one line for title
        winTitle = f'{title}  [{h}x{w}]'

    if txtStr:  # If there is one or more text strings to write on image:
        # Ensure a color image so that we can write color text
        if len(imgShape)==2:  # If it is a monochrome (2D) image:
            dispImg = np.dstack((dispImg, dispImg, dispImg))

        if isinstance(txtStr, list):  # If there are multiple strings to add to image:
            for idxStr, string in enumerate(txtStr):  # step through each string
                position = txtPos[idxStr]  # get the position for this string
                if isinstance(fontColor, list):  # There is one color per item:
                    fontClr = fontColor[idxStr]  # Get the color for this item
                else:  # Not a list; just one color for the whole image
                    fontClr = fontColor

                dispImg = write_str_on_image(dispImg, string, txtPos=position,
                                     fontClr=fontClr, fontStroke=fontStroke, fontSize=txtSize,
                                     font=cv2.FONT_HERSHEY_SIMPLEX,
                                     verbose=False)

        else:  # Not a list; just one string
            dispImg = write_str_on_image(dispImg, txtStr, txtPos=txtPos,
                                         fontClr=fontColor, fontStroke=fontStroke, font=cv2.FONT_HERSHEY_SIMPLEX,
                                         verbose=False)

    if durationSec > 0.01:  # Only move to front if it is meant to be viewed
        cv2.namedWindow(winTitle)  # Create a named window
        cv2.moveWindow(winTitle, winX, winY)  # Move window

    if durationSec > 0.01:  # Only move to front if it is meant to be viewed
        cv2.setWindowProperty(winTitle, cv2.WND_PROP_TOPMOST, 1)  # Ensure that this window will be 'front'
        cv2.imshow(winTitle, dispImg)

    if saveFigure:
        # First, insert ISO & expSec in saveFigure string:

        timeStr = time.strftime("%Y-%b-%d_%Hh%Mm")

        figFname = f'{saveFigure}_{timeStr}.jpg'

        inFunc, calledBy = in_func_called_by()  # In case verbose==False
        if not 'module' in calledBy.lower():  # If this was called by another function (not main)
            figFname = f'{saveFigure}_from_{calledBy}_{timeStr}.jpg'

        isList = isinstance(title, list)

        dispImg = add_title_image_header(dispImg, title, verbose=verbose)

        if saveDir:  # Prepend the path if it is given
            if imgFolder:  # is not None, the default - check and create folder with imgFolder
                imgFolderPath = os.path.join(saveDir, imgFolder)
                if not os.path.isdir(imgFolderPath):  # If doesn't exist yet
                    os.mkdir(imgFolderPath)

            figFnameWpath = os.path.join(imgFolderPath, figFname)
            cv2.imwrite(figFnameWpath, dispImg)

    key = cv2.waitKey(int(durationSec * 1000))  # milliseconds

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return dispImg

# ###################################################################################################
def add_title_image_header(img, titleTxt, headerH=50,
                           font=cv2.FONT_HERSHEY_SIMPLEX,
                           txtPos=(12, 25), fontClr=(0, 0, 0), fontStroke=1,
                           verbose=False):

    whiteVal = np.amax(img)  # Find maximum value in img to match white
    imgDims = list(img.shape)  # make it mutable
    imgDims[0] = headerH  # replace height with headerH for each line

    # If a list of titles is passed (i.e., titleTxt is list), then print each one.
    if isinstance(titleTxt, list):  # Treat as multiple lines; increase header and print each
        # Find the length of the longest string
        lenLongestLine = len(max(titleTxt, key=len))  # max(x, key=len) returns the longest one
        fontSize = -3.3898 * lenLongestLine / imgDims[1] + 0.8972  # Empirical fit for chars on width

        firstLine = True

        for titleLine in titleTxt:  # Get each line in the list
            titleLineImg = whiteVal * np.ones_like(img, shape=imgDims)  # Make white rect the same type & width as the image
            titleLineImg = cv2.putText(titleLineImg, titleLine, txtPos, font, fontSize, fontClr, fontStroke)

            if verbose:
                print(
                    f'title = |{titleTxt}| ({len(titleTxt)} chars)  imgDims[1] = {imgDims[1]}   fontSize = {fontSize:0.3f}')

            if firstLine:  # For the first line of the title;
                titleImg = titleLineImg
                firstLine = False
            else:
                titleImg = np.vstack((titleImg, titleLineImg))

    else:  # Set default vals for 1-line titleTxt
        lenLongestLine = len(titleTxt)

        imgDims[0] = headerH  # replace height with headerH
        titleImg = whiteVal*np.ones_like(img, shape=imgDims)  # Make white rect the same type & width as the image

        fontSize = -3.3898 * lenLongestLine/imgDims[1] + 0.8972  # Empirical fit for chars on width

        if verbose:
            print(f'title = |{titleTxt}| ({len(titleTxt)} chars)  imgDims[1] = {imgDims[1]}   fontSize = {fontSize:0.3f}')

        titleImg = cv2.putText(titleImg, titleTxt, txtPos, font, fontSize, fontClr, fontStroke)

    imgWtitle = np.vstack((titleImg, img))

    return imgWtitle

# ###################################################################################################
def write_str_on_image(imgIn, txtStr, txtPos=(150, 75), fontSize=1,
                       fontClr=(0, 150, 150), fontStroke=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                       verbose=False):
    img = copy.copy(imgIn)
    imgDims = list(img.shape)

    if verbose:
        print(f'title = |{txtStr}| ({len(txtStr)} chars)  imgDims[1] = {imgDims[1]}   fontSize = {fontSize:0.3f}')

    # First put an oversize, black charachter to form a dark background
    img = cv2.putText(img, txtStr, txtPos, font, fontSize, (10,10,10), int(fontStroke*3))
    img = cv2.putText(img, txtStr, txtPos, font, fontSize, fontClr, fontStroke)

    return img

# ###################################################################################################
def clahe_color_img(img, verbose=False):

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    gridsize = 8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ###################################################################################################
def clahe_monochrome_img(img, verbose=False):

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')

    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #
    # lab_planes = cv2.split(lab)

    gridsize = 8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))

    claheEqImg = clahe.apply(img)

    # lab = cv2.merge(lab_planes)

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return claheEqImg

# ###################################################################################################
# Check r1,c1 and h,w against image to be sure it 'fits in' the image.
def clean_section_defs(r1, c1, h, w, imgH, imgW, verbose=False):
    # Clean up individual functions by moving the validity checks of corner and size here

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')

    start = time.time()

    # ensure that all values are non-negative
    if r1 < 0:
        r1 = 0

    if c1 < 0:
        c1 = 0

    if h < 1:
        h = 0

    if w < 1:
        w = 0

    if h == 0 or w == 0:  # By default, use the whole image
        r1, c1 = 0, 0  # Start at the upper-left corner
        r2, c2 = r1 + imgH - 0, c1 + imgW - 0  # Include the whole image
        # r2, c2 = r1 + imgH - 1, c1 + imgW - 1  # Include the whole image
    else:
        r2, c2 = r1 + h - 0, c1 + w - 0  # Include the specified region of the image
        # r2, c2 = r1 + h - 1, c1 + w - 1  # Include the specified region of the image

    # Check to be sure we aren't exceeding the image dimensions:
    if r2 >= imgH:  # past bottom of image; set to limit
        r2 = imgH - 0

    if c2 >= imgW:  # past right edge of image; set to limit
        c2 = imgW - 0

    # Calculate the actual height and width after cleanup:
    h,w = r2-r1, c2-c1

    if verbose:
        print(f'[{time.time() - start:0.6f} s to check/correct limits')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return r1,c1, r2,c2, h,w

# ###################################################################################################
#
def recode_rggb_to_rg1g2b(rawColors, RGrGbB=(0, 1, 2, 3), verbose=False):  # default number code for colors
#   recode_g_as_gr_and_gb(rawColors, RGrBGb=(0, 1, 2, 3), verbose=False):  # default number code for colors

    #  Jeff B. Pelz May 2020

    #  Takes a rawImg.raw_colors array as input. If it is a 3-color (RGB) image instead of
    #  a 4 color (RG1BG2) image, it renames the two G colors Gr and Gb to match
    #  the color of the other filter on the same row.
    #  For example, the filter array | R G R G | is renamed to  | R  Gr  R  Gr | (G on red row)
    #                                | G B G B |                | Gb B   Gb B  | (G on blue row)

    #  input
    #         rawColors  rawImg.raw_colors array
    #         [default]  RGrBGb list of ints default=[0,1,2,3] defining number code for R, Rg, B, & Gb
    #         verbose=False boolean

    #  output
    #         rawColors  rawimg.raw_colors array  rewritten array, with G in B rows renamed as Gb

    # local color names from default number codes:

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')
        print(f'Recoding the two green channels as separate: '
                              f'Recoding to output order: RGrGbB = {RGrGbB}\n')

    Red, Gred, Gblue, Blue = RGrGbB[0], RGrGbB[1], RGrGbB[2], RGrGbB[3]

    Green = Gred  # A single green (G1 == G2) is coded the same as Gred

    origBlue = 2


    if verbose:
        print(f'\nrawColors before recoding: \n{rawColors[0:2, 0:2]}')

    if verbose==2:
        print(f'RGrGbB = {RGrGbB}   '
              f'Green (orig)  = {Green}   '
              f'Blue (orig)  = {origBlue} \n')
        print(f'Red    = {Red}')
        print(f'G_red  = {Gred}')
        print(f'G_blue = {Gblue}')
        print(f'Blue   = {Blue}')
        print(f'')

    for idx, row in enumerate(rawColors):  # Step through each row in raw color
        if origBlue in row:  # If this is a "Blue Row"
            row[row == origBlue] = Blue  # Rename each B pixel in this row to the new B
            row[row == Green] = Gblue  # Rename each G pixel in this row to a Gb pixel
            rawColors[idx, :] = row  # copy the renamed row back into the image

    if verbose:
        print(f'\nrawColors after recoding: \n{rawColors[0:2, 0:2]}')

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return rawColors


# ###################################################################################################
# Extract the specified color channel from a raw image
def extract_1_color_channel(rawImg, rawCFA, bayerChannel, verbose=False):
    # # Extract visible portion of the raw image:
    # rawImgVisible = rawImgObj.raw_value_visible

    if verbose:
        inFunc, calledBy = in_func_called_by()
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Entering {inFunc}(), called from {calledBy}().')
        # print(f'Extracting channel {bayerChannel} from: rawImg.shape = {rawImg.shape}. '
        #                 f'rawCFA = \n{rawCFA}')
        print(f'Extracting channel {bayerChannel} from: rawImg.shape = {rawImg.shape}. ')

    # r,c size of visible image: We will have to reshape after masking
    rawImgRows, rawImgCols = rawImg.shape[0:2]  # Should only be two dimensions
    if verbose:
        print(f'rawImgRows, rawImgCols = {rawImgRows, rawImgCols} ')

    # Set r,c size of channel image, which is 1/2 raw image in each dimension
    chanImgRows, chanImgCols = rawImgRows // 2, rawImgCols // 2
    if verbose:
        print(f'chanImgRows, chanImgCols = {chanImgRows, chanImgCols} ')

    # Make a channel mask == 1 at the pixels we want to extract:
    # Create a boolean mask which is True for the target channel, False elsewhere
    chanMask = (rawCFA == bayerChannel)  # rawImg.raw_colors==colorChan
    if verbose:
        print(f'np.count_nonzero(chanMask) =  {np.count_nonzero(chanMask)} '
              f'of {chanMask.size}')

    # Extract the channel image; defined where the chanMask is non-zero
    chanImg1D = rawImg[chanMask]  # Extract the pixels where chanMask==True, returns 1D array
    if verbose:
        print(f'chanImg1D.shape =  {chanImg1D.shape} ')

    # Reshape the masked pixels from 1D to 2D
    chanImg = chanImg1D.reshape(chanImgRows, chanImgCols)  # reshape 1D -> 2D

    if verbose:
        inFunc, calledBy = in_func_called_by()  # Call again because stack is affected by intermediate calls
        print(f'\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Returning to {calledBy}() from {inFunc}().')

    return chanImg

# ###################################################################################################

# Extract all color channels from a raw image
def extract_4_color_channels(rawImg, rawCFA, verbose=False):
    bayerChannels = []  # Initialized list

    for bayerChan in [0, 1, 2, 3]:  # Step through each channel and extract that chanImg
        bayerChannels.append(extract_1_color_channel(rawImg, rawCFA, bayerChan, verbose=verbose))

    if verbose:
        print(f'Extracted {len(bayerChannels)} channel images with sizes: '
              f'{bayerChannels[0].shape}, {bayerChannels[1].shape}, {bayerChannels[2].shape}, {bayerChannels[3].shape}')

    return bayerChannels





