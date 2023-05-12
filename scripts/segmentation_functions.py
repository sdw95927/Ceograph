#Functions for nuclei segmentation in Kaggle 2018

import numpy                    as np
import matplotlib.image         as mpimg
import matplotlib.pyplot        as plt
from sklearn                    import preprocessing
import scipy.misc
import cv2
import skimage
from skimage                    import measure
from skimage                    import img_as_bool, io, color, morphology, segmentation
from skimage.morphology         import binary_closing, binary_opening, disk
from PIL                        import Image

import time
import re
import sys
import os
import openslide
from openslide                  import open_slide, ImageSlide
import matplotlib.pyplot        as plt

import pandas                   as pd
import xml.etree.ElementTree    as ET
from skimage.draw import polygon
import random

#####################################################################
#Functions for color deconvolution
#####################################################################
def normalize(mat):
    """Do min-max normalization for input matrix of any dimension."""
    mat_normalized = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    return mat_normalized

def convert_to_optical_densities(img_RGB, r0=255, g0=255, b0=255):
    """Conver RGB image to optical densities with same shape as input image."""
    OD = img_RGB.astype(float)
    OD[:,:,0] /= r0
    OD[:,:,1] /= g0
    OD[:,:,2] /= b0
    return -np.log(OD+0.00001)
    
def channel_deconvolution (img_RGB, staining_type, plot_image=False):
    """Deconvolute RGB image into different staining channels.
    
    Args:
        img_RGB: A uint8 numpy array with RGB channels.
        staining_type: Dyes used to stain the image; choose one from ("HDB", "HRB", "HDR", "HEB").
        plot_image: Set True if want to real-time display results. Default is False.

    Returns:
        An unnormlized h*w*3 deconvoluted matrix and 3 different channels normalized to [0, 1] seperately.
        
    Raises:
        Exception: An error occured if staining_type is not defined. 
    """ 
    if staining_type == "HDB":
        channels = ("Hematoxylin", "DAB", "Background")
        stain_OD = np.asarray([[0.650,0.704,0.286],[0.268,0.570,0.776],[0.754,0.077,0.652]])
    elif staining_type == "HRB":
        channels = ("Hematoxylin", "Red", "Background")
        stain_OD = np.asarray([[0.650,0.704,0.286],[0.214,0.851,0.478],[0.754,0.077,0.652]])
    elif staining_type == "HDR":
        channels = ("Hematoxylin", "DAB", "Red")
        stain_OD = np.asarray([[0.650,0.704,0.286],[0.268,0.570,0.776],[0.214,0.851,0.478]])
    elif staining_type == "HEB":
        channels = ("Hematoxylin", "Eosin", "Background")
        stain_OD = np.asarray([[0.550,0.758,0.351],[0.398,0.634,0.600],[0.754,0.077,0.652]])
    else:
        raise Exception("Staining type not defined. Choose one from the following: HDB, HRB, HDR, HEB.")
    
    # Stain absorbance matrix normalization
    normalized_stain_OD = []
    for r in stain_OD:
        normalized_stain_OD.append(r/np.linalg.norm(r))
    normalized_stain_OD = np.asarray(normalized_stain_OD)
    stain_OD_inverse = np.linalg.inv(normalized_stain_OD)
    
    # Calculate optical density of input image
    OD = convert_to_optical_densities(img_RGB, 255, 255, 255)

    # Deconvolution
    img_deconvoluted = np.reshape(np.dot(np.reshape(OD, (-1, 3)), stain_OD_inverse), OD.shape)
    
    # Define each channel
    channel1 = normalize(img_deconvoluted[:, :, 0]) # First dye
    channel2 = normalize(img_deconvoluted[:, :, 1]) # Second dye
    channel3 = normalize(img_deconvoluted[:, :, 2]) # Third dye or background
    
    if plot_image: 
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()
        ax[0].imshow(img_RGB)
        ax[0].set_title("Original image")
        ax[1].imshow(channel1, cmap="gray")
        ax[1].set_title(channels[0])
        ax[2].imshow(channel2, cmap="gray")
        ax[2].set_title(channels[1])
        ax[3].imshow(channel3, cmap="gray")
        ax[3].set_title(channels[2])        
        plt.show()

    return img_deconvoluted, channel1, channel2, channel3

##################################################################
#Functions for morphological operations
##################################################################
def make_8UC(mat, normalized=True):
    """Convert the matrix to the equivalent matrix of the unsigned 8 bit integer datatype."""
    if normalized:
        mat_uint8 = np.array(mat.copy()*255, dtype=np.uint8)
    else:
        mat_uint8 = np.array(normalize(mat)*255, dtype=np.uint8)
    return mat_uint8

def make_8UC3(mat, normalized=True):
    """Convert the matrix to the equivalent matrix of the unsigned 8 bit integer datatype with 3 channels."""
    mat_uint8 = make_8UC(mat, normalized)
    mat_uint8 = np.stack((mat_uint8,)*3, axis = -1)
    return mat_uint8

def check_channel(channel):
    """Check whether there is any signals in a channel (yes: 1; no: 0)."""
    channel_uint8 = make_8UC(normalize(channel))
    if np.var(channel_uint8) < 0.02:
        return 0
    else:
        return 1

def fill_holes(img_bw):
    """Fill holes in input 0/255 matrix; equivalent of MATLAB's imfill(BW, 'holes')."""
    height, width = img_bw.shape
    
    # Needs to be 2 pixels larger than image sent to cv2.floodFill
    mask = np.zeros((height + 4, width + 4), np.uint8)
    
    # Add one pixel of padding all around so that objects touching border aren't filled against border
    img_bw_copy = np.zeros((height + 2, width + 2), np.uint8)
    img_bw_copy[1:(height + 1), 1:(width + 1)] = img_bw
    cv2.floodFill(img_bw_copy, mask, (0, 0), 255)
    img_bw = img_bw | (255 - img_bw_copy[1:(height+1), 1:(width+1)])
    return img_bw

def otsu_thresholding(img, thresh=None, plot_image=False, fill_hole=False):
    """Do image thresholding.
    
    Args:
        img: A uint8 matrix for thresholding.
        thresh: If provided, do binary thresholding use this threshold. If not, do default Otsu thresholding.
        plot_image: Set Ture if want to real-time display results. Default is False.
        fill_hole: Set True if want to fill holes in the generated mask. Default is False.
        
    Returns:
        A 0/255 mask matrix same size as img; object: 255; backgroung: 0.
    """
    if thresh is None:
        # Perform Otsu thresholding
        thresh, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Manually set threshold
        thresh, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    
    mask = skimage.morphology.remove_small_objects(mask, 2)
    
    # Fill holes
    if fill_hole:
        mask = fill_holes(mask)
    
    if plot_image:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title("Original")
        plt.figure()
        plt.imshow(mask)
        plt.title("After Thresholding")
        plt.colorbar()
        plt.show()
            
    return mask

def watershed(mask, img, plot_image=False, kernel_size=2):
    """Do watershed segmentation for input mask and image.
    
    Args:
        mask: A 0/255 matrix with 255 indicating objects.
        img: An 8UC3 matrix for watershed segmentation.
        plot_image: Set True if want to real-time display results. Default is False.
        kernel_size: Kernal size for inner marker erosion. Default is 2.
        
    Returns:
        A uint8 mask same size as input image, with -1 indicating boundary, 1 indicating background, 
        and numbers>1 indicating objects.
    """
    img_copy = img.copy()
    mask_copy = np.array(mask.copy(), dtype=np.uint8)

    # Sure foreground area (inner marker)
    mask_closed = closing(np.array(mask_copy, dtype=np.uint8))
    mask_closed = closing(np.array(mask_closed, dtype=np.uint8))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_fg = cv2.erode(mask_closed, kernel, iterations=2)
    sure_fg = skimage.morphology.closing(np.array(sure_fg, dtype=np.uint8))
    
    # Sure background area (outer marker)
    sure_fg_bool = 1 - img_as_bool(sure_fg)
    sure_bg = np.uint8(1 - morphology.skeletonize(sure_fg_bool))

    # Unknown region (the region other than inner or outer marker)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker for cv2.watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Set background to 1
    markers[unknown==1] = 0
    
    # Watershed
    # TODO(shidan.wang@utsouthwestern.edu): Replace cv2.watershed with skimage.morphology.watershed
    marker = cv2.watershed(img_copy, markers.copy())

    if plot_image:
        plt.figure()
        plt.imshow(sure_fg)
        plt.title("Inner Marker")
        plt.figure()
        plt.imshow(sure_bg)
        plt.title("Outer Marker")
        plt.figure()
        plt.imshow(unknown)
        plt.title("Unknown")
        plt.figure()
        plt.imshow(markers, cmap='jet')
        plt.title("Markers")
        plt.figure()
        plt.imshow(marker, cmap='jet')
        plt.title("Mask")
        plt.figure()
        plt.imshow(img)
        plt.title("Original Image")
        plt.figure()
        img_copy[marker == -1] = [0, 255 ,0]
        plt.imshow(img_copy)
        plt.title("Marked Image")
        plt.show()

    return marker

def generate_mask(channel, original_img=None, overlap_color=(0, 1, 0), 
                  plot_process=False, plot_result=False, title="",
                  fill_hole=False, thresh=None,
                  use_watershed=True, watershed_kernel_size=2,
                  save_img=False, save_path=None):
    """Generate mask for a gray-value image.
    
    Args:
        channel: Channel returned by function 'channel_deconvolution'. A gray-value image is also accepted.
        original_img: A image used for plotting overlapped segmentation result, optional.
        overlap_color: A 3-value tuple setting the color used to mark segmentation boundaries on original
            image. Default is green (0, 1, 0).
        plot_process: Set True if want to display the whole mask generation process. Default is False.
        plot_result: Set True if want to display the final result. Default is False.
        title: The title used for plot_result, optional.
        fill_hole: Set True if want to fill mask holes. Default is False.
        thresh: Provide this value to do binary thresholding instead of default otsu thresholding.
        use_watershed: Set False if want to skip the watershed segmentation step. Default is True.
        watershed_kernel_size: Kernel size of inner marker erosion. Default is 2.
        save_img: Set True if want to save the mask image. Default is False.
        save_path: The path to save the mask image, optional. Prefer *.png or *.pdf.
        
    Returns: 
        A binary mask with 1 indicating an object and 0 indicating background.
        
    Raises:
        IOError: An error occured writing image to save_path.
    """  
    if not check_channel(channel):
        # If there is not any signal
        print("No signals detected for this channel")
        return np.zeros(channel.shape)
    else:
        channel = normalize(channel)
        if use_watershed:
            mask_threshold = otsu_thresholding(make_8UC(channel),
                                               plot_image=plot_process, fill_hole=fill_hole, thresh=thresh)
            marker = watershed(mask_threshold, make_8UC3(channel),
                               plot_image = plot_process, kernel_size = watershed_kernel_size)     
            # create mask
            mask = np.zeros(marker.shape)
            mask[marker == 1] = 1
            mask = 1 - mask
            # Set boundary as mask from Otsu_thresholding, since cv2.watershed automatically set boundary as -1
            mask[0, :] = mask_threshold[0, :] == 255
            mask[-1, :] = mask_threshold[-1, :] == 255
            mask[:, 0] = mask_threshold[:, 0] == 255
            mask[:, -1] = mask_threshold[:, -1] == 255
        else:
            mask = otsu_thresholding(make8UC(channel), 
                                     plot_image=plot_process, fill_hole=fill_hole, thresh=thresh)
        
        if plot_result or save_img:
            if original_img is None:
                # If original image is not provided, plot mask only
                plt.figure()
                plt.imshow(mask, cmap = "gray")
            else:
                # If original image is provided
                overlapped_img = segmentation.mark_boundaries(original_img, skimage.measure.label(mask), 
                                                             overlap_color, mode = "thick")
                fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True,
                                     subplot_kw={'adjustable': 'box-forced'})
                ax = axes.ravel()
                ax[0].imshow(mask, cmap="gray")
                ax[0].set_title(str(title)+" Mask")
                ax[1].imshow(overlapped_img)
                ax[1].set_title("Overlapped with Original Image")
            if save_img:
                try: 
                    plt.savefig(save_path)
                except:
                    raise IOError("Error saving image to {}".format(save_path))
            if plot_result:
                plt.show()
            plt.close()
    return mask

def get_mask_for_slide_image(filePath, display_progress=False):
    """Generate mask for slide"""
    slide = open_slide(filePath)
    
    # Use the lowest resolution
    level_dims = slide.level_dimensions
    level_to_analyze = len(level_dims)-1
    dims_of_selected = level_dims[-1]

    if display_progress:
        print('Selected image of size (' + str(dims_of_selected[0]) + ', ' + str(dims_of_selected[1]) + ')')
    slide_image = slide.read_region((0, 0), level_to_analyze, dims_of_selected)
    slide_image = np.array(slide_image)
    if display_progress:
        plt.figure()
        plt.imshow(slide_image)
        
    # Perform Otsu thresholding
    threshR, maskR = cv2.threshold(slide_image[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshG, maskG = cv2.threshold(slide_image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshB, maskB = cv2.threshold(slide_image[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Add the channels together
    mask = ((255-maskR) | (255-maskG) | (255-maskB))
    if display_progress:
        plt.figure()
        plt.imshow(mask)

    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    if display_progress:
        plt.figure()
        plt.imshow(mask)
        plt.show()
    
    # Delete small objects
    min_pixel_count = 0.005 * dims_of_selected[0] * dims_of_selected[1]
    mask = np.array(skimage.morphology.remove_small_objects(np.array(mask/255, dtype=bool), min_pixel_count),
                    dtype=np.uint8)
    if display_progress:
        print("Min pixel count: {}".format(min_pixel_count))
        plt.figure()
        plt.imshow(mask)
        plt.show()
        
    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Fill holes
    mask = fill_holes(mask)
    if display_progress:
        plt.figure()
        plt.imshow(mask)
        plt.show()
        
    return mask, slide_image
    
##################################################################
#Functions for extracting patches from slide image
##################################################################

def extract_patch_by_location(filepath, location, patch_size=(500, 500), 
                           plot_image=False, level_to_analyze=0, save=False, savepath='.'):
    """
    Args:
        location: Absolute location regardless of "level_to_analyze"
    
    Output: A PIL Image object. Need numpy to transfer to np.array.
    """
    if not os.path.isfile(filepath):
        raise IOError("Image not found!")
        return []
    
    slide = open_slide(filepath)
    filename = re.search("(?<=/)[^/]+\.svs", filepath).group(0)[0:-4]
    slide_image = slide.read_region(location, level_to_analyze, patch_size)
    if plot_image:
        plt.figure()
        plt.imshow(slide_image)
        plt.show()
        
    if save:
        savename = os.path.join(savepath, str(filename)+'_'+str(location[0])+'_'+str(location[1])+'.png')
        misc.imsave(savename, slide_image)
        print("Writed to "+savename)
    return slide_image
    
def extract_patch_by_tissue_area(filePath, nPatch=0, maxPatch=10, filename=None, savePath=None, displayProgress=False):
    '''Input: slide
       Output: image patches'''
    if filename is None:
        filename = re.search("(?<=/)[0-9]+\.svs", filePath).group(0)
    if savePath is None:
        savePath = '/home/swan15/python/brainTumor/sample_patches/'
    bwMask, slideImageCV = getMaskForSlideImage(filePath, displayProgress=displayProgress)
    slide = open_slide(filePath)
    levelDims = slide.level_dimensions
    #find magnitude
    for i in range(0, len(levelDims)):
        if bwMask.shape[0] == levelDims[i][1]:
            magnitude = levelDims[0][1]/levelDims[i][1]
            break
    nCol = int(math.ceil(levelDims[0][1]/patchSize))
    nRow = int(math.ceil(levelDims[0][0]/patchSize))
    #get contour
    _, contours, _ = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for nContours in range(0, len(contours)):
        print(nContours)
        # i is the y axis in the image
        for i in range(0, nRow):
            minRow = i*patchSize/magnitude
            maxRow = (i+1)*patchSize/magnitude
            matches = [x for x in range(0, len(contours[nContours][:, 0, 0]))
                       if (contours[nContours][x, 0, 1] > minRow and contours[nContours][x, 0, 1] < maxRow)]
            try:
                print [min(contours[nContours][matches, 0, 0]), max(contours[nContours][matches, 0, 0])]
                
                #save image
                minCol = min(contours[nContours][matches, 0, 0])*magnitude
                maxCol = max(contours[nContours][matches, 0, 0])*magnitude
                minColInt = int(math.floor(minCol/patchSize))
                maxColInt = int(math.ceil(maxCol/patchSize))
                
                for j in range(minColInt, maxColInt):
                    startCol = j*patchSize
                    startRow = i*patchSize
                    patch = slide.read_region((startCol, startRow), desiredLevel, (patchSize, patchSize))
                    patchCV = np.array(patch)
                    patchCV = patchCV[:, :, 0:3]
                    
                    fname = os.path.join(savePath, filename+'_'+str(i)+'_'+str(j)+'.png')
                    
                    if not os.path.isfile(fname):
                        misc.imsave(fname, patchCV)
                        nPatch = nPatch + 1
                        print(nPatch)
                    
                    if nPatch >= maxPatch:
                        break
            except ValueError:
                continue      
            if nPatch >= maxPatch:
                break   
        if nPatch >= maxPatch:
            break
            
def parseXML(xmlFile, pattern):
    """
    Parse XML File and returns an object containing all the vertices 
    Verticies: (dict)
         pattern: (list) of dicts, each with 'X' and 'Y' key 
                [{ 'X': [1,2,3], 
                   'Y': [1,2,3]  }]
    """
    
    tree = ET.parse(xmlFile) # Convert XML file into tree representation
    root = tree.getroot()

    regions = root.iter('Region') # Extract all Regions
    vertices = {pattern: []} # Store all vertices in a dictionary

    for region in regions: 
        label = region.get('Text') # label either as 'ROI' or 'normal'
        if label == pattern:
            vertices[label].append({'X':[], 'Y':[]})

            for vertex in region.iter('Vertex'): 
                X = float(vertex.get('X'))
                Y = float(vertex.get('Y'))

                vertices[label][-1]['X'].append(X)
                vertices[label][-1]['Y'].append(Y)

    return vertices

def calculateRatio(levelDims):
    """ Calculates the ratio between the highest resolution image and lowest resolution image.
    Returns the ratio as a tuple (Xratio, Yratio). 
    """
    highestReso = np.asarray(levelDims[0])
    lowestReso = np.asarray(levelDims[-1])
    Xratio, Yratio = highestReso/lowestReso
    return (Xratio, Yratio)

def createMask(levelDims, vertices, pattern):
    """
    Input: levelDims (nested list): dimensions of each layer of the slide.
           vertices (dict object as describe above)
    Output: (tuple) mask
            numpy nd array of 0/1, where 1 indicates inside the region
            and 0 is outside the region
    """
    # Down scale the XML region to create a low reso image mask, and then 
    # rescale the image to retain reso of image mask to save memory and time 
    Xratio, Yratio = calculateRatio(levelDims)

    nRows, nCols = levelDims[-1]
    mask = np.zeros((nRows, nCols), dtype=np.uint8)

    for i in range(len(vertices[pattern])):
        lowX = np.array(vertices[pattern][i]['X'])/Xratio
        lowY = np.array(vertices[pattern][i]['Y'])/Yratio
        rr, cc = polygon(lowX, lowY, (nRows, nCols))
        mask[rr, cc] = 1

    return mask

def getMask(xmlFile, svsFile, pattern):
    """ Parses XML File to get mask vertices and returns matrix masks 
    where 1 indicates the pixel is inside the mask, and 0 indicates outside the mask.

    @param: {string} xmlFile: name of xml file that contains annotation vertices outlining the mask. 
    @param: {string} svsFile: name of svs file that contains the slide image.
    @param: {pattern} string: name of the xml labeling
    Returns: slide - openslide slide Object 
             mask - matrix mask of pattern
    """
    vertices = parseXML(xmlFile, pattern) # Parse XML to get vertices of mask
    
    if not len(vertices[pattern]):
        slide = 0
        mask = 0
        return slide, mask

    slide = open_slide(svsFile)
    levelDims = slide.level_dimensions
    mask = createMask(levelDims, vertices, pattern)

    return slide, mask

def plotMask(mask):
    fig, ax1 = plt.subplots(nrows=1, figsize=(6,10))
    ax1.imshow(mask)
    plt.show()

def chooseRandPixel(mask):
    """ Returns [x,y] numpy array of random pixel.
    @param {numpy matrix} mask from which to choose random pixel.
    """
    array = np.transpose(np.nonzero(mask)) # Get the indices of nonzero elements of mask.
    index = random.randint(0,len(array)-1) # Select a random index
    return array[index]

def plotImage(image):
    plt.imshow(image)
    plt.show()
    
def checkWhiteSlide(image):
    im = np.array(image.convert(mode='RGB'))
    pixels = np.ravel(im)
    mean = np.mean(pixels)
    return mean >= 230

# extractPatchByXMLLabeling
def getPatches(slide, mask, numPatches=0, dims=(0,0), dirPath='', slideNum='', plot=False, plotMask=False):
    """ Generates and saves 'numPatches' patches with dimension 'dims' from image 'slide' contained within 'mask'.
    @param {Openslide Slide obj} slide: image object
    @param {numpy matrix} mask: where 0 is outside region of interest and 1 indicates within 
    @param {int} numPatches
    @param {tuple} dims: (w,h) dimensions of patches
    @param {string} dirPath: directory in which to save patches
    @param {string} slideNum: slide number 
    Saves patches in directory specified by dirPath as [slideNum]_[patchNum]_[Xpixel]x[Ypixel].png
    """ 
    w,h = dims 
    levelDims = slide.level_dimensions
    Xratio, Yratio = calculateRatio(levelDims)

    i = 0
    while i < numPatches:
        firstLoop = True # Boolean to ensure while loop runs at least once. 

        while firstLoop: # or not mask[rr,cc].all(): # True if it is the first loop or if all pixels are in the mask 
            firstLoop = False
            x, y = chooseRandPixel(mask) # Get random top left pixel of patch. 
            xVertices = np.array([x, x+(w/Xratio), x+(w/Xratio), x, x])
            yVertices = np.array([y, y, y-(h/Yratio), y-(h/Yratio), y])
            rr, cc = polygon(xVertices, yVertices)

        image = slide.read_region((x*Xratio, y*Yratio), 0, (w,h))
        
        isWhite = checkWhiteSlide(image)
        newPath = 'other' if isWhite else dirPath
        if not isWhite: i += 1

        slideName = '_'.join([slideNum, 'x'.join([str(x*Xratio),str(y*Yratio)])])
        image.save(os.path.join(newPath, slideName+".png"))

        if plot: 
            plotImage(image)
        if plotMask: mask[rr,cc] = 0

    if plotMask:
        plotImage(mask)
        
'''Example codes for getting patches from labeled svs files:
#define the patterns
patterns = ['small_acinar', 
            'large_acinar',
            'tubular',
            'trabecular', 
            'aveolar', 
            'solid', 
            'pseudopapillary', 
            'rhabdoid',
            'sarcomatoid',
            'necrosis', 
            'normal', 
            'other']
#create folders
for pattern in patterns:
    if not os.path.exists(pattern):
        os.makedirs(pattern)
#define parameters
patchSize = 500
numPatches = 50
dirName = '/home/swan15/kidney/ccRCC/slides' 
annotatedSlides = 'slide_region_of_interests.txt'

f = open(annotatedSlides, 'r+')
slides = [re.search('.*(?=\.svs)', line).group(0) for line in f 
          if re.search('.*(?=\.svs)', line) is not None]
print slides
f.close()
for slideID in slides:
    print('Start '+slideID)
    try: 
        xmlFile = slideID+'.xml'
        svsFile = slideID+'.svs'

        xmlFile = os.path.join(dirName, xmlFile)
        svsFile = os.path.join(dirName, svsFile)
        
        if not os.path.isfile(xmlFile):
            print xmlFile+' not exist'
            continue
        
        for pattern in patterns:
            
            numPatchesGenerated = len([files for files in os.listdir(pattern)
                                      if re.search(slideID+'_.+\.png', files) is not None])
            if numPatchesGenerated >= numPatches:
                print(pattern+' existed')
                continue
            else:
                numPatchesTemp = numPatches - numPatchesGenerated
                
            slide, mask = getMask(xmlFile, svsFile, pattern)
            
            if not slide:
                #print(pattern+' not detected.')
                continue
            
            getPatches(slide, mask, numPatches = numPatchesTemp, dims = (patchSize, patchSize), 
                       dirPath = pattern+'/', slideNum = slideID, plotMask = False)  # Get Patches
            print(pattern+' done.')

        print('Done with ' + slideID)
        print('----------------------')

    except:
        print('Error with ' + slideID)
'''

##################################################################
# RGB color processing
##################################################################

# convert RGBA image to RGB (specifically designed for masks)
def convert_RGBA(RGBA_img):
    if np.shape(RGBA_img)[2] == 4:
        RGB_img = np.zeros((np.shape(RGBA_img)[0], np.shape(RGBA_img)[1], 3))
        RGB_img[RGBA_img[:, :, 3] == 0] = [255, 255, 255]
        RGB_img[RGBA_img[:, :, 3] == 255] = RGBA_img[RGBA_img[:, :, 3] == 255, 0:3]
        return RGB_img
    else:
        print("Not an RGBA image")
        return RGBA_img
        
# Convert RGB mask to one-channel mask
def RGB_to_index(RGB_img, RGB_markers=None, RGB_labels=None):
    """Change RGB to 2D index matrix; each RGB color corresponds to one index.
    
    Args:
        RGB_markers: start from background (marked as 0); 
            Example format:
                [[255, 255, 255],
                [160, 255, 0]]
        RGB_labels: a numeric vector corresponding to the labels of RGB_markers; 
            length should be the same as RGB_markers.
    """
    if np.shape(RGB_img)[2] is not 3:
        print("Not an RGB image")
        return RGB_img
    else:
        if RGB_markers == None:
            RGB_markers = [[255, 255, 255]]
        if RGB_labels == None:
            RGB_labels = range(np.shape(RGB_markers)[0])
        mask_index = np.zeros((np.shape(RGB_img)[0], np.shape(RGB_img)[1]))
        for i, RGB_marker in enumerate(RGB_markers):
            mask_index[np.all(RGB_img == RGB_marker, axis=2)] = RGB_labels[i]
    return mask_index
    
def index_to_RGB(mask_index, RGB_markers=None):
    """Change index to 2D image; each index corresponds to one color"""
    mask_index_copy = mask_index.copy()
    mask_index_copy = np.squeeze(mask_index_copy)  # In case the mask shape is not [height, width]
    if RGB_markers == None:
        print("RGB_markers not provided!")
        RGB_markers = [[255, 255, 255]]
    RGB_img = np.zeros((np.shape(mask_index_copy)[0], np.shape(mask_index_copy)[1], 3), dtype=np.uint8)
    RGB_img[:, :] = RGB_markers[0]  # Background
    for i in range(np.shape(RGB_markers)[0]):
        RGB_img[mask_index_copy == i] = RGB_markers[i]
    return RGB_img
    
def shift_HSV(img, amount=(0.9, 0.9, 0.9)):
    """Function to tune Hue, Saturation, and Value for image img"""
    img = Image.fromarray(img, 'RGB')
    hsv = img.convert('HSV')
    hsv = np.array(hsv)
    hsv[..., 0] = np.clip((hsv[..., 0] * amount[0]), a_max=255, a_min=0)
    hsv[..., 1] = np.clip((hsv[..., 1] * amount[1]), a_max=255, a_min=0)
    hsv[..., 2] = np.clip((hsv[..., 2] * amount[2]), a_max=255, a_min=0)
    new_img = Image.fromarray(hsv, 'HSV')
    return np.array(new_img.convert('RGB'))
    
def overlay_segmentation(patch, mask):
    """Overlay mask to patch.
    
    Args:
        patch: Original image;
        mask: Background as [255, 255, 255] with be transparent.
    """
    overlayed = patch.copy()
    overlayed[np.any(mask != [255, 255, 255], axis=2)] = mask[np.any(mask != [255, 255, 255], axis=2)]
    return overlayed