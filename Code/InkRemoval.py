# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 2021


# author          :     Laura Purcaro
# title           :     InkRemoval.py
# description     :     Ink removal of Whole Slide Images by color detection
# date            :     02.05.2021
# version         :     1.0
# python_version  :     3.8.8

"""

import pandas as pd
import cv2
import openslide as op
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re



# Get current working directory
current_path = os.path.abspath(os.getcwd())

# Add path into a list for each image
png_images = []
path = r'D:\Classification\Good_PNG'

for filename in os.listdir(path):
    if filename.endswith(".png"):
        png_images.append(filename)
        

def MatchbyColor(image,lower,upper):
    # Read image
    img = cv2.imread('D:/Classification/Good_PNG/'+str(image))
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # Lower and upper boundaries
    lg = np.array([lower])        
    ug = np.array([upper])
    
    # Segment image based on lower and upper boundaries
    gmask = cv2.inRange(rgb_img,lg ,ug)
    # Count colour pixels
    pixels = cv2.countNonZero(gmask)
    
    
    select_image = False
    # If at least there is one coloured pixels, set as True
    if pixels > 0:
        print("Image: {} | Green exist".format(image))
        select_image = True
        
    else: 
        print("Image: {} | Not found".format(image))
        select_image = False    
    
    return select_image

# List of colour ranges
green = ((80,120,140),(86,150,160))
blue = ((90,100,200), (195,216,250))
red_yellow = ((35,12,10), (100,100,100))
black = ((100,80,85), (152,150,158))

def selectImage(images, color):
    for i in range(0, len(images)):
        selected = MatchbyColor(images[i],color[0],color[1])
        if selected:
            bgr = cv2.imread(path+images[i])
            RGB = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            new_path = current_path + 'Check color'
            
            if not os.path.exists(new_path):
                os.makedirs(new_path, exist_ok=True)
                
            plt.imsave(r'Check color/selected_{}.png'.format(i), RGB)   
            

# Look for ink
selectImage(png_images, green)
selectImage(png_images, red_yellow)
selectImage(png_images, blue)
selectImage(png_images, black)

        
'''Remove ink'''

# Get Ink images path

ink_images = []

for filename in os.listdir(new_path):
    if filename.endswith(".png"):
        ink_path = os.path.join(new_path, filename)
        ink_images.append(filename)
        
print(ink_images)
        
        
def InkRemoval(images):
    for i in range(0, len(images)):
        image = cv2.imread('4 weeks project - ink/'+images[i])
        wsi_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        wsi_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # red mask
        lower_red = (0, 78, 78)
        upper_red = (124, 255, 187)
        mask_red = cv2.inRange(wsi_hsv, lower_red, upper_red)
        # black mask
        lower_black = (14, 16, 48) 
        upper_black = (179, 40, 204)
        mask_black = cv2.inRange(wsi_hsv, lower_black, upper_black)
        # green mask
        lower_green = (20, 22, 23) 
        upper_green = (124, 220, 251)
        mask_green = cv2.inRange(wsi_hsv, lower_green, upper_green)
        # yellow mask
        lower_yellow = (0, 42, 141)
        upper_yellow = (104, 201, 255)
        mask_yellow = cv2.inRange(wsi_hsv, lower_yellow, upper_yellow)
        # black 2
        lower_black2 = (0, 0, 0)
        upper_black2 = (179, 91, 161)
        mask_black2 = cv2.inRange(wsi_hsv, lower_black2, upper_black2)
        
        # Add together all 5 masks into final ink detection mask
        final_mask = mask_red + mask_black + mask_green + mask_yellow + mask_black2
        final_result = cv2.bitwise_and(wsi_rgb, wsi_rgb, mask=final_mask)
        
        # Create new folder
        mask_inkremoved = current_path + 'Ink removed Mask'
        if not os.path.exists(mask_inkremoved):
                os.makedirs(mask_inkremoved, exist_ok=True)
                
        plt.imsave(r'Ink removed Mask\mask_{}'.format(images[i]), final_result)
        
        # Copy original image and remove ink
        result = wsi_rgb.copy()
        result[final_mask!=0] = (255,255,255)
        
        # Create new folder
        inkremoved = current_path + 'Ink removed'
        if not os.path.exists(inkremoved):
                os.makedirs(inkremoved, exist_ok=True)
                
        plt.imsave(r'Ink removed\result_{}'.format(images[i]), result)
        
# Run InkRemoval function
InkRemoval(ink_images)
        
'''Evaluation of ground truth images vs. predicted'''

# Convert ground truth tif to png
annotation_folder = input(str('Please insert your Annotation folder path: '))
mrxs_folder = input(str('Please insert your WSI in mrxs format folder path: '))

for i in glob.glob(annotation_folder+'study n.*.tif'):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        m = int(m[i])
        
        # Open .tif mask
        path1 = 'r'+annotation_folder+'study n.*.tif'.format(m)
        wsi_tif = op.OpenSlide(path1)
        slide_level = 5
        
        # Open corresponding .mrxs slide number
        path2 = mrxs_folder + "study no.{}.mrxs".format(m)
        wsi = op.OpenSlide(path2)
        # Decide at what level we want to process the slides
        slide_level = 5
    
        # Read region in order to process the image with numpy
        # Starting point of the tissue scanned
        x = wsi.properties[op.PROPERTY_NAME_BOUNDS_X]
        y = wsi.properties[op.PROPERTY_NAME_BOUNDS_Y]
    
        # Dimension of the image is previous x and y
        dim = (int(int(x)),int(int(y)))

        # Maximum width and height of just tissue scanned dimension
        w,h = wsi.properties[op.PROPERTY_NAME_BOUNDS_WIDTH], wsi.properties[op.PROPERTY_NAME_BOUNDS_HEIGHT]
        wh = (int(int(w)/2**slide_level),int(int(h)/2**slide_level))
        
        img = wsi_tif.read_region(dim,slide_level,wh)
        img_wsi = np.array(img)
        RGB_image = cv2.cvtColor(img_wsi, cv2.COLOR_RGB2GRAY)

        # Create new folder
        converted_ann = current_path + 'Converted annotation'
        if not os.path.exists(converted_ann):
                os.makedirs(converted_ann, exist_ok=True)
        plt.imsave(r"Converted annotation\study n.{}.png".format(m), RGB_image, cmap="gray")
        
#######################################################        
# EVALUATION METRICS

# Dice score
 
''' Dice score prediction with ink '''
dice_scores_ink = []


for i in glob.glob("Annotation\\GT\\new\\result_study n.*.png"):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        m = int(m[i])        
        # Create list for dice scores

        '''DICE SCORE'''
        #load images
        # Open .png mask of predicted folds
        y_pred = cv2.imread(r'Annotation\Prediction ink\study n.{}.png'.format(m))
        # Open .png mask (previous .tif mask)
        y_true = cv2.imread(r'Annotation\GT\new\result_study n.{}.png'.format(m)) 

        # Dice similarity function
        def dice(pred, true, k = 1):
            intersection = np.sum(pred[true==k]) * 2.0
            dice = intersection / (np.sum(pred) + np.sum(true))
            return dice

        dice_score = dice(y_pred, y_true, k = 255) #255 in my case, can be 1 
        print ("Dice Similarity: {} for study no.{}".format(dice_score, str(m)))
        
        '''Append results on a list'''
        dice_scores_ink.append('Study n.'+ str(m) + ' :' + str(dice_score))
        


# Create Folds info.xlsx excel file

# Save output "Area of folds" into excel file
df = pd.DataFrame({"Study n.: Dice score ink": dice_scores_ink})

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(r'Results\Dice score ink_new.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and save the output
writer.close()
        
''' Dice score prediction without ink '''
dice_scores_noink = []


for i in glob.glob("Annotation\\GT\\new\\result_study n.*.png"):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        m = int(m[i])        
        # Create list for dice scores

        '''DICE SCORE'''
        #load images
        # Open .png mask of predicted folds
        y_pred = cv2.imread(r'Annotation\Prediction no ink\result_study n.{}.png'.format(m))
        # Open .png mask (previous .tif mask)
        y_true = cv2.imread(r'Annotation\GT\new\result_study n.{}.png'.format(m)) 

        # Dice similarity function
        def dice(pred, true, k = 1):
            intersection = np.sum(pred[true==k]) * 2.0
            dice = intersection / (np.sum(pred) + np.sum(true))
            return dice

        dice_score = dice(y_pred, y_true, k = 255) #255 in my case, can be 1 
        print ("Dice Similarity: {} for study no.{}".format(dice_score, str(m)))
        
        '''Append results on a list'''
        dice_scores_noink.append('Study n.'+ str(m) + ' :' + str(dice_score))
        
# Save output "Area of folds" into excel file
df = pd.DataFrame({"Study n.: Dice score": dice_scores_noink})

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(r'Results\Dice score no ink_new.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and save the output
writer.close()

####################################################
# IoU score

  '''IoU score with ink '''

slides_number = []
iou_scores_ink = []
for i in glob.glob("Annotation\\GT\\new\\result_study n.*.png"):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        m = int(m[i])

        prediction = cv2.imread(r'Annotation\Prediction ink\study n.{}.png'.format(m),0)
        # Open .png mask (previous .tif mask)
        target = cv2.imread(r'Annotation\GT\new\\result_study n.{}.png'.format(m),0) 

        # Iou score function
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)

        
        '''Append results on a list'''
        iou_scores_ink.append('Study n.'+ str(m) + ' :' + str(iou_score))

        print('Study no.'+str(m)+': '+ str(iou_score))
        
# Save output "Area of folds" into excel file
df = pd.DataFrame({"Study n.: IoU score ink": iou_scores_ink})

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(r'Results\\IoU score ink_new.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and save the output
writer.close()


''' IoU score with no ink '''
slides_number = []
iou_scores_no_ink = []
for i in glob.glob("Annotation\\GT\\new\\result_study n.*.png"):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        m = int(m[i])

        prediction = cv2.imread(r'Annotation\Prediction no ink\result_study n.{}.png'.format(m),0)
        # Open .png mask (previous .tif mask)
        target = cv2.imread(r'Annotation\GT\new\\result_study n.{}.png'.format(m),0) 

        # Iou score function
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)

        
        '''Append results on a list'''
        iou_scores_no_ink.append('Study n.'+ str(m) + ' :' + str(iou_score))

        print('Study no.'+str(m)+': '+ str(iou_score))

# Save output "Area of folds" into excel file
df = pd.DataFrame({"Study n.: IoU score NO ink": iou_scores_no_ink})

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(r'Results\\IoU score no ink_new.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and save the output
writer.close()


''' Localization metric '''

def LocalizationMetric(slide_n,img1,img2):
    # img1 = ground truth
    # img2 = prediction
    rng.seed(12345)

    blur1 = cv2.blur(img1, (3,3))
    blur2 = cv2.blur(img2, (3,3))

    ret, thresh = cv2.threshold(blur1, 127, 255,0)
    ret, thresh2 = cv2.threshold(blur2, 127, 255,0)

    kernel = np.ones((5,5),np.uint8)
    dilation1 = cv2.dilate(img1,kernel,iterations = 1)

    dilation2 = cv2.dilate(img2,kernel,iterations = 1)

    contours1,hierarchy1 = cv2.findContours(dilation1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours2,hierarchy2 = cv2.findContours(dilation2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Approximate contours to polygons + get bounding rects
    contours_poly1 = [None]*len(contours1)
    boundRect1 = [None]*len(contours1)

    for i, c in enumerate(contours1):
        contours_poly1[i] = cv2.approxPolyDP(c, 3, True)
        boundRect1[i] = cv2.boundingRect(contours_poly1[i])
        #centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])


        drawing1 = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)


    for i in range(len(contours1)):
        color1 = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing1, contours_poly1, i, color1)
        cv2.rectangle(drawing1, (int(boundRect1[i][0]), int(boundRect1[i][1])), \
              (int(boundRect1[i][0]+boundRect1[i][2]), int(boundRect1[i][1]+boundRect1[i][3])), color1, 2)

    #####################################################################
    contours_poly2 = [None]*len(contours2)
    boundRect2 = [None]*len(contours2)

    for i, c in enumerate(contours2):
        contours_poly2[i] = cv2.approxPolyDP(c, 3, True)
        boundRect2[i] = cv2.boundingRect(contours_poly2[i])
        #centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])


        drawing2 = np.zeros((thresh2.shape[0], thresh2.shape[1], 3), dtype=np.uint8)
    ######################################################################
    gt_match = []
    summ_area_match = 0
    summ_area_not_match = 0
    tot_folds_area = 0
    # Get the moments
    mu = [None]*len(contours2)
    # Get the mass centers
    mc = [None]*len(contours2)
    
    blank_image = np.zeros((img2.shape),np.uint8)
    image_area = np.prod(img2.shape)


    for i in range(len(contours2)):
        color2 = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing2, contours_poly2, i, color2)
        # my coordinates x,y and w,h
        cv2.rectangle(drawing2, (int(boundRect2[i][0]), int(boundRect2[i][1])), \
              (int(boundRect2[i][0]+boundRect2[i][2]), int(boundRect2[i][1]+boundRect2[i][3])), color2, 2)
        
        for i in range(len(contours2)):
            mu[i] = cv2.moments(contours2[i])

        # add 1e-5 to avoid division by zero
        # MASS CENTER of folds objects, not rectangle
        for i in range(len(contours2)):
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            # Drawing the center
            cv2.circle(drawing2, (int(mc[i][0]), int(mc[i][1])), 4, color2, -1)
    
    for i in range(len(contours2)):
        x = mc[i][0]
        y = mc[i][1]

        for j in range(len(contours1)):
            x1 = int(boundRect1[j][0])
            y1 = int(boundRect1[j][1])
            x2 = int(boundRect1[j][0]+boundRect1[j][2])
            y2 = int(boundRect1[j][1]+boundRect1[j][3])

            if (x1 <= x and x <= x2):
                if (y1 <= y and y <= y2):
                    #print("N. {} - Contour gt {}: correspond to prediction contour {}".format(slide_n, j, i), True)
                    gt_match.append(j)
                    area2 = cv2.contourArea(contours2[i])
                    # It count the number of objects that are non zero
                    #area2 = cv2.countNonZero(area)
                    summ_area_match += area2
                else:
                    
                    area3 = cv2.contourArea(contours2[i])
                    #area4 = cv2.countNonZero(area3)
                    summ_area_not_match += area3
                #print(True)
            pass
        
    #print("Slide n.", slide_n)   
    
    
    unique = list(set(gt_match))
    # Avoid division by 0
    try:
        percentage = (len(unique)/len(contours1))*100
    except ZeroDivisionError:
        percentage = 0
    
    return slide_n, percentage


''' Localization metric with ink '''

loc_score_ink = []
for i in glob.glob('Annotation\\GT\\new\\result_study n.*.png'):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        slide_n = int(m[i])
        img1 = cv2.imread(r"Annotation\GT\new\result_study n.{}.png".format(slide_n),0)
        img2 = cv2.imread(r"Annotation\Prediction ink\study n.{}.png".format(slide_n),0)
        a, b = LocalizationMetric(slide_n,img1,img2)
        
        loc_score_ink.append('Study n.'+str(a)+';'+str(b)+'%')
        print('Study n.' + str(a) + ' : ' + str(b))


# Save output "Area of folds" into excel file
df = pd.DataFrame({"Study n.; Localization score ink": loc_score_ink})

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(r'Results\Localization score ink_new.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and save the output
writer.close()


''' Localization score no ink '''
loc_score_no_ink = []
for i in glob.glob('Annotation\\GT\\new\\result_study n.*.png'):
    # Matching just the number of slide present in the folder JustFolds
    m = re.findall("[0-9]+", i)
    for i in range(0, len(m)): 
        slide_n = int(m[i])
        img1 = cv2.imread(r"Annotation\GT\new\\result_study n.{}.png".format(slide_n),0)
        img2 = cv2.imread(r"Annotation\Prediction no ink\result_study n.{}.png".format(slide_n),0)
        a, b = LocalizationMetric(slide_n,img1,img2)
        
        loc_score_no_ink.append('Study n.'+str(a)+';'+str(b)+'%')
        
        print('Study n.' + str(a) + ' : ' + str(b))


# Save output "Area of folds" into excel file
df = pd.DataFrame({"Study n.; Localization score no ink": loc_score_no_ink})

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(r'Results\Localization score no ink_new.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and save the output
writer.close()
