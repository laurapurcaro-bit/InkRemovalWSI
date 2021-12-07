# InkRemovalWSI
Ink removal of Whole Slide Images by color detection


## Introduction
Pathology specimens frequently have significant borders that must be recognized from gross inspection to microscopic examination1. 
Each boundary is usually marked with a different color of tissue-marking dye (TMD) or Indian ink, which aids in both macroscopic and microscopic identification2. 
The following are some of the benefits of using a variety of ink colors: 
1) To determine the margin status of a specimen 
2) To determine the orientation of a specimen 
3) To benefit from post-grossing three-dimensional reconstruction
4) To reduce identification error when multiple samples are required from the same tissue (e.g. prostate needle biopsies) or when similar specimens are obtained from different patients. 
Indeed, the presence or absence of ink is frequently the deciding factor in whether the margin is positive for tumor cells. 
In order to avoid error of interpretation, the most used colors are black, green, red, blue and yellow.

<p align="center">
<img src="Images/Picture1.png?raw=true" width="700" height="200"><br>
  <em>Figure 1. Examples of ink in surgical pathology specimens</em>
</p>

A skilled pathologist can make a definitive diagnosis of a variety of illnesses by performing a careful visual investigation, which takes a long time and effort and is unavoidably prone to error. 
The digitalization of biopsy glass slides is required for computer-assisted diagnosis. Marker marks emerge in the digital pictures once the slides have been scanned. As a result, computer techniques may be prone to misinterpreting marker-colored areas as tissue information. 
The removal of regions with marker signs is beneficial since, in most situations, areas covered by marker are of lesser value (therefore overpainted by the pathologists to show something more important). As a result, removing these regions may have little impact on the tissue that is important for diagnosis.
There is no standard way for deleting this data from whole slide pictures, which limits their utility in research and study. Marker ink contaminates various locations and in an uneven manner, making removal of marker ink from these high-resolution full slide pictures a difficult and time-consuming task. We present a color detection-based pipeline that produces ink-free pictures without sacrificing information or image quality. This process will then be incorporated into FolDet, a folds detection algorithm, to see how ink removal affects fold detection performance.

## Materials and Methods
#### Dataset
The dataset for this project includes twenty-one WSI with ink, H&E stained. The images were selected with the function ```MatchbyColor()``` within a larger dataset of 440 WSI, where 220 were from the “Folds” group and 220 were “Folds-free”. Within the 21 selected digital slides, ten of them were obtained from the “Folds” group and eleven of them were “Folds-free”.


#### Algorithm
#####	Step 1: Find ink in Whole Slide Images.
Firstly, we create a list with all images’ path to loop throughout the dataset. This is achieved with ```os.listdir()``` function.
Next, ```MatchbyColor()``` function was specifically created to check for all 5 ink colour’s types: green, blue, yellow, red and black. The function takes 3 arguments: ```src```, which correspond to the source image; ```lower``` and ```upper```, which represent lower and upper bounds of ```cv2.inRange()``` sub-function.
A list of three RGB colors ranges was created to cover all the possible ink types of an image could have: green-blue, yellow-red, black. Each colour range has a lower and upper bound. 
 Once these ranges are passed to ```cv2.inRange()``` sub-function, a mask (black or background; white or foreground) of the corresponded colour is outputted and passed to ```cv2.countNonZero()``` function, where it will count how many white pixels are present in the mask. If at least one white pixel exists, then it means that ```cv2.inRange()``` founded the specific colour ink and ```MatchbyColor()``` function will output a ```True``` Boolean value, ```False``` otherwise. This output will be used by ```selectImageWithInk()``` to select the corresponding image in case of ```True``` value and save it into a new folder called 'Ink check' that will automatically create.












