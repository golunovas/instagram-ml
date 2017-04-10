# instagram-ml
Unpossibly machine learning competition

My solution uses three CNNs:
    1. CNN for image aesthetics analysis (https://github.com/aimerykong/deepImageAestheticsAnalysis)
    2. VGG for image classification
    3. GoogleNet for style recognition

For each image I extract features using those CNNs and concatenate them. 
Then I apply PCA to reduce number of features to 64 and use SVM regression for getting prediction.
PCA and SVM were trained for each account separately.U
