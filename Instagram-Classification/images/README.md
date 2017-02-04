# Applied Data Analysis project

## Instagram Classification

This folder serves two purposes:

1) It is kind of buffer for images queue. In other words, the download workers download the images here so that classification workers can analyse them. Once the classification worker is done it removes the image. 

2) Images that failed to download are stored here in form of empty files with name InstagramID.jpg.failed.

In a perfect scenario this folder will remain empty after the pipeline is finished.

Optimally, clean this folder before rerunning the pipeline. If you don't the pipeline will work, however the list of failed-to-download images won't be really helpful as it will contain failed-to-download images with you not being able to tell which belong to your run of interest.