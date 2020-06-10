# halloween_ar_mask

A computer vision project to overlay augmented reality masks over people's faces for Halloween. To run the program, after cloning it, navigate to the project directory and run the command: `python mask.py`

To stop program execution, press `q` (lowercase Q) in your terminal window.

To swap your own mask image over the current mask image, replace the reference to the current image in the line below (from mask.py) to the name of the image that you wish to swap it with. 

`mask = cv2.imread('groucho_glasses.png', -1)`

Also, make sure that the new image is contained within the current directory or else modify the path of the command to refer to the proper directory where the file is located.