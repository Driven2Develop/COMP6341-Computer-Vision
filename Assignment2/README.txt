Iymen Abdella ID: 40218280
COMP425-6341 COMPUTER VISION
Programming Assignment 2
March 04 2022

Follow the command line prompts to progress through the scripts. 

Part 1: assignment2_Hough.py
- this follows the most basic of Hough Line detection by casting a vote system according to edges.
- The Upper and Lower threshold of the canny edge detector can be modified, however the tested values are default.
- the line threshold for the minimum number of votes to be cast in order for a line to be detected can be modified but the default is 80
The line threshold can be optimized depending on the picture as well. 

Part 2: assignment2_Harris.py
- looks for features in a black and white image, specifically corners with a high response according to the Harris algorithm
- the threshold can be modified, but too high and it wont detect anything, too low and it will begin to detect anomalies like edges. 
- the answer is normalized before returning because otherwise the threshold would not be as influential.
- the results detected far too many edges and not enough corners. 

Part 3: Feature descriptors
- matches features using ratio test