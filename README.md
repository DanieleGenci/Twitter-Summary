# Twitter-Summary
Our code is divided in 3 part:

1st file is named by_id_download.py, to be run must be installed python version 2.8 and tweppy library.
(in IOS terminal)>> pip install tweepy
Notice: to run the code it's needed to attach the "id_list" file found in the repository.  
This file permits us to download a list of tweets inherent to a specific event, consider the id(readme) file with the list of possible event. Changing the two number value in first "for cycle" and running the code, downloads tweets' text inherent a select event. Automatically saves it in a txt file named "event_text.txt".

2nd file is named clean_tweets.py allows to clean the tweets' text from special characters(@,#,http:). The file takes in input the txt file which was created in the privious file of our code.(event_text.txt). The output is saved in another file named "clean_tweets.txt".

3rd file is named Makefile.py contains the main part. To be run, must be installed Tkinter library and import some libraries.
(in IOS terminal)>> pip install python-tk
The default input is the txt file produced by the privious file(clean_tweets.txt). But if you want, changing the string line (txt_open), you can select different text file to apply our algorithm.
As output we have the Tree rappresentation of the information about the selected event and the possibility by clicking on radio botton of corrispondig block to see all the tweets specific to the event block. 
