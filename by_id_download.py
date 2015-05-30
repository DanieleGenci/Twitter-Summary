#download tweets' text by id. Thanks to a list (attached file: id_list) we can catalogate the tweets in function of the events. 


import tweepy

data = open("id_list",'r')
list_num1 = list()

#to select a different list relative a different event, search in the README file
#and change in the number in a for cicle 

for line in data:
      if line.find(" 15\n") != -1:
            string = line[0:line.find(" 15")]
            list_num1.append(string)
      else: exit


list_event1 = list()    

consumer_key= 'S15IIrFXmRX51ZKqca7ZoSEUK'
consumer_secret= 'lEVPd7fdNeFjk2Et45JW5Nb0tDuWg9vUgWiKzh7OHY2Y5IPoeP'
access_key = '3081475989-cEb5rwZBdeF6s10KXWpe3AN0qiiJa2T1m2YiNu5'
access_secret = 'ZIn6J6aKLk7M8BozBDVk4IYYMTPzh4KzGeVz3szaZq9aD'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

a=0

#create file txt with all the tweets' text related to a specific event.
#each line corresponds to tweet's text

file = open("event_text.txt",'w')

for line in list_num1:
      
   try:
         tweet = api.get_status(line)
         print (tweet.text)
         file.write(str(tweet.text))
         file.write("\n")
         list_event1.append(tweet.text)
   except: list_num1.remove(line)


file.close()



