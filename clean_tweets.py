#takes as input a list of tweets' text, previously downloaded, and gives as output
#the only text, without '@...','http:...','#' and 'RT'(Re-tweets)

data = open("event_text.txt",'r')
file = open("clean_tweets.txt",'w')
tweet_list = list()
for line in data:
      if len(line) > 0:
            tweet_list.append(line)
data.close()
tweet_list_clean1 = list()
for tweet in tweet_list:
      tweet_list_clean1.append(tweet.replace("#",""))

tweet_list_clean2 = list()

for tweet in tweet_list_clean1:
      a = tweet.find("http:")
      b = tweet.find(" ",a)
      new = tweet.replace(tweet[a:b],"")
      tweet_list_clean2.append(new)
      
tweet_list_clean3 = list()
tweet_list_clean4 = list()
      

for tweet in tweet_list_clean2:
            a = tweet.find("@")
            b = tweet.find(" ",a)
            new = tweet.replace(tweet[a:b],"")
            a = new.find("@")
            b = new.find(" ",a)
            new1 = new.replace(tweet[a:b],"")
            a = new1.find("@")
            b = new1.find(" ",a)
            new2 = new1.replace(tweet[a:b],"")
            tweet_list_clean3.append(new2)
            
for tweet in tweet_list_clean3:
     tweet_list_clean4.append(tweet.replace("RT",""))
      
      
for tweet in tweet_list_clean4:
      file.write(tweet)
file.close()


      
      
      




      
