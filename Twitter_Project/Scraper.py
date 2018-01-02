import twitter 
import pymongo
import time
import pandas as pd
import numpy as np
import ordered_set 
import matplotlib.pyplot as plt
from collections import Counter




class Scraper(object): 

    def __init__(self, db_name):
    
        # AUTHENTICATE TWITTER ACCOUNT
        CONSUMER_KEY = 'LpCTmKNwakL1QUqavVL6Plz4J'
        CONSUMER_SECRET = 'yq1znQZdXq7QCnWmzXO7QK3pnPUJ701hRWhiXPtxsfoBIzfnwV'
        OAUTH_TOKEN = '3064452698-7uSHiEaTborY8fWoYcSZurF0Zf2hpNCkmTmPMDw'
        OAUTH_TOKEN_SECRET = 'VTbSI0QCXfp6wmYzJdjPhrcFgCSFNqC5qbFXNChuqqhro'

        auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)

        self.twitter_api = twitter.Twitter(auth=auth)

        # references mongo data base called db_name
        self.mongoDB_url = pymongo.MongoClient().db_name
        # tweet search result collecion 
        self.collect     = [] 

    def Web_request_to_API(self, total_tweets, twitter_username):

        """ Returns  a collection populated with tweets"""
        self.mongoDB_url = self.mongoDB_url.twitter_username
        ids = ordered_set.OrderedSet()
        #### statuses.user_timeline
        #### don't have time to use num_tweets as the total number of tweets that are scraped
        search = self.twitter_api.statuses.user_timeline(screen_name = twitter_username, 
                                                         count = 200)
        # to prevent duplicate tweets
        maxID = search[-1]['id'] - 1

        while len(ids) <= total_tweets:

            self.collect.append(search)
            try:
                search = self.twitter_api.statuses.user_timeline(screen_name = twitter_username, 
                                                                 count = 200, 
                                                                 max_id = maxID) 
            except KeyError, e:
                print 'keyerror'
                break
            except twitter.TwitterHTTPError:
                print 'sleeping for 16 minutes'
                time.sleep(16*60)

            maxID = search[-1]['id'] -1

                #check number of tweets scarped 
            for i in range(0,len(search)):
                ids.add(search[i]['id'])

    def Insert_into_mongoDB(self,collection_name):
        for i,row_search_result in enumerate(self.collect):
            for j,col in enumerate(row_search_result):
                self.mongoDB_url[collection_name].insert(self.collect[i][j])
        print "{0} has been inserted into mongo".format(collection_name)

    def unicode_filter_df_series(self, df_series):
        """Returns filtered strings as chars in numpy array"""
        SL = [] # char list

        for i in xrange(0,len(df_series)): # turns string into list of chars
            sl = list (df_series[i])
            SL.append(sl)
        
        for i in xrange(0, len(SL)):       # filters chars 
            for j in xrange(0,len(SL[i])):
                SL[i][j] = self.filter_unicode(SL[i][j])

        return np.array(SL)

    def filter_unicode(self,x):
        list_of_ascii = [y for y in x if ord(y) < 128]
        filtered_string = ''.join(list_of_ascii)
        return filtered_string








class Parser(object):

    def __init__(self, collect_name, db_name):

        sc = Scraper(db_name)
        sc.Web_request_to_API(200,'Galvanize')
        sc.Insert_into_mongoDB(collect_name)

        self.mongoDB_URL     = sc.mongoDB_url

        self.collection_name = collect_name

        self.hash_df         = pd.DataFrame({'emptylist':[0,0,0]})

        self.scraper = sc


    def Extract_Hashtags(self):
        """returns a data frame with frequencies of hashtags and the hashtages themselves"""

        # connect to database
        db = self.mongoDB_URL

        #connect to collection
        ## data is a df cursor
        data = db[self.collection_name]

        ## Data is a list
        Data = list(data.find())

        datalist = [doc for doc in Data]
        ## change datalist into dataframe
        datadf = pd.DataFrame(datalist)
    
        ## group all relevent fields into one dataframe 
        tweetdf = pd.DataFrame([datadf['text'],datadf['retweet_count'],datadf['favorite_count'], datadf['_id']])

        tdf = tweetdf.T

        #print "head for tdf"
        print tdf['text'].head()
        print tdf['text'].count()

        # series wont display, so a unicode filter is applied
        tdf['text'] = self.scraper.unicode_filter_df_series(tdf['text'].values)

        # change tweet data type from unicode to string
        for i in range(0,len(tdf['text'].values)):
            tdf['text'].values[i] = str(tdf['text'].values[i])


        hashtage = []
        print type(tdf['text'].values)
        print type(tdf['text'].values[0])
        #print type(tdf['text'].values[0])


        for tweet in tdf['text']:
            # turns tweet into a list of words 
            wordlist =  tweet.split(' ')
            
            for word in wordlist:
                # picks out individual words from list
                
                for i,char in enumerate(word):
                    # picks out characters from words
                    if(char == '#'):
                        
                        hashtage.append(word)

        hashtag = []


        print "hastage types"
        print type(hashtage[0])
        print type(hashtage[0][0])
        print type(hashtage[0][0][0])




        # applies unicode filter and strips away extra column containing list of object data types
        for i,word in enumerate(hashtage):
            hashtag.append(self.scraper.filter_unicode(hashtage[i]))
    
            # changes data type from unicode to string  
            for i, word in enumerate(hashtag):
                hashtag[i] = str(hashtag[i])

        c = Counter(hashtag)
        hashdic = {}
        # counts the frequency of each hashtag, then stores the pair in a dictionary
        for word in hashtag:
            hashdic[word] = c[word]

        print "first element in hashtag (post filter) is {0}".format(hashdic.items())
    
        # use hashtage dict to create hashtage dataframe 
        hashdf = pd.DataFrame({'hashtags':hashdic.keys(),'frequency':hashdic.values()})
        # set the index as the hashtags so that the hashtags appear in the graph
        #hashdf = hashdf.set_index(['hashtags'])

        # sort frequency for better graphical presentation
        hashdf = hashdf.sort(['frequency'])
        # reverse sorting order for better graphical presentation 
        hashdf = hashdf[::-1]

        self.hash_df = hashdf

        print "head of self.hash_df"
        print self.hash_df.head()
        print "self.hash_df.columns"
        print self.hash_df.columns

    def Dist_of_Hashtags(self, num_of_hashtages):

        # The top most used hashtages by the @Galvanize user
        self.hash_df['frequency'][0:num_of_hashtages].plot(kind='bar', figsize=(12, 5) )#, use_index = True , title = 'Top {0} Most Used Hashtags'.format(num_of_hashtages));
        plt.savefig('freq')
        #### MUST save plot to a file
        print "freq bar graph has been saved"

    def Dist_of_Retweets(self, num_of_tweets):
        selftcountdf[0:num_of_tweets].plot(kind='bar', figsize=(15, 5) ,title = 'Frequency of Retweets')
        plt.savefig('retweets')
        #### MUST save plot to a file

    def Dist_of_Favorites(self, num_of_tweets):
        return favdf[0:num_of_tweets].plot(kind='bar', figsize=(15, 5) ,title = 'Favorite Tweets')
        plt.savefig('favoriteTweets')
        ## SAVE plots to a file 


if __name__ == '__main__':

    p = Parser('project2','test')
    p.Extract_Hashtags()
    p.Dist_of_Hashtags(50)








