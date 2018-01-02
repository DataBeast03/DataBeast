import twitter 
import pymongo
import time
import pandas as pd
import numpy as np
import ordered_set 
import matplotlib.pyplot as plt
from collections import Counter

class Scraper(object): 

    def __init__(self):
    
        # AUTHENTICATE TWITTER ACCOUNT
        CONSUMER_KEY = 'LpCTmKNwakL1QUqavVL6Plz4J'
        CONSUMER_SECRET = 'yq1znQZdXq7QCnWmzXO7QK3pnPUJ701hRWhiXPtxsfoBIzfnwV'
        OAUTH_TOKEN = '3064452698-7uSHiEaTborY8fWoYcSZurF0Zf2hpNCkmTmPMDw'
        OAUTH_TOKEN_SECRET = 'VTbSI0QCXfp6wmYzJdjPhrcFgCSFNqC5qbFXNChuqqhro'

        auth = twitter.oauth.OAuth(OAUTH_TOKEN, 
        						   OAUTH_TOKEN_SECRET,
                                   CONSUMER_KEY, 
                                   CONSUMER_SECRET)

        self.twitter_api = twitter.Twitter(auth=auth)
        # # references mongo data base called db_name
        # self.mongoDB_url = pymongo.MongoClient().db_name
        # # tweet search result collecion 
        # self.collect     = [] 

  	def Web_request_to_API(self, total_tweets, username):

		count = 199
		ids = ordered_set.OrderedSet()

		search = self.twitter_api.statuses.user_timeline(screen_name = username,
														 count = count)
		maxID = search[-1]['id'] -1

		# end loop once number of scraped tweets drops below 5 per search 
		while len(ids) < total_tweets:
		    
		    self.collect.append(search)
		    
		    try:
		        search = self.twitter_api.statuses.user_timeline(screen_name = username,
		        												 count = count, 
		        												 max_id = maxID) 
		    except KeyError, e:
		        print 'keyerror'
		        break
		    except twitter.TwitterHTTPError:
		        print "twitter.TwitterHTTPError"
		        print 'sleeping for ~ 16 minutes'
		        time.sleep(16*60)

			# track ids in order to check for duplicates later
		    for i in xrange(0,len(search)):
		        ids.add(search[i]['id'])

		print len(ids)


if __name__ == '__main__':

	scrape = Scraper()
	scrape.Web_request_to_API(3000,"Galvanize")