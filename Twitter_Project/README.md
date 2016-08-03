# 6002-Project-1

Welcome to the first project for 6002! In this project we will be scraping Twitter and performing some basic EDA and analysis on the data.  It will be divided into two parts:

1. Data collection and EDA
2. Encapsulation

## Submission: Always be commiting

For this project you will utilize Github to submit your work using a pull request.  You will first need to fork the repository if you haven't already (go to [https://github.com/gschool/6002-project-1](https://github.com/gschool/6002-project-1) and click "Fork")

![][fork]

Remember to commit your completed assignment to your Github fork. You will submit your solution with a git [pull request](https://help.github.com/articles/using-pull-requests). Here are step by step instructions of how to do this:

1. Fork this repository
1. [Install git](https://help.github.com/articles/set-up-git) on your computer
if you haven't already
1. Clone your forked repository onto your computer: `git clone https://github.com/<your username>/6002-project-1`
1. Edit the assignment files with your solutions
1. Add your changes to the repository: `git add <file you edited>`
1. Commit your changes: `git commit -m "Project 1 Solution"`
1. Push your changes to your fork: `git push origin master`
1. Make a pull request by going to `https://github.com/<your username>/6002-project-1` and clicking "Pull Requests" and then "New Pull Request"

It's good workflow to commit and push to your fork often, even when you're not done. Then you have older versions of your work in case you screw something up or lose something. We will get a notification of your submission once you submit a pull request.

#### Common git issues
1. If we've made changes to the repository after you forked it and you want to update your repository to reflect them, you can run this command: `git pull https://github.com/gschool/6002-project-1 master`

2. If you try to push and get "Repository does not exist" this probably means that you cloned from the Galvanize repo rather than your fork. Make sure you did step 1 and created a fork.

## Part 1

In part 1 we will be concerned with acquiring data from Twitter in a maintainable manner.  We will do this by combining our knowledge of the first 2 weeks of class to use libraries in the scientific Python ecosystem.

When working with APIs from companies there are a few points to keep in mind:

* Companies get tremendous value from leveraging ALL of their data
* Because of this they share some data (rate limiting and page limiting)
* If you use their platform with their users, you can get access to more (authenticate with API key)

### Resources

The following resources and references will be useful:

* [Python Twitter wrapper](https://github.com/sixohsix/twitter)
* [Mining the Social Web](https://github.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition)
* [Twitter API docs](https://dev.twitter.com/rest/public)
* [The Little MongoDB book](http://openmymind.net/mongodb.pdf)

### Getting Started

The first step in working with the Twitter API (or any API) is to usually get a API Key.  For Twitter we can do that here: https://apps.twitter.com/ (instructions [here](http://stackoverflow.com/questions/12916539/simplest-php-example-for-retrieving-user-timeline-with-twitter-api-version-1-1))

The next step to working with an API (Twitter in this case) is to locate the end points: https://dev.twitter.com/rest/public

Once you know what requests you can make and what responses will get returned, it is time to start constructing a query (think of an API as a interface into Twitter's "database")

For this exercise we will use the the Python [twitter](https://github.com/sixohsix/twitter) library to handle the authentication, an example can be found [here](https://rawgit.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition/master/ipynb/html/Chapter%201%20-%20Mining%20Twitter.html)

### Assignment

The overall goal of this assignment will be to build a data pipeline to capture tweets from the twitter stream, store them in MongoDB, and perform an analysis on the recovered tweets to perform EDA on.

Before we can perform our analysis however, we need to be sure we are properly collecting our data and verify that it is correct.

1. The first step is to download tweets, for this analysis we will be using tweets from the @Galvanize handle. To start let us use the REST search API.
    * Using the python wrapper for the twitter API authenticate with your credentials
    * Make a request and look at how many tweets are returned
    * Try to get as many tweets from the API as possible

2. Similar to the NYT, Twitter rate limits. Using the date parameter, try to get as many historic tweets as possible.

2. Once you have figured out how to capture as many tweets as you can, write this into a function.

3. Create a second function that given a set of tweets (in JSON), store these Tweets in a MongoDB collection.

3. Once you have these historic tweets in your database we can begin some EDA to verify the data we have collected.  Use the [Variety](https://github.com/variety/variety) tool to learn the distribution of each of the fields.

4. Plot a histogram of the distribution of each field of your results (using the Variety output): i.e. what percentage of the tweets have a `user` field, what percentage have a `geo.coordinates` field, etc.

 Now that we have some data to work with, we can begin to explore the textual content of the tweets.

5. The first step to exploring the text of our tweets is to extract the data from MongoDB into a more usable environment (i.e. Python).  Extract the text content/field, retweet count, and favorite count of the tweets using `pandas` to create a data frame.

6. Parse the text of each tweet to pull out all the hashtags, basically any "word" beginning with a `#` until the next whitespace separator.

7. Plot the distribution of hashtag frequency for the top 100 most used hashtags by the @Galvanize user.  Which is the most common?

5. In a similar manner, plot a distribution of retweets and a distribution of favorites.

6. What are the top 5 favorited tweets and top 5 retweeted tweets? 

7. Which tweets (if any) are in both the top 5 favorited and top 5 retweeted?

## Part 2: Refactor

Now that we have done some exploratory analysis and have determined a somewhat maintainable way to collect tweets, it is time to properly encapsulate our code.

### Assignment

7. First things first, we should create a `Scraper` class.  This will be the class to encapsulate all of the logic related to the scraping of tweets.

8. Allow your `Scraper` to be initialized with:
    * the URL endpoint you wish to scrape
    * the mongoDB url you would like to connect to

9. Define parameterized methods (if you haven't already created functions above) to encapsulate the following logic:
    * Web request to API
        * parameter: the timeout to wait between requests
        * parameter: number of tweets to scrape
        * parameter: the Twitter username to scrape the timeline of
    * Insertion into mongoDB
        * parameter: collection name

 __if a given tweet already exists, you should not insert it again__

 We want our scraper to be what is referred to as idempotent, or without side effects.  What does this mean? If you run your scraper over and over again with the same parameters (i.e. url endpoint), the result should be the same as if you ran it once (i.e. do not insert duplicate tweets into your database)

 Now that we have our scraper pretty well encapsulated, we want to encapsulate our EDA.

11. Create a `Parser` class that is initialized with:
    * a mongoDB database url
    * collection name to draw tweets from

12. Create methods to:
    * Parse/extract the hashtags
    * Create a distribution of hashtags
        * parameter: number of top n hashtags to use in distribution
    * Create a distribution of retweets
        * parameter: number of top n tweets to use in distribution
    * Create a distribution of favorites
        * parameter: number of top n tweets to use in distribution
    * Save these plots to a file

## Extra Credit 

Now that we have all of our functionality properly encapsulated in classes, let us make all of this functionality available on the command line.

13. Create a command line app that can run your scraper from the command line and output the appropriate plots

[fork]: images/forking_z.png
