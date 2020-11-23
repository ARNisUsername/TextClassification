#Import neccessary modules
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

#Read the first 2000 parts of the dataset
number_titles = 2000
movie_text = pd.read_csv('movies_metadata.csv', low_memory=False)[:number_titles][['original_title','overview']]
movie_text = movie_text.fillna("none")
#Create a list with the movie overviews(summaries)
movie_overview = list(np.array(movie_text['overview']))

#Begin making the lists that will be used for the parsing
titles = [title for title in movie_text['original_title']]
ratings = []
for title in titles:
    #Format the title so that the rotten tomatoes website can read it
    title = '_'.join(title.split())
    #Get the URL based off of the title, and use BeautifulSoup and requests to parse through it in html
    theURL = 'https://www.rottentomatoes.com/m/{}'.format(title)
    google_page = requests.get(theURL)
    soup_google = BeautifulSoup(google_page.text, "html.parser")
    #In the span class, there is the Rotten Tomatoes rating, and this line of code finds it
    rating_html = soup_google.find_all('span', {'class':'mop-ratings-wrap__percentage'})
    #There are some movies that Rotten Tomatoes hasn't rated. This try except code will append 'no rating' if that's the case
    canDo = True
    try:
        the_split = str(rating_html[0]).split('>')[1]
    except:
        canDo = False
        ratings.append('no rating')
    #If there was no exception, the code then appends the actual rating to the list
    if canDo:
        the_num = int(the_split.rstrip('\n').split('%')[0])
        ratings.append(the_num)
        
#Make everything into numpy arrays for the pd DataFrame to understand        
titles = np.array(titles)
ratings = np.array(ratings)
movie_overview = np.array(movie_overview)

#Finally, make everything into a pandas datafram and convert that to a csv file
dataset = pd.DataFrame({
    "title":titles,
    "RT rating":ratings,
    "overview":movie_overview
})
dataset.to_csv('movie_rating_overview.csv', index=False)
