Title: Retrieving your Google Scholar data
Date: 2015-01-27 
Slug: scholar
Category: Data Posts
Tags: scraping, python
Authors: Thomas Buhrmann

For my interactive CV I decided to try not only to automate the creation of a bibliography of my publications, but also to extend it with a citation count for each paper, which Google Scholar happens to keep track of. Unfortunately there is no Scholar API. But I figured since my own profile is based on data I essentially donated to Google, it is only fair that I can have access to it too. Hence I wrote a little scraper that iterates over the publications in my Scholar profile, extracts all citations, and bins them per year. That way I can track how many citations each paper had over time. Like this graph, for example:

<figure>
<img src="/images/scholar/citations.png" />
</figure>

The source code for the scraper is available <a href="https://github.com/synergenz/datawerk/blob/master/theme/static/py/citations.py">here on github</a>. It uses <a href="http://www.crummy.com/software/BeautifulSoup/">Beautiful Soup</a> for the html parsing. Note that it deliberately runs very slowly (waits a random amount of time between page loads), because otherwise Google will notice the scraping attempt and will lock your IP out, at least for a while (it happened to me initially). Feel free to use it to liberate your own Scholar data.

