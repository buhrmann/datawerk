Title: Ideological twitter communities
Date: 2015-09-15
Slug: sna-twitter
Category: Data Posts
Tags: R, hadoop, networks, sna, flume, hive
Authors: Thomas Buhrmann

My current academic work revolves around the interactional "autonomy" of ideological communities in social networks. As part of my investigation I sometimes come across interesting little factoids. For example, I have been looking at the interaction of communities formed around the major political parties in Spain, and the most important media outlets (newspapers and TV stations). One observation I thought was interesting has to do with the bias in media consumption exhibited by different communities. For example, without going into detail here about how I identified the individual communities, here is a graph showing the inequality in retweet activity exhibited by party-affiliated twitter communities:

<img src="/images/sna/media-consumption.png" alt="Inequality in media consumption."/>

Here each dot represents the gini coefficient of the distribution of retweets by a given community over different media outlets. In other words, for each community we count how many times its members have retweeted a given media account. One community may, for example, primarily retweet a particular newspaper, while another community may prefer to retweet news from a particular TV station. The gini coefficient measures how unequal each community divides its retweet activity amongst the different media outlets. A very low score would mean that all media outlets receive more or less the same amount of retweets by a given community. A very high score, on the other hand, implies that the community mostly retweets a single outlet's news.

What may or may not be surprising to some, is that according to the graph above Podemos (arguably a very polarizing if not polarized) ideological community, most equally divides its attention across the different media. Of course, that does not mean they agree with all of them. It is likely, rather, that many retweets are in fact critical of the retweeted content (particularly if it originates in a very right wing media outlet).

To complement the above we can also look at the "transposed" picture, i.e. the number of retweets each media outlet receives from a given twitter community, expressed in the following graph as proportions:

<img src="/images/sna/by_media.png" alt="Inequality in media consumption."/>

It is interesting, though not surprising, that most outlets have a well defined readership (or rather retweeters). The more leftwing party-affiliated communities make up a good proportion of the retweeters of the online newspaper Publico, for example. The extreme right, in contrast, i.e. PP and Vox, make up most of the ABC retweeters. 

(Note that "Unknown" here does not correspond to a well-defined community, but rather represents the remaining mass of "undefinable", or "non-affiliated" twitter accounts).

The code for reproducing the analysis can be found <a href="https://github.com/buhrmann/tweetonomy" target="_blank">on github</a>.