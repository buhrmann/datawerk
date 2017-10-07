Title: Tag graph plugin for Pelican
Date: 2015-01-27 
Slug: tag-graph
Category: Data Posts
Tags: graph, python, visualization, d3
Authors: Thomas Buhrmann

On my front page I display a sort of sitemap for my blog. Since the structure of the site is not very hierarchical, I decided to show pages and posts as a graph along with their tags. To do so, I created a mini plugin for the Pelican static blog engine. The plugin is essentially a sort of callback that gets executed when the engine has generated all posts and pages from their markdown files. I then simply take the results and write them out in a json format that d3.js understands (a list of nodes and a list of edges indexed on node positions in their list).

The plugin's source code is available <a href="https://github.com/synergenz/datawerk/blob/master/plugins/tag_graph.py">here on github</a>. It you're interested in the Pelican blog engine, check it out <a href="http://blog.getpelican.com/">here</a>.

