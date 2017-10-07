Title: C. elegans connectome explorer
Date: 2015-01-15
Category: Data Apps
Tags: visualization, d3, neo4j, python, graph, nosql
Slug: elegans
Authors: Thomas Buhrmann

I've build a prototype <a href="https://elegans.herokuapp.com">visual exploration tool</a> for the connectome of c. elegans. The data describing the worm's neural network is preprocessed from publicly available information and stored as a graph database in neo4j. The d3.js visualization then fetches either the whole network or a subgraph and displays it using a force-directed layout (for now).

<figure>
<a href="https://elegans.herokuapp.com"><img src="/images/elegans/elegans.png" alt="Elegans"/></a>
</figure>