Title: Elegans now features PubMed search
Date: 2015-07-20
Slug: elegans-pubmed
Category: Data Posts
Tags: visualization, d3, neo4j, python, graph, nosql
Authors: Thomas Buhrmann

I've added a new PubMed search feature to <a href="https://elegans.herokuapp.com/">_Elegans_</a>, the visual worm brain explorer. The idea here is to show the network of _C. Elegans_ neurons that get mentioned in more than n papers on PubMed, in the context of a given search query. So, for example, if one is interested in the worm's chemotaxis behaviour, one would type in 'chemotaxis' and choose the citation threshold n. Initiating the search will then return the neurons that get mentioned in at least n papers along with the word 'chemotaxis'. The search is in fact performed once for each neuron class, and the results collated. One can further select to either show the network of only those neurons returned by the search, or use the resulting neurons to populate the subgraph search panel, which contains additional filtering options.

<img src="/images/elegans/pubmed.png" alt="Pubmed search in Elegans."/>