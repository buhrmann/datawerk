from pelican import signals
import json

def index_of(slug, list):
    for i, e in enumerate(list):
        if e['name'] == slug:
            return i
    return -1

def tag_graph(generator):

    nodes = []
    for tag, articles in generator.tags.items():
        nodes.append({"name":tag.slug, "value":len(articles), "group":1, "text":tag.name, "url":"/tag/"+tag.slug+".html"})
    
    for a in generator.articles:
        nodes.append({"name":a.slug, "value":1, "group":2, "text":a.title, "category": a.category.name, "url": a.settings['SITEURL'] + "/" + a.url})
            
    links = []
    i = 0
    for tag, articles in generator.tags.items():
        for a in articles:
            links.append({"source":i, "target":index_of(a.slug, nodes), "value":1})
        i += 1
        
    graph = {"nodes":nodes, "links":links}    
    with open("./theme/static/js/tag_graph.json", 'w') as file:
        json.dump(graph, file)
    

def register():
    signals.article_generator_finalized.connect(tag_graph)