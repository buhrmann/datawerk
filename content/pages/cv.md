Title: Interactive Resume
Date: 2014-10-12 13:25
Modified: 2014-10-12 13:25
Category: Visualization
Tags: javascript, d3.js, visualization, scraping
Slug: cv
Template: cv

I'm a computer scientist with 8 years of academic and commercial experience in the areas of artificial intelligence and machine learning. My academic output primarily deals with the evolutionary optimization of neural networks to understand cognitive functions. Outside of academia I spent more than 4 years developing and managing the development of commercial AI software at NaturalMotion (a provider of middleware solutions to some of the world's biggest computer game studios, such as Rockstar and Lucasarts). I'm now looking to apply my skills and experience in the field of data science and big data. I've worked on both back-end (SQL, NoSql such as MongoDB or Neo4j, Hadoop, Flume, Hive, etc.) and front-end (javascript and R/shiny-based data visualizations and python web apps). I'm  comfortable programming in C++, Python, R or javascript and regular use R and scikit-learn for machine learning projects (see this blog for non-academic samples).

Find a printer-friendly <a href="http://bit.ly/1UEi7dF" target="_blank">pdf resume here</a>.

###Experience###
For a full resume also see my <a href="https://www.linkedin.com/in/thomasbuhrmann" target="_blank">LinkedIn</a> account.
<div id="timeline-exp"></div>

###Education###
<div id="timeline-ed"></div>

###Publications###
A list of my publications. For the data used in this graph I've written a little python robot to scrape articles and citations off of <a href="http://scholar.google.es/citations?user=M1FQwSUAAAAJ&hl=en">google scholar</a>. The <a href="https://github.com/synergenz/datawerk/blob/master/theme/static/py/citations.py">source code</a> is available on <a href="https://github.com/synergenz/">github</a> along with the rest of this <a href="https://github.com/synergenz/datawerk">blog</a>.

<ul class="nav nav-tabs" role="tablist">
  <li class="active"><a href="#stream" role="tab" data-toggle="tab">Stream</a></li>
  <li><a href="#stack" role="tab" data-toggle="tab">Stack</a></li>
</ul>


<div class="tab-content">
  <div class="tab-pane active" id="stream">
      <span class="capCenter">Each colored stream represents a single publication's number of citations over the years.</span>
  </div>
  <div class="tab-pane" id="stack">
      <span class="capCenter">Each colored rectangle represents the number of citations for a given publication in a given year.</span>
  </div>
</div>

#### All publications ####
As found on google scholar. Number of citations in brackets.

<div class="row">
    <div class="col-sm-12 col-md-12">
        <ul class="bib" id="list" style="margin-top: 15px;"></ul>
    </div>
</div>