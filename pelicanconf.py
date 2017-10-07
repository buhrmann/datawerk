#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Thomas Buhrmann'
SITENAME = 'datawerk'
SITEURL = 'http://localhost:8000'

PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'
DEFAULT_DATE = 'fs'
DEFAULT_DATE_FORMAT = '%B %d, %Y'

SUMMARY_MAX_LENGTH = 100

TYPOGRIFY = True

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

THEME = 'theme'

INDEX_SAVE_AS = 'posts.html'

PLUGIN_PATHS = ["../pelican_plugins", "plugins"]
PLUGINS = ["sitemap", "clean_summary", "tag_cloud", "tag_graph"]

DEFAULT_PAGINATION = 10

DISQUS_SITENAME = "datawerk"

TAG_CLOUD_STEPS = 4
TAG_CLOUD_MAX_ITEMS = 100

SITEMAP =  {
    'format': 'xml',
    'priorities': {
        'articles': 0.5,
        'indexes': 0.5,
        'pages': 0.5
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly'
    }
}

DELETE_OUTPUT_DIRECTORY = True

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
