pelican -s publishconf.py
ghp-import output
git push git@github.com:buhrmann/buhrmann.github.io.git gh-pages:master
pelican -s pelicanconf.py
