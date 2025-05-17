# padjective

Calculate p-adic adjective embeddings

# Purpose

We want to get a hierarchy of tags: for any given pair of tags, which one is more likely to appear first in a product title? Then we want to identify "equivalent depth"
tags where they generally appear at the same depth, and ultimately assign an integer depth to every tag.

# Method

## tagbattle.py

(Named tagbattle in honour of kittenwar, a very addictive website.)

For each product:
 - For each tag, we look to see if the tag is a subset of another tag in that same product line. e.g. if the tags are "chocolate,milk chocolate" then we ignore chocolate
   and only work with milk chocolate
 - If the title has a " - " in it (a dash surrounded by whitespace), then we pretend that we have two separate titles. e.g. If we have "Easter bunny - milk chocolate" 
   then we don't say that milk comes after Easter. We just don't know the relationship between Easter and milk from this example
 - For each title:
   - We determine where the tag appears in the title, i.e. which character in the title is the start of the tag using a case-insensitive search. For many tags the answer
     will be "nowhere"
   - For each pair of tags which are somewhere in the title, record which one came first. Pretend that it's a competition, and record which tag won and which tag lost into
     a sqlite database table

## ranking.py
 
Use the choix library and the sqlite database from tagbattle.py to produce ranking tables for each tag.

## display.py

Creates text and HTML and images from the results of ranking.py

    
