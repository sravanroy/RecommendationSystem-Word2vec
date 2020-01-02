# RecommendationSystem-Word2vec
Recommendation system for customers in an E-commerce site based on Word2Vec embeddings

# *Goal* :
* To build a recommendation engine that automatically recommends a certain number of products to the consumers on an E-commerce website based on the past purchase behavior of the consumers

# *Data* :
* The E-Commerce data is taken from the UCI repository and can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/)

# *Solution* :
* Implemented Word2Vec model in NLP to form word embeddings for all the products
* Recommend top 'n' products for a customer based on the similarity of vectors between his purchased products and the entire products in the site
