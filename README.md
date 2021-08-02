# RecommendationSystem-Word2vec
[Play around with the interactive app by clicking this link](https://share.streamlit.io/sravanroy/recommendationsystem-word2vec/main.py)

# *Goal* :
* To build a recommendation engine that recommends a desired number of products to the consumers on an E-commerce website based on the past purchase behavior of the consumers

# *Data* :
* The E-Commerce data is taken from the UCI repository and can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/)

# *Solution* :
* Implemented Word2Vec model to form word embeddings for all the products
* The model can recommend similar products for a given product and also recommend products to customers by looking at most recent purchases ( can be limited such as last 10 products, 6 months etc.. )
