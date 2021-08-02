import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec
import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE
#from sklearn.externals.joblib import joblib
import joblib
from PIL import Image

import warnings;
warnings.filterwarnings('ignore')

header = st.beta_container()
dataset = st.beta_container()
dataprep = st.beta_container()
model_training = st.beta_container()
prompt = st.beta_container()
evaluation = st.beta_container()
evaluationfinal = st.beta_container()



def append_list(sim_words, words):

    list_of_words = []

    for i in range(len(sim_words)):

        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words



def display_tsne_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, perplexity = 0, learning_rate = 0, iteration = 0, topn=None, sample=10, name = None):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]

    word_vectors = np.array([model[w] for w in words])

    three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]

    data = []


    count = 0
    for i in range (len(user_input)):
                text = words[count:count+topn]
                text = [products_dict[i][0] for i in text]

                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0],
                    y = three_dim[count:count+topn,1],
                    z = three_dim[count:count+topn,2],
                    text = text,
                    name = name,
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )


                data.append(trace)
                count = count+topn

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0],
                    y = three_dim[count:,1],
                    z = three_dim[count:,2],
                    #text = products_dict[words[count:][0]][0],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 25,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'red'
                    }
                    )

    data.append(trace_input)

# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        #x=1,
        #y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 600
        )


    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def aggregate_vectors(products):
	product_vec = []
	for i in products:
		try:
			product_vec.append(model[i])
		except KeyError:
			continue

	return np.mean(product_vec, axis=0)


def get_data(filename):
	df = pd.read_csv(filename)

	return df

with header:
	st.title('Recommendation System using Word2Vec')

with dataset:
	st.markdown('## The model recommends a desired number of similar products to the consumers on an E-commerce website based on the past purchase behavior of the consumers')
	st.markdown('The data used for this task is taken from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/online+retail)')

	df = get_data('Online_Retail.csv')
	st.dataframe(df.head())

	st.markdown('- Here is the snapshot of the data after deleting the unwanted columns')
	st.markdown('- The **StockCode** is unique to each product and **Description** column contains the information for it. **CustomerID** is the unique identifier for the customers')


with dataprep:
	st.markdown('> # Data Preparation...')

	st.markdown('- After dropping the bad rows and converting the columns to their appropriate types, we split the customer ids to training and testing.')
	# check for missing values
	prep_code = '''
customers = df['CustomerID'].unique().tolist()

# shuffle customer ID's
random.shuffle(customers)

# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# split data into train and validation set
train_df = df[df['CustomerID'].isin(customers_train)]
validation_df = df[~df['CustomerID'].isin(customers_train)]
		'''

	st.code(prep_code, language = 'python')

	st.markdown('- Now we create a list of products purchased by the customers in the training set')

	train_code = '''
# list to capture purchase history of the customers
purchases_train = []

# populate the list with the product codes
for i in tqdm(customers_train):
    temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_train.append(temp)
		'''

	st.code(train_code, language = 'python')
	st.markdown('- Similarly, we create a list of purchased products for customers in the validation set')
	test_code = '''
purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['CustomerID'].unique()):
    temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_val.append(temp)
		'''

	st.code(test_code, language = 'python')
	st.markdown("- Let's look at the frequency distribution of the customers based on the number of products they purchased")

	image = Image.open('distribution.png')
	st.image(image, caption='# of customers by products purchased', width = 500)


with model_training:
	st.markdown('> # Model Training...')

	st.markdown('- **Word2Vec** algorithm is used to create the embeddings for the products purchased')
	st.markdown('- Since the training corpus is less (~ 4k), we use **skip gram** architecture to train our model')
	st.markdown('- As the main goal is to capture more **topic/domain** information of products purchased by customers and not focus on individual products, we chose a larger **window** of 10')
	st.markdown('- To optimize the training of skipgram, we use **negative sampling** and randomly select negative examples for each product while training')
	model_code = '''
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(purchases_train, progress_per=200)

model.train(purchases_train, total_examples = model.corpus_count,
            epochs=10, report_delay=1)
		'''

	st.code(model_code, language = 'python')

	st.markdown("- Since the model is trained on product stockcodes, let's create a dictionary with product stockcodes and their descriptions")
	dict_code = '''
products = train_df[["StockCode", "Description"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='StockCode', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()
		'''

	st.code(dict_code, language = 'python')


# remove missing values
df.dropna(inplace=True)

#convert stock_code to string type since it's a unique combinations of numbers and letters
df['StockCode']= df['StockCode'].astype(str)


customers = df['CustomerID'].unique().tolist()


# shuffle customer ID's
random.shuffle(customers)

# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# split data into train and validation set
train_df = df[df['CustomerID'].isin(customers_train)]
validation_df = df[~df['CustomerID'].isin(customers_train)]


purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['CustomerID'].unique()):
	temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
	purchases_val.append(temp)


loaded = 1

if not loaded:
	# list to capture purchase history of the customers
	purchases_train = []

	# populate the list with the product codes
	for i in tqdm(customers_train):
		temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
		purchases_train.append(temp)

		# train word2vec model
	model = Word2Vec(window = 10, sg = 1, hs = 0,
					negative = 10, # for negative sampling
					alpha=0.03, min_alpha=0.0007,
					seed = 14)

	model.build_vocab(purchases_train, progress_per=200)

	model.train(purchases_train, total_examples = model.corpus_count,
				epochs=10, report_delay=1)

	# save the model

	joblib.dump(model, 'word2vec.pkl')

model = joblib.load('word2vec.pkl')

products = train_df[["StockCode", "Description"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='StockCode', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()


with evaluation:
	st.markdown('> # Visualizing Results...')
	sel_col1, sel_col2 = st.beta_columns(2)

	option = sel_col1.selectbox(
	'Pick one from a random list of products',
	['LARGE CHINESE STYLE SCISSOR', 'PINK DOG BOWL', 'BLUE EGG  SPOON', 'VIP HOLIDAY PURSE', 'WOODLAND CHARLOTTE BAG', 'WOODLAND  STICKERS', 'LARGE SKULL WINDMILL', '12 COLOURED PARTY BALLOONS', 'CALENDAR PAPER CUT DESIGN'])

	n_option = sel_col2.slider('Choose the number of similar products to display', min_value = 2, max_value = 7, value = 5, step = 1)

	input_word = list(products_dict.keys())[list(products_dict.values()).index([option])]
	user_input = [x.strip() for x in input_word.split(',')]
	result_word = []

	for words in user_input:
		sim_words = model.wv.most_similar(words, topn = n_option)
		sim_words = append_list(sim_words, words)

		result_word.extend(sim_words)

	similar_word = [word[0] for word in result_word]
	similarity = [word[1] for word in result_word]
	similar_word.extend(user_input)
	labels = [word[2] for word in result_word]
	labelsf = [products_dict[i][0] for i in labels]
	similar_wordf = [products_dict[i][0] for i in similar_word]

	labels = labelsf
	#similar_word = similar_wordf
	label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
	color_map = [label_dict[x] for x in labels]

	display_tsne_scatterplot_3D(model, user_input , similar_word, labels, color_map, 0, 0, 250,topn=n_option, name = 'Similar Words')

	st.markdown('- As we can see, the recommendations for the products are very relatable for the selected product')


with evaluationfinal:
	st.markdown('> # Recommendations based on purchase history...')
	st.markdown('- The idea is to capture all the previous purchases info of a customer into a single point to suggest products')
	st.markdown("- Let's create a function to return the mean of vectors for purchased products")
	mean_code = """
def aggregate_vectors(products):
	product_vec = []
	for i in products:
		try:
			product_vec.append(model[i])
		except KeyError:
			continue

	return np.mean(product_vec, axis=0)
"""

	st.code(mean_code, language = 'python')
	sel_col1, sel_col2 = st.beta_columns(2)

	sel_col1.markdown('> *Select a random customer from the validation set*')
	val_cust = sel_col1.selectbox('Pick a customer', ['15799', '12747', '14224', '16217', '13925'])
	last_n = sel_col1.slider('Choose the number of recent purchases to be considered for recommendations',min_value = 2, max_value = 10, value = 10, step = 1)

	cust = df[df["CustomerID"] == int(val_cust)]["StockCode"].tolist()


	with sel_col1:
		st.markdown('> *Few of the recently purchased items by the selected customer are -*')
		st.write([products_dict[i][0] for i in cust[10:20]])

	mean_val = aggregate_vectors(cust[-last_n:])
	sim_words = model.wv.similar_by_vector(mean_val, topn = 5)

	similar_word = [sim_word[0] for sim_word in sim_words]
	labels  = [products_dict[i][0] for i in similar_word]

	label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
	color_map = [label_dict[x] for x in labels]

	with sel_col2:
		st.write('Top 5 recommendations based on the last ', last_n, 'products for the selected customer')
		display_tsne_scatterplot_3D(model, user_input , similar_word, labels, color_map, 0, 0, 250,topn=5, name = 'items')

	st.markdown('### As we can see, the recommendations provided to the customer are very close to their previous purchase history !!!')
