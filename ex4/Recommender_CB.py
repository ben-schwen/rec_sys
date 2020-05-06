import pandas as pd
import numpy as np
from scipy import sparse as sp
import sklearn
import sklearn.preprocessing as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Recommender_CB:

    def __init__(self, profile_type='plot'):
        self.profile_type = profile_type

    def get_item_titles(self, item_ids):
        return [ self.items_meta[self.items_meta['item_id'] == id]['title'].item() for id in item_ids]


    def build_item_contents(self):

        vectorizer = TfidfVectorizer(stop_words='english') # Define a TF-IDF Vectorizer that removes all english stop words (e.g., 'the', 'a')

        ########## BEGIN HERE ##########
        self.plot_tfidf = vectorizer.fit_transform(self.items_meta['plot'])
        self.plot_tfidf_tokens = vectorizer.get_feature_names()

        self.meta_tfidf = vectorizer.fit_transform(self.items_meta['metadata'])
        self.meta_tfidf_tokens = vectorizer.get_feature_names()
        ##########  END HERE  ##########

        self.set_content_type()


    def set_content_type(self):
        if self.profile_type == 'plot':
            self.tfidf = self.plot_tfidf
            self.tfidf_tokens = self.plot_tfidf_tokens
        else:
            self.tfidf = self.meta_tfidf
            self.tfidf_tokens = self.meta_tfidf_tokens


    def get_item_vectors(self, item_ids):
        i_ids = [self.i_id_to_item_id.index(item_id) for item_id in item_ids]
        item_vector = self.tfidf[i_ids]
        return item_vector


    def get_user_profile(self, user_id, ratings):
        item_ids_rated_by_user_id = np.array( ratings.loc[ ratings['user'] == user_id ]['item'] )
        user_ratings = np.array( ratings.loc[ ratings['user'] == user_id ]['rating'] )

        ########## BEGIN HERE ##########

        item_vecs = self.get_item_vectors(item_ids_rated_by_user_id)
        weights = user_ratings / user_ratings.sum()

        # without matrix multiplication
        # row_lengths = item_vecs.indptr[1:] - item_vecs.indptr[:-1]
        # item_vecs.data *= np.repeat(weights, row_lengths)

        # with matrix multiplication
        item_vecs = sp.csr_matrix(item_vecs).multiply(sp.csr_matrix(weights).T)

        user_profile = item_vecs.sum(axis=0)
        ##########  END HERE  ##########

        user_profile = pp.normalize(user_profile)

        return user_profile



    def build_user_profiles(self):
        positive_ratings = self.ratings[self.ratings['rating']>3]
        self.user_profiles = {}
        for user_id in positive_ratings['user'].unique():
            self.user_profiles[user_id] = self.get_user_profile(user_id, positive_ratings)



    def recommend(self, user_id, from_item_ids=None, topN=20):
        item_ids_rated_by_user_id = self.ratings.loc[ self.ratings['user'] == user_id ]['item'].tolist()


        if from_item_ids is None:
            from_item_ids = self.item_ids

        ########## BEGIN HERE ##########
        # step 1- retrieve user profile
        user_profile = self.user_profiles.get(user_id)

        # step 2 compute cosine similarity
        # sims = linear_kernel(user_profile, self.tfidf)
        sims = linear_kernel(user_profile, self.get_item_vectors(from_item_ids))

        # step 3
        index = np.argsort(sims, axis = None)[::-1]

        # step 4
        # items_sorted_sim = np.array(self.items_meta['item_id'])[index]
        items_sorted_sim = np.array(from_item_ids)[index]

        # step 5
        # recommendations = [x for x in items_sorted_sim if x in from_item_ids and x not in item_ids_rated_by_user_id]
        recommendations = [x for x in items_sorted_sim if x not in item_ids_rated_by_user_id]

        # step 6
        recommendations = recommendations[:topN]
        ##########  END HERE  ##########

        return recommendations




    def build_model(self, ratings, items_meta):
        self.ratings = ratings
        self.items_meta = items_meta

        ## user_id and item_id are external ids; i_id is internal id
        self.item_ids = self.ratings.item.unique()
        self.item_ids.sort()

        self.user_ids = self.ratings.user.unique()
        self.user_ids.sort()

        self.i_id_to_item_id = self.items_meta['item_id'].tolist()

        self.build_item_contents()
        self.build_user_profiles()

