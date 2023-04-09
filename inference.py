import os
import urllib
import zipfile
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model_file import NNHybridFiltering
device = torch.device("cpu")
def load_data():
	df=pd.read_csv("reviews_airbnb.csv")
	return (df)
ratings=load_data()
X=ratings.drop(columns='ratings')   
def load_model(wt_file_path):
	# import fine tuned model
    X=load_data()
    n_users = X.loc[:,'reviewer_id'].max()+1
    n_items = X.loc[:,'id'].max()+1
	model = NNHybridFiltering(n_users,
                       n_items,
                       embdim_users=50, 
                       embdim_items=50,n_numerical_cols=16,
                       n_activations = 100,
                       rating_range=[0.,5.])
	model=model.to(device)
	model.load_state_dict(torch.load(wt_file_path))
	return (model)
    
def predict_rating(model,reviewer_id,id, device):
    # Get predicted rating for a specific user-item pair from model
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([reviewer_id,id]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
        return pred

def generate_recommendations(ratings1,X,model,reviewer_id,device):
    # Get predicted ratings for every listing
    pred_ratings = []
    for listing in ratings1['id'].tolist():
        pred = predict_rating(model,reviewer_id,listing,device)
        pred_ratings.append(pred.detach().cpu().item())
    # Sort listings by predicted rating
    idxs = np.argsort(np.array(pred_ratings))[::-1]
    recs = ratings1.iloc[idxs]['id'].values.tolist()
    # Filter out places the user has already stayed at
    places_stayed = X.loc[X['reviewer_id']==userId, 'id'].tolist()
    recs = [rec for rec in recs if not rec in places_stayed]
    # Filter to top 10 recommendations
    #recs = recs[:10]
    #new
    res = []
    [res.append(x) for x in recs if x not in res]
    res = res[:10]
    
    #recs=list(set(recs))[:10]
    # Convert listing ids to listing urls
    recs_names = []
    for rec in res:
        recs_names.append(ratings1.loc[ratings1['id']==rec,'listing_url'].values[0])
    return recs_names

if __name__ == "__main__":
	userId=1045
    path='best_model_weights.pth'
    model1=load_model(path)
    recs = generate_recommendations(ratings,X,model1,userId,device)
    for i,rec in enumerate(recs):
        print('Recommendation {}: {}'.format(i,rec))