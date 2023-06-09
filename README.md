# RecSys_Project
## Problem: 

Using Airbnb's review data published for each year since 2018, provide a list of recommendations for users wanting to rent the airbnb properties. For this project we have limited the scope to airbnb listings in Amsterdam.

## Data source and description:

1. The data was sourced from the InsideAirbnb site which holds the entire data repository for all the cities that Airbnb is operational in.
-http://insideairbnb.com/explore/
2. The 2 files used here are- 

a.reviews_detail.csv : which contains reviewer id, reviewer name, the listing id, the actual review comments in plain text 

b.listing_detail.csv: which contains specifics of the property, the amenities provided for the stay, the rating accuracies, host descriptions, listing url etc.

3. EDA showing the skewness of the data:

![image](https://user-images.githubusercontent.com/110474064/231043705-f9ab2d5a-051a-460e-88e8-455b9b98e914.png)

![image](https://user-images.githubusercontent.com/110474064/231039384-5b7c67eb-419b-4494-87e7-d330b052b9f7.png)

![image](https://user-images.githubusercontent.com/110474064/231039478-7803546d-fd0e-4d49-b234-11ac8f778727.png)


## Data manipulation:

1. The reviews_detail.csv file contains only the natural text comments for each stay by the user. There is no numerical rating that this has been translated to. So we used the NLTK library and specifically the vader-lexicon to get a general senitment of the comment and respective decided a range of numerical rating.
2. The vader-lexicon was used for the below reasons-

a. It provided a zero-shot way of translating the text comments into a usable numerical value using sentiment analyser function.

b. It evaluates using a "compound" score which takes into account the sarcasm in a comment and accurately returns the correct sentiment.

c. This library has 75K weighted words, which worked reasonably well for us as users providing feedback seldom do so using complicated vocabulary.

3. We then joined the reviews and the listings using the listing id to form the combined_fin.csv.
4. In the listing file, there was just 1 column inclusing all the amenities, so we separeted those out using regex and 1 hot encoded them. This lead to a very sparse dataset, which is why we ended up taking the top 10 amenities that each listing offered. This was majorly created for the hybrid model that we planned to test.

## Models and Evaluation:

Collaborative Filtering using NN: here we created reviewer and listing embeddings , used concatenation to retain info for both embeddings. This model perfomed the best in terms of train and validation loss and provided decent recommendations.

![collab](https://user-images.githubusercontent.com/110474064/231039068-d7ba26d8-07b4-4ba2-8966-91bb61da8f23.jpeg)

Hybrid Model: here we fed the amenities, rating accuracy, super host status, etc. to the model along with the reviewer and item embeddings. This model showed the most unstable training, took longer to train , took longer to infer and ended up providing nearly the same recommendations.
This is why, for the final deployment we went ahead with the first approach.

![hybrid](https://user-images.githubusercontent.com/110474064/231039106-22c17f89-a87e-4f32-8afd-4dfba628480e.jpeg)

Note: the inference takes time with CPU and thus the streamlit implementation running using the most basic infrastructure, can take a lot of time. So it is recommended that inference as well be done using GPUs. The reason for this is, the prediction is being generated for each listing available in the file (as the number of listings increase, the time taken for inference can increase if it is run using only the CPU)

## Results:

Though the model convergence and the train and validation losses look promising, the algorithm rarely provides distinct recommendations, which brings into question the coverage of the recommendation system built using the algorithm. A couple of reasons for this can be the highly skewed number of reviews for a particular popular listing and the highly skewed positive ratings wrt the negative ones. To make this more robust, we would need subject matter experts to define a set of heuristics to determine the intrinsic appeal, so that we can modify the data accordingly and retrain the model.

## References:

1. https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata :for data search

2. https://www.kaggle.com/code/erikbruin/airbnb-the-amsterdam-story-with-interactive-maps/input?select=listings_details.csv : for basic EDA and understanding the data

3. DCAR: Deep Collaborative Autoencoder for Recommendation with Implicit Feedback- Wang Jiong et. al : for the merits of the methodology chosen, concatenation vs dot product and the combination of the 2.

4. https://github.com/AIPI540/AIPI540-Deep-Learning-Applications by Prof. Jon Reifschneider : for code reference
