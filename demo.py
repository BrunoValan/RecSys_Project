import streamlit as st 
from inference import *

loaded_data = None
def populate():
    global loaded_data
    if loaded_data is None:
        # Load the data from the file
        loaded_data = load_data()
    return loaded_data

def run_ui():
    st.set_page_config(
            page_title="Amsterdam AirBnB Recommendations",
            page_icon="✈️",
            layout="wide")

    st.title("Amsterdam AirBnB Recommendations")
    st.markdown("""---""")
    ratings = populate()
    X = ratings.loc[:,['reviewer_id','id']]

    userId= st.number_input('Please Input your UserID')
    path = 'best_model_weights.pth'
    model1 = load_model(ratings,path)
    recs = generate_recommendations(ratings,X,model1,userId,device)
    for i,rec in enumerate(recs):
        st.write('Recommendation {}: {}'.format(i,rec))  

if __name__ == "__main__":
    run_ui()