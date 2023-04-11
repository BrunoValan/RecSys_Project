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
            page_title="Rec++: AirBnB Recommendations",
            page_icon="✈️",
            layout="wide")

    st.title("Amsterdam AirBnB Recommendations")
    st.markdown("""---""")

    st.sidebar.title('Rec++: A Personalized Airbnb Recomendation System')
    st.sidebar.write('Our goal is to further personalize Airbnbs recommendations using a combination of NLP and RecSys algorithms')
    st.sidebar.write('Presented By: Neha Barde, Andrew Bonafede, and Bruno Valan')

    ratings = populate()
    X = ratings.loc[:,['reviewer_id','id']]

    userID = st.number_input('Please Input your UserID',value = 0)
    if userID != 0:
        device = torch.device("cpu")
        path = 'best_model_weights.pth'
        model1 = load_model(ratings,path,device)
        recs = generate_recommendations(ratings,X,model1,userID,device)
        for i,rec in enumerate(recs):
            st.write('Recommendation {}: {}'.format(i+1,rec))

if __name__ == "__main__":
    run_ui()