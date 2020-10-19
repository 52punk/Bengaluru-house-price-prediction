# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:14:48 2020


@author: Pankaj Kumar Sah
@LinkedIn: https://www.linkedin.com/in/pankaj-sah-b7aa39186/
@Github: https://github.com/52punk


"""
import numpy as np
import pandas as pd

import streamlit as st
import pickle

X=pd.read_csv("X_save.csv")

pickle_in = open("Bengaluru_house_model.picle","rb")
regressor=pickle.load(pickle_in)


def predict_values(location,sqft,bath,bhk):
    try:
        loc_index=X.index(location)
    except:
        loc_index=-1
    #loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return regressor.predict([x])[0]



def price_predict(location,sqft,bath,bhk):
    prediction=predict_values(location,sqft,bath,bhk)
    return(prediction)



def temp():
    location = st.selectbox(
            'Enter the location',
            ('1st Phase JP Nagar', '5th Phase JP Nagar', '7th Phase JP Nagar',
       '8th Phase JP Nagar', '9th Phase JP Nagar', 'AECS Layout',
       'Abbigere', 'Akshaya Nagar', 'Ambalipura', 'Ambedkar Nagar',
       'Amruthahalli', 'Anandapura', 'Ardendale', 'Arekere', 'Attibele',
       'BTM 2nd Stage', 'BTM Layout', 'Badavala Nagar', 'Balagere',
       'Banashankari', 'Banashankari Stage III', 'Bannerghatta Road',
       'Battarahalli', 'Begur', 'Begur Road', 'Bellandur',
       'Bharathi Nagar', 'Bhoganhalli', 'Billekahalli', 'Binny Pete',
       'Bisuvanahalli', 'Bommanahalli', 'Bommasandra', 'Bommenahalli',
       'Brookefield', 'Budigere', 'CV Raman Nagar', 'Chandapura',
       'Channasandra', 'Chikka Tirupathi', 'Choodasandra', 'Dairy Circle',
       'Dasanapura', 'Dasarahalli', 'Devanahalli', 'Dodda Nekkundi',
       'Doddathoguru', 'Domlur', 'EPIP Zone', 'Electronic City',
       'Electronic City Phase II', 'Electronics City Phase 1',
       'Frazer Town', 'Gollarapalya Hosahalli', 'Gottigere',
       'Green Glen Layout', 'Gubbalala', 'Gunjur', 'HSR Layout',
       'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura',
       'Hegde Nagar', 'Hennur', 'Hennur Road', 'Hoodi', 'Horamavu Agara',
       'Hormavu', 'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road',
       'Hulimavu', 'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur',
       'Jalahalli', 'Jalahalli East', 'Jigani', 'KR Puram', 'Kadugodi',
       'Kaggadasapura', 'Kaggalipura', 'Kalena Agrahara', 'Kalyan nagar',
       'Kambipura', 'Kammasandra', 'Kanakapura', 'Kanakpura Road',
       'Kannamangala', 'Karuna Nagar', 'Kasavanhalli', 'Kenchenahalli',
       'Kengeri', 'Kengeri Satellite Town', 'Kereguddadahalli',
       'Kodichikkanahalli', 'Kodihalli', 'Koramangala', 'Kothanur',
       'Kudlu', 'Kudlu Gate', 'Kundalahalli', 'Lakshminarayana Pura',
       'Lingadheeranahalli', 'Magadi Road', 'Mahadevpura', 'Mallasandra',
       'Malleshwaram', 'Marathahalli', 'Marsur', 'Munnekollal',
       'Mysore Road', 'Nagarbhavi', 'Nagavarapalya', 'Neeladri Nagar',
       'Old Airport Road', 'Old Madras Road', 'Padmanabhanagar',
       'Panathur', 'Parappana Agrahara', 'Pattandur Agrahara',
       'Prithvi Layout', 'Rachenahalli', 'Raja Rajeshwari Nagar',
       'Rajaji Nagar', 'Rajiv Nagar', 'Ramagondanahalli',
       'Ramamurthy Nagar', 'Rayasandra', 'Sahakara Nagar', 'Sarjapur',
       'Sarjapur  Road', 'Sector 2 HSR Layout', 'Seegehalli',
       'Somasundara Palya', 'Sonnenahalli', 'Subramanyapura',
       'Talaghattapura', 'Thanisandra', 'Thigalarapalya', 'Thubarahalli',
       'Tumkur Road', 'Uttarahalli', 'Varthur', 'Vidyaranyapura',
       'Vijayanagar', 'Vittasandra', 'Whitefield', 'Yelahanka',
       'Yelahanka New Town', 'Yeshwanthpur'))
    sqft = st.slider("Choose the area ( in square feet ) ",min_value=300.0,max_value=14000.0,step=0.1)
    bath = st.slider("Choose the number of bathrooms you want ",min_value=1,max_value=4,step=1)
    bhk = st.slider("Choose the bhk size",min_value=1,max_value=11,step=1)
    st.write("You chosed "+location+" for "+str(bhk)+" BHK with "+str(bath)+" bathrooms with area of "+str(sqft)+" Squarefeet")
    if st.button("Predict the price"):
        price=price_predict(location,sqft,bath,bhk)
        st.write("Predicted Price is "+str(price)+" lakhs")

def main():
    st.title("BENGALURU HOUSE PRICE PREDICTION GATEWAY")
    temp()

if __name__=="__main__":
    main()