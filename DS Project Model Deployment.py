import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer



# Load data
df = pd.read_excel(r"C:/Users/DELL/Desktop/Data Science Project DS_P_241/World_development_mesurement.xlsx")
df1=df.copy()
# Creating a function to handle string characters and convert the non numeric into float
def Stringfunction(x):
    if isinstance(x, str):
        x = x.replace('$', '')
        x = x.replace(',', '')
        x = x.replace('%', '')
        x = float(x)
    elif isinstance(x, float):
        pass  # no need to do anything if it's already a float
    else:
        try:
            x = x.replace('$', '')
            x = x.replace(',', '')
            x = x.replace('%', '')
            x = float(x)
        except:
            pass
    return x
df=df.drop('Country', axis=1)
df = df.applymap(Stringfunction) # Applymap aplies function to each element of the dataframe
df['Country']=df1['Country']
# Dropping unnecessary columns
df = df.drop(['Number of Records', 'Ease of Business'], axis=1)

# Handling missing values
imputer = KNNImputer(n_neighbors=3)
df_impute = df.drop('Country', axis=1)
imputed = imputer.fit_transform(df_impute)
df_imputed = pd.DataFrame(imputed, columns=df_impute.columns)

# Dropping features with high missing values, unnecessary features
df_imputed = df_imputed.drop(['Business Tax Rate', 'Hours to do Tax', 'Days to Start Business','Lending Interest','Health Exp/Capita'], axis=1)
df_imputed['Country']=df1['Country']
# Handling outliers using IQR
for col in df_imputed.columns:
    if col != 'Country':
        Q1 = df_imputed[col].quantile(0.25)
        Q3 = df_imputed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_imputed[col] = np.where(df_imputed[col] < lower_bound, lower_bound, df_imputed[col])
        df_imputed[col] = np.where(df_imputed[col] > upper_bound, upper_bound, df_imputed[col])

# Scaling data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_imputed.drop('Country', axis=1))

# PCA for dimensionality reduction
pca = PCA(n_components=4) 
pca_values = pca.fit_transform(scaled_data)
pca_data = pd.DataFrame(pca_values, columns=['pc1', 'pc2', 'pc3', 'pc4'])
pca_data=np.array(pca_data)
# Hierarchical Clustering
kmeans_pca = KMeans(n_clusters=3,random_state=0)
kmeans_pca.fit(pca_data)


# Assigning labels to the data
labels = kmeans_pca.labels_
df['Cluster'] = labels
print(df['Cluster'])

model = {'scaler': scaler, 'pca': pca, 'kmean': kmeans_pca}
with open('trained_model_clustering.pkl', 'wb') as f:
    pickle.dump(model, f)

import streamlit as st 
st.title('Cluster Prediction Model For Global Development Measurement Dataset')

import pickle
import numpy as np

with open(r"/content/trained_model_clustering.pkl", "rb") as f:
    loaded_model = pickle.load(f, encoding="utf-8")

scaler = loaded_model['scaler']
pca = loaded_model['pca']
kmean = loaded_model['kmean']
# write the function for scaling and pca
def prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
    input_scaled = scaler.transform(input_data_reshaped)
    
    input_pca = pca.transform(input_scaled)
        
    predict =kmean.predict(input_pca)
     
            
    if predict[0] == 2:
        return 'Developed country'
    elif predict[0] == 1:
        return 'Under Developed Country'
    else:
        return 'Developing Country'

# Define the Streamlit app
def main():
    # Getting inout from user
    Country = st.sidebar.text_input('Enter Country Name')
    Birth_Rate = st.sidebar.number_input('Enter the Birth Rate ')
    CO2_Emissions = st.sidebar.number_input('Enter the CO2 Emissions ')
    Energy_Usage = st.sidebar.number_input('Enter the Energy Usage')
    GDP = st.sidebar.number_input('Enter the GDP')
    Health_ExpGDP= st.sidebar.number_input('Enter the Health Expenditure % on GDP')
    Infant_Mortality_Rate = st.sidebar.number_input("Enter the Infant Mortality Rate (0-1)",min_value=0.0,max_value=1.0,step=0.001)
    Internet_Usage = st.sidebar.number_input("Enter the Internet Usage  (0-1)",min_value=0.0,max_value=1.0,step=0.001)
    LifeExpFemale = st.sidebar.number_input('Enter the Life Expectancy of Female')
    LifeExpMale = st.sidebar.number_input('Enter the Life Expectancy of Male')
    MobilePhoneUsage = st.sidebar.number_input("Enter the Mobile Phone Usage (0-1)",min_value=0.0,max_value=1.0,step=0.01)
    Population_0to14 = st.sidebar.number_input("Enter the Value of % Population of Age between 0-14",min_value=0.0,max_value=1.0,step=0.01)
    Population_15to64 = st.sidebar.number_input("Enter the value of % Population of Age between 15-64",min_value=0.0,max_value=1.0,step=0.01)
    Population_65plus = st.sidebar.number_input("Enter the value of % Population of Age above 65",min_value=0.0,max_value=1.0,step=0.01)
    Population_Total = st.sidebar.number_input("Enter the Total Population Count")
    UrbanPopulation = st.sidebar.number_input("Enter the Value of % Population in Urban")
    Tourism_inbound = st.sidebar.number_input("Enter the Tourism Inbound")
    Tourism_outbound = st.sidebar.number_input("Enter the Tourism Outbound")

    if st.button('Clustering Prediction Result'):
       pred = prediction([Birth_Rate, CO2_Emissions,Energy_Usage,GDP,Health_ExpGDP,Infant_Mortality_Rate,Internet_Usage,LifeExpFemale,LifeExpMale,MobilePhoneUsage,Population_0to14,Population_15to64,Population_65plus,Population_Total,UrbanPopulation,Tourism_inbound,Tourism_outbound])
       st.success(pred)
    
# Run the app
if __name__ == '__main__':
    main()