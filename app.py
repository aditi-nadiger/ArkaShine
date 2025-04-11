import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
feature_columns = ['A(410)', 'B(435)', 'C(460)', 'D(485)', 'E(510)', 'F(535)', 'G(560)',
                   'H(585)', 'R(610)', 'I(645)', 'S(680)', 'J(705)', 'U(760)',
                   'V(810)', 'W(860)', 'K(900)', 'L(940)', 'T(730)']
target_columns = ['pH', 'EC  (dS/m)','OC (%)','P   (kg/ha)', 'K (kg/ha)', 'Ca (meq/100g)', 'Mg (meq/100g)',
             'S (ppm)', 'Fe (ppm)', 'Mn (ppm)', 'Cu (ppm)', 'Zn (ppm)', 'B (ppm)']

st.title("Soil Nutrients Prediction")
st.image(r"C:\Users\Aditi\Documents\ArkaShine\Premium Photo _ Seedlings are growing from fertile soil, environmental concepts_.jpg",use_column_width="auto")
# Load the saved model and scaler
@st.cache_data
def load_model():
    with open('rfc.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Sidebar
st.sidebar.header("Please enter the light wavelengths 📊")

# User inputs
A = st.sidebar.number_input("A", min_value=0.0)
B = st.sidebar.number_input("B", min_value=0.0)
C = st.sidebar.number_input("C", min_value=0.0)
D = st.sidebar.number_input("D", min_value=0.0)
E = st.sidebar.number_input("E", min_value=0.0)
F = st.sidebar.number_input("F", min_value=0.0)
G = st.sidebar.number_input("G", min_value=0.0)
H = st.sidebar.number_input("H", min_value=0.0)
R = st.sidebar.number_input("R", min_value=0.0)
I = st.sidebar.number_input("I", min_value=0.0)
S = st.sidebar.number_input("S", min_value=0.0)
J = st.sidebar.number_input("J", min_value=0.0)
U = st.sidebar.number_input("U", min_value=0.0)
V = st.sidebar.number_input("V", min_value=0.0)
W = st.sidebar.number_input("W", min_value=0.0)
K = st.sidebar.number_input("K", min_value=0.0)
L = st.sidebar.number_input("L", min_value=0.0)
T = st.sidebar.number_input("T", min_value=0.0)

# Create input data dictionary
data = {
    'A(410)': [A],
    'B(435)': [B],
    'C(460)': [C],
    'D(485)': [D],
    'E(510)': [E],
    'F(535)': [F],
    'G(560)': [G],
    'H(585)': [H],
    'R(610)': [R],
    'I(645)': [I],
    'S(680)': [S],
    'J(705)': [J],
    'U(760)': [U],
    'V(810)': [V],
    'W(860)': [W],
    'K(900)': [K],
    'L(940)': [L],
    'T(730)': [T]
}

def preprocess_input_data(input_data):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame(input_data)
    return input_df

# Load model and scaler
model, scaler = load_model()

# Make prediction when the button is clicked
if st.button("Predict"):
    df = preprocess_input_data(data)
    scaled_input = scaler.transform(df)
    prediction = model.predict(scaled_input)

    st.subheader("Prediction")
    st.write("Predicted Nutrient Levels:")
    predicted_values = {}
    for i, col in enumerate(target_columns):
        st.write(f'{col}: {prediction[0][i]:.4f}')
        predicted_values[col] = prediction[0][i]

    
    #plotting bar graph to represent the predicted nutrient 
    colors = list(mcolors.TABLEAU_COLORS.values())
    fig, ax = plt.subplots()
    bars = ax.barh(list(predicted_values.keys()), list(predicted_values.values()), color=colors[:len(predicted_values)])
    ax.set_xlabel('Predicted Levels')
    ax.set_title('Predicted Nutrient Levels')

    # Adding color to each bar and displaying the value
    for bar, value in zip(bars, predicted_values.values()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{value:.2f}', va='center')

    st.pyplot(fig)



    # Plot for actual vs predicted values
    df2=pd.read_csv(r"C:\Users\Aditi\Documents\ArkaShine\values.csv")
    X = df2[feature_columns]
    y = df2[target_columns]
    xts = scaler.fit_transform(X) 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(xts, y, test_size=0.2, random_state=42)
    y_pred_rfr = model.predict(X_test)

    plt.figure(figsize=(12, 8))

    # Bar width for both actual and predicted values
    bar_width = 0.35

    # Indices for nutrient labels
    indices = np.arange(len(target_columns))


    # Plot actual values
    actual_bars = plt.barh(indices - bar_width/2, y_test.mean(), bar_width, label='Actual Values', color='lightskyblue')
    
    # Plot predicted values
    predicted_bars = plt.barh(indices + bar_width/2, [predicted_values[col] for col in target_columns], bar_width, label='Predicted Values', color='blue')

    # Adding nutrient labels
    plt.yticks(indices, target_columns)

    # Adding labels and title
    plt.xlabel('Values')
    plt.title('Actual vs Predicted Nutrient Values')
    plt.legend()

    # Adding values on the bars
    for actual_bar, predicted_bar in zip(actual_bars, predicted_bars):
        plt.text(actual_bar.get_width(), actual_bar.get_y() + actual_bar.get_height() / 2, f'{actual_bar.get_width():.2f}', va='center')
        plt.text(predicted_bar.get_width(), predicted_bar.get_y() + predicted_bar.get_height() / 2, f'{predicted_bar.get_width():.2f}', va='center')

    st.pyplot(plt)