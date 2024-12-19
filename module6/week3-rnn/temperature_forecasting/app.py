import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import pandas as pd

class WeatherForecastor(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, dropout_prob):
        super(WeatherForecastor, self).__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Load the model
embedding_dim = 1
hidden_size = 8
n_layers = 3
dropout_prob = 0.2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#The Device must be match with the type of device you train model!
#For example: if you use cuda to train model and save as model.pth, then you must use cuda device.
device = 'cpu'

model = WeatherForecastor(
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout_prob=dropout_prob
).to(device)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title('Hourly Temperature Forecasting')
example = [24.86,22.75,20.07,17.81,17.16,15.01]
target = 14.47
st.write("Example:", ", ".join(map(str, example)), "Target:", target)


# User input for temperature data
temp_data = st.text_input("Enter 6 temperatures separated by commas:", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
temp_data = [float(temp) for temp in temp_data.split(",")]

# Convert user input to tensor with the required shape
temp_tensor = torch.FloatTensor(temp_data).unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions

# Predict button
if st.button('Predict'):
    # Make prediction
    with torch.no_grad():
        prediction = model(temp_tensor)
        predicted_temp = prediction.item()

    st.markdown("## Predicted Temperature for the Next Hour:")
    st.write(f"Predicted Temperature: {predicted_temp:.2f}°C")