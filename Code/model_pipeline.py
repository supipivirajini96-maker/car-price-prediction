import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("car-prediction.model", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
# Load defaults for missing values
defaults = pickle.load(open("defaults.pkl", "rb"))

def predict_car_price(sample: np.ndarray) -> float:
    
    # Scale the user input
    #sample_scaled = scaler.transform(sample)

    # Predict the car price
    predicted_price = model.predict(sample)

    # Undo log transform 
    predicted_price = np.exp(predicted_price)

    return float(predicted_price[0])



def fill_missing_values(transmission, max_power):
    # Handle missing categorical values of transmission
    if transmission is None:
        probs = defaults["transmission_ratio"]
        transmission = np.random.choice(list(probs.keys()), p=list(probs.values()))

    # Handle missing numeric values of max_power
    if max_power is None:
        max_power = defaults["mean_max_power"]  

    return transmission, max_power

def predict_car_price(sample: np.ndarray) -> float:
    # Predict
    predicted_price = model.predict(sample)
    predicted_price = np.exp(predicted_price)  # undo log transform
    return float(predicted_price[0])


# Example
if __name__ == "__main__":
    # Replace with actual user input
    #sample = np.array([[1, 67.05, 998.00, 2009.00]])  # example
    print("Predicted Car Price:", predict_car_price(sample))
