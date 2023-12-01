import tkinter as tk
from tkinter import ttk
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fetch historical data
stocks = ["WMT", "TGT", "COST"]
data = yf.download(stocks, start="2020-01-01", end="2023-01-01")
close_prices = data['Close']

# Calculate daily returns
close_prices = close_prices.pct_change().dropna() * 100  # Convert returns to percentage

# Define a function to create and train a linear regression model for a stock
def train_model(stock_returns):
    X = np.array(range(len(stock_returns))).reshape(-1, 1)  # Days as independent variable
    y = stock_returns.values  # Stock returns as dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X, y

# Training models for each stock
models = {}
for stock in stocks:
    model, X, y = train_model(close_prices[stock].dropna())
    models[stock] = (model, X, y)

# Generating future dates (e.g., next 30 days)
last_date = close_prices.index[-1]
future_dates = pd.date_range(start=last_date, periods=31, closed='right')
start_date = close_prices.index[0]
days_since_start = (future_dates - start_date).days

# Function to update the plot based on selected stock
def update_plot():
    selected_stock = stock_var.get()
    model, X, y = models[selected_stock]
    predicted_returns = model.predict(X)
    future_returns = model.predict(days_since_start.reshape(-1, 1))

    # Clear the previous plot
    ax.clear()

    # Plotting the actual and predicted returns
    ax.plot(close_prices[selected_stock].index, y, label=f'{selected_stock} Actual Returns')
    ax.plot(future_dates, future_returns, label=f'{selected_stock} Future Predicted Returns', linestyle='dashed')

    ax.set_title(f'Return Prediction for {selected_stock}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.legend()
    canvas.draw()

# Create the main window
root = tk.Tk()
root.title("Stock Return Predictor")

# Dropdown menu for stock selection
stock_var = tk.StringVar()
stock_dropdown = ttk.Combobox(root, textvariable=stock_var, values=stocks)
stock_dropdown.grid(column=0, row=0)
stock_dropdown.current(0)  # Set the default value to the first stock

# Button to update the plot
update_button = tk.Button(root, text="Update Plot", command=update_plot)
update_button.grid(column=1, row=0)

# Plotting area
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)  # Embedding the plot in the Tkinter window
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(column=0, row=1, columnspan=2)

root.mainloop()
