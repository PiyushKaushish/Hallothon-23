import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from datetime import datetime

def time_elapsed(start_time, end_time):
    if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
        return None
    elapsed_time = end_time - start_time
    return elapsed_time.days  # Return the number of days

# Fetch historical stock price data for Apple
apple = yf.download('AAPL', start='2009-01-01')

# Current price / first record (e.g. price at the beginning of 2009)
# provides us with the total growth %
total_growth = apple['Adj Close'][-1] / apple['Adj Close'][0]

# Next, we want to annualize this percentage
# First, we convert our time elapsed to the number of years elapsed
number_of_days = time_elapsed(apple.index[0], apple.index[-1])
number_of_years = number_of_days / 365.0  # Assuming 365 days per year

# Second, we can raise the total growth to the inverse of the number of years
# (e.g., ~1/10 at the time of writing) to annualize our growth rate
cagr = (total_growth ** (1/number_of_years)) - 1

# Now that we have the mean annual growth rate above,
# we'll also need to calculate the standard deviation of the
# daily price changes
std_dev = apple['Adj Close'].pct_change().std()

# Next, because there are roughly ~252 trading days in a year,
# we'll need to scale this by an annualization factor
# reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx

number_of_trading_days = 252
std_dev = std_dev * math.sqrt(number_of_trading_days)

# Calculate daily return percentages for 1 year's worth of trading (252 days)
daily_return_percentages = np.random.normal(cagr / number_of_trading_days, std_dev / math.sqrt(number_of_trading_days), number_of_trading_days) + 1

# Now that we have created a random series of future daily return percentages,
# we can apply these forward-looking to our last stock price in the window,
# effectively carrying forward a price prediction for the next year

# This distribution is known as a 'random walk'
price_series = [apple['Adj Close'][-1]]

for j in daily_return_percentages:
    price_series.append(price_series[-1] * j)

# Plot a single 'random walk' of stock prices
plt.plot(price_series)
plt.show()

# Now that we've created a single random walk above,
# we can simulate this process over a large sample size to
# get a better sense of the true expected distribution
number_of_trials = 3000

# Set up an additional array to collect all possible
# closing prices on the last day of the window.
# We can use this for a histogram to get a clearer sense of possible outcomes
closing_prices = []

for i in range(number_of_trials):
    # Calculate randomized return percentages following our normal distribution
    # and using the mean / std dev we calculated above
    daily_return_percentages = np.random.normal(cagr / number_of_trading_days, std_dev / math.sqrt(number_of_trading_days), number_of_trading_days) + 1
    price_series = [apple['Adj Close'][-1]]

    for j in daily_return_percentages:
        # Extrapolate the price out for the next year
        price_series.append(price_series[-1] * j)

    # Append closing prices on the last day of the window for the histogram
    closing_prices.append(price_series[-1])

    # Plot all random walks
    plt.plot(price_series)

plt.show()

# Plot histogram
plt.hist(closing_prices, bins=40)
plt.show()

# Calculate the mean of all ending prices, allowing us to arrive at the most probable ending point
mean_end_price = round(np.mean(closing_prices), 2)
print("Expected price: ", str(mean_end_price))

# Lastly, split the distribution into percentiles to help gauge risk vs. reward

# Pull the top 10% of possible outcomes
top_ten = np.percentile(closing_prices, 100 - 10)

# Pull the bottom 10% of possible outcomes
bottom_ten = np.percentile(closing_prices, 10)

# Create the histogram again
plt.hist(closing_prices, bins=40)
# Append with top 10% line
plt.axvline(top_ten, color='r', linestyle='dashed', linewidth=2)
# Append with bottom 10% line
plt.axvline(bottom_ten, color='r', linestyle='dashed', linewidth=2)
# Append with the current price
plt.axvline(apple['Adj Close'][-1], color='g', linestyle='dashed', linewidth=2)
plt.show()

# Store the data in a pickle file
data_to_pickle = {
    'apple_data': apple,
    'closing_prices': closing_prices,
    'mean_end_price': mean_end_price,
    'top_ten_percentile': top_ten,
    'bottom_ten_percentile': bottom_ten
}

pickle_file_path = 'stock_simulation_data.pkl'

with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(data_to_pickle, pickle_file)

print(f'Data has been saved to {pickle_file_path}')
