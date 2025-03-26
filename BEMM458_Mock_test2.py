#######################################################################################################################################################
# 
# Name: SYED MOHAMMED ARSH
# SID:
# Exam Date: 27TH MARCH 2025
# Module: PROGRAMMING FOR BUSINESS ANALYTICS
# Github link for this assignment:  
#
#######################################################################################################################################################
# Instruction 1. Read each question carefully and complete the scripts as instructed.

# Instruction 2. Only ethical and minimal use of AI is allowed. You may use AI to get advice on tool usage or language syntax, 
#                but not to generate code. Clearly indicate how and where you used AI.

# Instruction 3. Include comments explaining the logic of your code and the output as a comment below the code.

# Instruction 4. Commit to Git and upload to ELE once you finish.

#######################################################################################################################################################

# Question 1 - Loops and Lists
# You are given a list of numbers representing weekly sales in units.
weekly_sales = [120, 85, 100, 90, 110, 95, 130]

# Write a for loop that iterates through the list and prints whether each week's sales were above or below the average sales for the period.
# Calculate and print the average sales.
# Given list of weekly sales

weekly_sales = [120, 85, 100, 90, 110, 95, 130]
average_sales = sum(weekly_sales) / len(weekly_sales)
print(f"Average sales: {average_sales:.2f}")

for i, sales in enumerate(weekly_sales, start=1):
    if sales > average_sales:
        print(f"Week {i}: {sales} units - Above average")
    else:
        print(f"Week {i}: {sales} units - Below average")


#######################################################################################################################################################

# Question 2 - String Manipulation
# A customer feedback string is provided:
customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# Find the first and last occurrence of the words 'good' and 'improved' in the feedback using string methods.
# Store each position in a list as a tuple (start, end) for both words and print the list.
# Customer feedback string
customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""


good_start_index = customer_feedback.find("good")  
good_end_index = good_start_index + len("good")    

improved_start_index = customer_feedback.find("improved")  
improved_end_index = improved_start_index + len("improved")

word_positions = [
    ("good", (good_start_index, good_end_index)),
    ("improved", (improved_start_index, improved_end_index))
]
print("Word positions in the feedback:")
for word, position in word_positions:
    print(f"'{word}' found at positions {position}")

#######################################################################################################################################################

# Question 3 - Functions for Business Metrics
# Define functions to calculate the following metrics, and call each function with sample values (use your student ID digits for customization).

# 1. Net Profit Margin: Calculate as (Net Profit / Revenue) * 100.
# 2. Customer Acquisition Cost (CAC): Calculate as (Total Marketing Cost / New Customers Acquired).
# 3. Net Promoter Score (NPS): Calculate as (Promoters - Detractors) / Total Respondents * 100.
# 4. Return on Investment (ROI): Calculate as (Net Gain from Investment / Investment Cost) * 100.

def net_profit_margin(net_profit, revenue):
    if revenue == 0:
        return "Revenue cannot be zero."
    return (net_profit / revenue) * 100

def customer_acquisition_cost(marketing_cost, new_customers):
    if new_customers == 0:
        return "New customers cannot be zero."
    return marketing_cost / new_customers


def net_promoter_score(promoters, detractors, total_respondents):
    if total_respondents == 0:
        return "Total respondents cannot be zero."
    return ((promoters - detractors) / total_respondents) * 100


def return_on_investment(net_gain, investment_cost):
    if investment_cost == 0:
        return "Investment cost cannot be zero."
    return (net_gain / investment_cost) * 100


net_profit = 5000  
revenue = 20000  
marketing_cost = 3000  
new_customers = 50  
promoters = 80  
detractors = 20  
total_respondents = 150  
net_gain = 7000  
investment_cost = 5000  

# Calling the functions and printing results
print(f"Net Profit Margin: {net_profit_margin(net_profit, revenue):.2f}%")
print(f"Customer Acquisition Cost (CAC): ${customer_acquisition_cost(marketing_cost, new_customers):.2f}")
print(f"Net Promoter Score (NPS): {net_promoter_score(promoters, detractors, total_respondents):.2f}%")
print(f"Return on Investment (ROI): {return_on_investment(net_gain, investment_cost):.2f}%")

#######################################################################################################################################################

# Question 4 - Data Analysis with Pandas
# Using a dictionary sales_data, create a DataFrame from this dictionary, and display the DataFrame.
# Write code to calculate and print the cumulative monthly sales up to each month.
import pandas as pd

sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Sales': [200, 220, 210, 240, 250]}
import pandas as pd


sales_data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [200, 220, 210, 240, 250]
}
df = pd.DataFrame(sales_data)
df['Cumulative Sales'] = df['Sales'].cumsum()

print(" Monthly Sales Data with Cumulative Sales ")
print(df)


#######################################################################################################################################################

# Question 5 - Linear Regression for Forecasting
# Using the dataset below, create a linear regression model to predict the demand for given prices.
# Predict the demand if the company sets the price at £26. Show a scatter plot of the data points and plot the regression line.

# Price (£): 15, 18, 20, 22, 25, 27, 30
# Demand (Units): 200, 180, 170, 160, 150, 140, 130
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


prices = np.array([15, 18, 20, 22, 25, 27, 30]).reshape(-1, 1)  
demand = np.array([200, 180, 170, 160, 150, 140, 130])


model = LinearRegression()
model.fit(prices, demand)

predicted_demand = model.predict(np.array([[26]]))[0]

price_range = np.linspace(15, 30, 100).reshape(-1, 1)  
predicted_demand_range = model.predict(price_range)

plt.scatter(prices, demand, color='blue', label="Actual Data")  # Scatter plot of actual data
plt.plot(price_range, predicted_demand_range, color='red', linestyle='--', label="Regression Line")  # Regression line
plt.scatter(26, predicted_demand, color='green', marker='x', s=100, label=f"Prediction at £26: {predicted_demand:.1f} units")

plt.xlabel("Price (£)")
plt.ylabel("Demand (Units)")
plt.title("Price vs Demand - Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
print(f" Predicted demand at £26: {predicted_demand:.1f} units")

#######################################################################################################################################################

# Question 6 - Error Handling
# You are given a dictionary of prices for different products.
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

# Write a function to calculate the total price of all items, handling any non-numeric values by skipping them.
# Include error handling in your function and explain where and why it’s needed.
# Given dictionary of product prices
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

def calculate_total_price(prices_dict):
    total = 0  

    for product, price in prices_dict.items():
        try:
            total += float(price) 
        except (ValueError, TypeError):
            print(f"  Skipping '{product}': Invalid price '{price}'") 

    return total

total_price = calculate_total_price(prices)
print(f"\n Total valid price: £{total_price:.2f}")

#######################################################################################################################################################

# Question 7 - Plotting and Visualization
# Generate 50 random numbers between 1 and 500, then:
# Plot a histogram to visualize the distribution of these numbers.
# Add appropriate labels for the x-axis and y-axis, and include a title for the histogram.

import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import random


random_numbers = [random.randint(1, 500) for _ in range(50)]

plt.figure(figsize=(10, 5)) 
plt.hist(random_numbers, bins=10, color='skyblue', edgecolor='black')  

plt.xlabel("Number Range")
plt.ylabel("Frequency")
plt.title("Histogram of 50 Random Numbers (1-500)")
plt.grid(axis='y', linestyle='--', alpha=0.7) 

plt.show()

#######################################################################################################################################################

# Question 8 - List Comprehensions
import matplotlib.pyplot as plt
# Given a list of integers representing order quantities.
import random
quantities = [5, 12, 9, 15, 7, 10]
import matplotlib.pyplot as plt
import random


quantities = [5, 12, 9, 15, 7, 10]

squared_quantities = [q ** 2 for q in quantities]

high_orders = [q for q in quantities if q > 10]

print(f" Original Quantities: {quantities}")
print(f" Squared Quantities: {squared_quantities}")
print(f" High Orders (Above 10): {high_orders}")

plt.figure(figsize=(8, 5))  
plt.bar(range(len(quantities)), quantities, color='lightcoral', edgecolor='black')

plt.xlabel("Order Index")
plt.ylabel("Quantity")
plt.title("Order Quantities Bar Chart")
plt.xticks(range(len(quantities)), labels=[f"Q{i+1}" for i in range(len(quantities))]) 

plt.show()


#######################################################################################################################################################

# Question 9 - Dictionary Manipulation
# Using the dictionary below, filter out the products with a rating of less than 4 and create a new dictionary with the remaining products.

ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}
high_rated_products = {product: rating for product, rating in ratings.items() if rating >= 4}
print(f"✅ High Rated Products: {high_rated_products}")


#######################################################################################################################################################

# Question 10 - Debugging and Correcting Code
# The following code intends to calculate the average of a list of numbers, but it contains errors:
values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i
average = total / len(values)
print("The average is" + average)

# Identify and correct the errors in the code.
# Comment on each error and explain your fixes.
values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i
average = total / len(values)
print("The average is" + average)  # ❌ ERROR HERE

#correct code
# List of values
values = [10, 20, 30, 40, 50]

# Calculate total sum using sum() (more efficient)
total = sum(values)

# Calculate average
average = total / len(values)

# Print result correctly using f-string
print(f"The average is {average:.2f}")  # Formats to 2 decimal places
#expected outcome
The average is 30.00



#######################################################################################################################################################
