import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Streamlit UI Title and Description
st.title("Zomato Data Analysis")
st.write("""
This application provides insights into Zomato restaurant data, including visualizations of 
ratings, votes, cost, and more.
""")

# Load the dataset
@st.cache_data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        st.error("The specified file was not found. Please check the file path.")
        return None

dataframe = load_data("Zomato data .csv")

if dataframe is not None:
    # Display the first few rows
    st.subheader("First 5 Rows of the Dataset")
    st.dataframe(dataframe.head())

    # Clean and preprocess the 'rate' column
    def handle_rate(value):
        try:
            value = str(value).split('/')[0]  # Extract the numeric part
            return float(value)
        except:
            return np.nan  # Handle invalid values

    dataframe['rate'] = dataframe['rate'].apply(handle_rate)

    # Restaurant Type Countplot
    st.subheader("Count of Restaurants by Type")
    if 'listed_in(type)' in dataframe.columns:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.countplot(x=dataframe['listed_in(type)'], ax=ax1, palette="viridis")
        ax1.set_xlabel("Type of Restaurant", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig1)

    # Votes by Type of Restaurant
    st.subheader("Votes by Type of Restaurant")
    if 'votes' in dataframe.columns and 'listed_in(type)' in dataframe.columns:
        grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
        result = pd.DataFrame({'votes': grouped_data})
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        plt.plot(result, color="green", marker="o")
        plt.xlabel("Type of Restaurant", color="red", fontsize=12)
        plt.ylabel("Votes", color="red", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

    # Restaurant with Maximum Votes
    st.subheader("Restaurant(s) with Maximum Votes")
    max_votes = dataframe['votes'].max()
    restaurant_with_max_votes = dataframe.loc[dataframe['votes'] == max_votes, 'name']
    st.write(f"Maximum Votes: {max_votes}")
    st.write(restaurant_with_max_votes)

    # Online Order Countplot
    st.subheader("Online Order Availability")
    if 'online_order' in dataframe.columns:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.countplot(x=dataframe['online_order'], ax=ax3, palette="coolwarm")
        ax3.set_xlabel("Online Order Available")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

    # Ratings Distribution
    st.subheader("Ratings Distribution")
    if 'rate' in dataframe.columns:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        plt.hist(dataframe['rate'].dropna(), bins=5, color="skyblue", edgecolor="black")
        plt.title("Ratings Distribution", fontsize=14)
        plt.xlabel("Ratings")
        plt.ylabel("Frequency")
        st.pyplot(fig4)

    # Approximate Cost for Two
    st.subheader("Approximate Cost for Two")
    if 'approx_cost(for two people)' in dataframe.columns:
        couple_data = dataframe['approx_cost(for two people)']
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        sns.countplot(x=couple_data, ax=ax5, palette="magma")
        ax5.set_xlabel("Cost for Two")
        ax5.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig5)

    # Boxplot: Online Order vs Ratings
    st.subheader("Online Order vs Ratings")
    if 'online_order' in dataframe.columns and 'rate' in dataframe.columns:
        fig6, ax6 = plt.subplots(figsize=(6, 6))
        sns.boxplot(x='online_order', y='rate', data=dataframe, ax=ax6, palette="Set2")
        ax6.set_xlabel("Online Order")
        ax6.set_ylabel("Ratings")
        st.pyplot(fig6)

    # Heatmap: Online Order vs Type of Restaurant
    st.subheader("Online Order vs Type of Restaurant (Heatmap)")
    if 'online_order' in dataframe.columns and 'listed_in(type)' in dataframe.columns:
        pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt='d', ax=ax7)
        plt.title("Heatmap: Online Order vs Type of Restaurant")
        plt.xlabel("Online Order")
        plt.ylabel("Type of Restaurant")
        st.pyplot(fig7)


# Correlation Heatmap
if dataframe is not None:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = dataframe[['rate', 'votes', 'approx_cost(for two people)']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    # Online Order Impact on Ratings and Votes
    st.subheader("Online Order Impact")
    online_order_stats = dataframe.groupby('online_order').agg({'rate': 'mean', 'votes': 'sum'})
    st.write("Online Order Stats:")
    st.bar_chart(online_order_stats)

    # Cost vs Rating Analysis
    st.subheader("Cost vs Ratings")
    fig, ax = plt.subplots()
    ax.scatter(dataframe['approx_cost(for two people)'], dataframe['rate'], alpha=0.5, color='blue')
    ax.set_title("Cost vs Ratings")
    ax.set_xlabel("Approx Cost for Two")
    ax.set_ylabel("Ratings")
    st.pyplot(fig)

    # Votes Distribution by Booking Options
    st.subheader("Votes Distribution by Table Booking")
    booking_votes = dataframe.groupby('book_table').agg({'votes': 'mean'})
    st.bar_chart(booking_votes)

    # Clustering Analysis
    st.subheader("Restaurant Clustering")
    features = dataframe[['rate', 'votes', 'approx_cost(for two people)']].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    features['Cluster'] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        features['rate'], 
        features['approx_cost(for two people)'], 
        c=features['Cluster'], 
        cmap='viridis', alpha=0.7
    )
    ax.set_title("Clusters of Restaurants")
    ax.set_xlabel("Ratings")
    ax.set_ylabel("Approx Cost")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    # Predicting Ratings Using Machine Learning
    st.subheader("Predicting Ratings")
    dataframe.dropna(subset=['rate'], inplace=True)
    X = pd.get_dummies(dataframe[['online_order', 'book_table', 'votes', 'approx_cost(for two people)']], drop_first=True)
    y = dataframe['rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"Mean Squared Error (MSE) for the Ratings Prediction Model: {mse:.2f}")
