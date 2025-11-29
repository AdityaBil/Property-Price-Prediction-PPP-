import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pickle

st.set_page_config(page_title="Property Price Prediction", layout="wide")
st.title("Property Price Prediction based on Inflation Forecast")

uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File Loaded Successfully!")
        
        with st.expander("View Data Preview"):
            st.dataframe(df.iloc[:20,:])
        
        try:
            with open('stacked_model.pkl', 'rb') as f:
                stacked_model = pickle.load(f)
            with open('y_pred.pkl', 'rb') as f:
                y_pred = pickle.load(f)
            with open('y_test.pkl', 'rb') as f:
                y_test = pickle.load(f)
            with open('metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
            
            st.subheader("=== Model Performance Metrics ===")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Train R2 Score", f"{metrics['r2_train']:.4f}")
            with col2:
                st.metric("Test R2 Score", f"{metrics['r2_test']:.4f}")
            with col3:
                st.metric("Test MSE", f"{metrics['mse_test']:.4f}")
            with col4:
                st.metric("Test RMSE", f"{metrics['rmse_test']:.4f}")
            with col5:
                st.metric("Test MAE", f"{metrics['mae_test']:.4f}")

            with st.expander("View Inflation Rate Prediction"):
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.lineplot(x=range(len(y_test)), y=y_test, label='Actual', marker='o', ax=ax1)
                sns.lineplot(x=range(len(y_pred)), y=y_pred, label='Predicted', marker='D', ax=ax1)
                ax1.set_title("Inflation Rate Prediction")
                ax1.set_xlabel("Index")
                ax1.set_ylabel("Inflation Rate")
                ax1.legend()
                plt.tight_layout()
                st.pyplot(fig1)

            with st.expander("View 3D Inflated Price Visualization"):
                Prices = np.linspace(50, 100, len(y_pred))
                Time = np.linspace(5, 10, len(y_pred))
                Time_mesh, Prices_mesh = np.meshgrid(Time, Prices)
                Inflation_mesh = np.tile(y_pred, (len(Prices), 1)).T

                fig2 = plt.figure(figsize=(8, 6))
                ax2 = fig2.add_subplot(111, projection='3d')
                ax2.plot_surface(Prices_mesh, Time_mesh, Inflation_mesh, cmap='viridis')
                ax2.set_xlabel('Average Price (Lakhs)')
                ax2.set_ylabel('Years')
                ax2.set_zlabel('Predicted Inflation Rate')
                plt.title('Inflated Price')
                plt.tight_layout()
                st.pyplot(fig2)
                
        except FileNotFoundError:
            st.warning("Model files not found. Please run the training script first.")
            y_pred = None

        st.subheader("Future Price Prediction")
        
        if 'Areas' in df.columns and 'AveragePrice' in df.columns:
            available_locations = df['Areas'].unique().tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                user_loc = st.selectbox("Enter the location", available_locations)
            with col2:
                user_area = st.number_input("Enter desired area", min_value=1, value=1000, step=100)
            
            if st.button("Calculate Future Price"):
                if y_pred is not None:
                    price = df.loc[df['Areas'] == user_loc, 'AveragePrice'].iloc[0]
                    area_price = price * user_area
                    
                    st.write(f"**Location:** {user_loc}")
                    st.write(f"**Area:** {user_area} sqft")
                    st.write(f"**Current Price:** ₹{area_price:,.2f}")
                    
                    avg_inflation = np.mean(y_pred) / 100
                    GrowthRate = 0.08
                    real_growth = GrowthRate - avg_inflation

                    st.write(f"**Average Inflation:** {avg_inflation*100:.2f}%")
                    st.write(f"**Real Growth Rate:** {real_growth*100:.2f}%")
                    
                    future_prices = {}
                    for years in [5, 7, 10]:
                        future_price = area_price * ((1 + real_growth) ** years)
                        future_prices[years] = future_price
                        st.write(f"Price after **{years} years**: ₹{round(future_price):,}")

                    st.subheader("Before vs After Property Price")
                    years = [5, 7, 10]
                    current_price = area_price
                    future_prices_list = [current_price * ((1 + real_growth) ** yr) for yr in years]

                    fig3, ax3 = plt.subplots(figsize=(7, 4))
                    ax3.bar([f"{yr} yrs (Now)" for yr in years], [current_price]*len(years), label='Current Price', alpha=0.6)
                    ax3.bar([f"{yr} yrs (Future)" for yr in years], future_prices_list, label='Predicted Price', alpha=0.8)
                    ax3.set_ylabel("Price (INR)")
                    ax3.set_title("Before vs After Property Price")
                    plt.xticks(rotation=45)
                    ax3.legend()
                    plt.tight_layout()
                    st.pyplot(fig3)
                else:
                    st.error("Model not loaded. Please run the training script first.")

        with st.expander("View Average Total Price by Area"):
            area_price_df = df[['Areas', 'AveragePrice']].groupby('Areas').mean()
            area_price_df = area_price_df.sort_values('AveragePrice', ascending=False)
            fig4, ax4 = plt.subplots(figsize=(80, 6))
            sns.heatmap(area_price_df.T, annot=True, fmt=".0f", cmap="YlOrRd", cbar=True, linewidths=0.8, ax=ax4)
            ax4.set_title("Average Total Price by Area")
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig4)

        st.subheader("Top Most Expensive Areas")
        top_n = st.slider("Select number of top areas to display", min_value=5, max_value=30, value=15, step=1)
        
        area_stats = df.groupby('Areas')['AveragePrice'].agg(['mean', 'count']).reset_index()
        area_stats = area_stats.nlargest(top_n, 'mean')
        
        fig5, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.barplot(data=area_stats, y='Areas', x='mean', ax=axes[0], palette='rocket')
        axes[0].set_xlabel('Average Price per Sqft (INR)', fontsize=11)
        axes[0].set_ylabel('Areas', fontsize=11)
        axes[0].set_title(f'Top {top_n} Most Expensive Areas', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        sns.scatterplot(data=area_stats, x='mean', y='count', size='mean', 
                        sizes=(50, 500), alpha=0.6, ax=axes[1], legend=False)
        axes[1].set_xlabel('Average Price per Sqft (INR)', fontsize=11)
        axes[1].set_ylabel('Number of Properties', fontsize=11)
        axes[1].set_title('Price vs Property Count', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig5)
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload an Excel file to begin analysis")