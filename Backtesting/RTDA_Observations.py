import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ------------------------------
    # 1) Load the CSV files (adjust file paths as needed)
    # ------------------------------
    df_bt = pd.read_csv("data/bt_prices.csv", parse_dates=["date"])

    # ------------------------------
    # 2) Calculate the cumulative revenue if you sell 1 unit of RTDA every hour.
    # (This is defined as the negative cumulative sum of price_error.)
    df_bt["cumulative_revenue_price_error"] = -df_bt["price_error"].cumsum()

    # ------------------------------
    # 3) Print descriptive summary for the entire data set (Price Error).
    # ------------------------------
    print("=== Entire Data Set Price Error Summary ===")
    print(df_bt["price_error"].describe())
    print()
    
    # ------------------------------
    # 4) Create a test set (last 20% of the time series) for BT prices,
    #    and print a price error summary for the period BEFORE the test set.
    # ------------------------------
    test_size = int(len(df_bt) * 0.2)
    df_before = df_bt.iloc[:-test_size].copy()
    print("=== Period Before Test Set Price Error Summary ===")
    print(df_before["price_error"].describe())
    print()
    
    # ------------------------------
    # 5) Also, print a summary for the Test Set Price Error.
    # ------------------------------
    df_test = df_bt.iloc[-test_size:].copy()
    print("=== Period Test Set Price Error Summary ===")
    print(df_test["price_error"].describe())
    print()
    
    # ------------------------------
    # 6) Plot the cumulative revenue graph (Entire Period) in a separate figure.
    # ------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_bt["date"], df_bt["cumulative_revenue_price_error"], color="red", linewidth=2, label="Cumulative Revenue")
    plt.title("Cumulative Revenue from Selling 1 Unit of RTDA Every Hour", fontsize=16)
    plt.ylabel("Cumulative Revenue", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # ------------------------------
    # 7) Plot the daily prices graph (RT Prices and DA Prices) for the entire period in a separate figure.
    # ------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["rt_prices"], color="blue", linewidth=2, label="RT Prices")
    plt.plot(df["date"], df["da_prices"], color="orange", linewidth=2, label="DA Prices")
    plt.title("Daily Prices - Real Time versus Day Ahead", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
