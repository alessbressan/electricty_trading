import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ------------------------------
    # 1) Load the CSV files (adjust file paths as needed)
    # ------------------------------
    df_bt = pd.read_csv("data/bt_prices.csv", parse_dates=["date"])
    df = pd.read_csv("data/ml_features.csv", parse_dates=["date"])

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
    # 6) Plot graphs for the entire period:
    #    Top: Cumulative Revenue (Entire Period)
    #    Bottom: Daily Prices (RT Prices and DA Prices) for the entire period
    # ------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top subplot: Cumulative Revenue from selling 1 unit of RTDA every hour.
    axs[0].plot(df_bt["date"], df_bt["cumulative_revenue_price_error"], color="red", linewidth=2,
                label="Cumulative Revenue (Entire Period)")
    axs[0].set_title("Cumulative Revenue from Selling 1 Unit of RTDA Every Hour\n(Entire Period)", fontsize=16)
    axs[0].set_ylabel("Cumulative Revenue", fontsize=14)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(fontsize=12)
    
    # Bottom subplot: Daily Prices (RT Prices and DA Prices) for the entire period.
    axs[1].plot(df["date"], df["rt_prices"], color="blue", linewidth=2, label="RT Prices")
    axs[1].plot(df["date"], df["da_prices"], color="orange", linewidth=2, label="DA Prices")
    axs[1].set_title("Daily Prices (Entire Period)", fontsize=16)
    axs[1].set_xlabel("Date", fontsize=14)
    axs[1].set_ylabel("Price", fontsize=14)
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "_main_":
    main()