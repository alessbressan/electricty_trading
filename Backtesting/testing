import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ------------------------------
    # 1) Load original RL predictions (with timestamp column)
    # ------------------------------
    pred_df = pd.read_csv('final_test_ioan_predicted_spikes.csv', parse_dates=['timestamp'])
    pred_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    pred_df.rename(columns={'predicted_spike': 'predicted_spike_RL'}, inplace=True)
    pred_df.set_index('timestamp', inplace=True)
    
    # ---------------------------------------
    # 2) Load the transformer model's predictions (only one column)
    # ---------------------------------------
    transformer_df = pd.read_csv('transformer_predictions.csv')
    transformer_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    n_rows_trans = len(transformer_df)
    timestamps_trans = pd.date_range(end='2023-12-31 23:00:00', periods=n_rows_trans, freq='h')
    transformer_df.insert(0, 'timestamp', timestamps_trans)
    transformer_df.set_index('timestamp', inplace=True)
    if 'predicted_spike_transformer' not in transformer_df.columns:
        col_to_rename = transformer_df.columns[0]
        transformer_df.rename(columns={col_to_rename: 'predicted_spike_transformer'}, inplace=True)
    
    # ---------------------------------------
    # 3) Load the LSTM model's predictions (only one column)
    # ---------------------------------------
    lstm_df = pd.read_csv('LSTM_predictions_30.csv')
    lstm_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    n_rows_lstm = len(lstm_df)
    timestamps_lstm = pd.date_range(end='2023-12-31 23:00:00', periods=n_rows_lstm, freq='h')
    lstm_df.insert(0, 'timestamp', timestamps_lstm)
    lstm_df.set_index('timestamp', inplace=True)
    if 'predicted_spike_lstm' not in lstm_df.columns:
        col_to_rename = lstm_df.columns[0]
        lstm_df.rename(columns={col_to_rename: 'predicted_spike_lstm'}, inplace=True)
    
    # ---------------------------
    # 4) Load RTDA prices
    # ---------------------------
    price_df = pd.read_csv('data/bt_prices.csv', parse_dates=['date'])
    price_df.set_index('date', inplace=True)
    
    # ---------------------------
    # 5) Subset each DataFrame to the same date range
    # ---------------------------
    start_date = '2022-10-26'
    end_date = '2023-12-31'
    pred_df = pred_df.loc[start_date:end_date]
    transformer_df = transformer_df.loc[start_date:end_date]
    lstm_df = lstm_df.loc[start_date:end_date]
    price_df = price_df.loc[start_date:end_date]
    
    # ---------------------------
    # 6) Merge all data on the datetime index
    # ---------------------------
    combined_preds = pred_df.join(transformer_df, how='inner')
    combined_preds = combined_preds.join(lstm_df, how='inner')
    df = combined_preds.join(price_df, how='inner')
    
    required_cols = [
        'predicted_spike_RL', 
        'predicted_spike_transformer',
        'predicted_spike_lstm',
        'price_error'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in merged DataFrame: {missing_cols}")
    
    # ---------------------------------
    # 7) Compute hourly returns (price changes)
    # ---------------------------------
    df['ret'] = df['price_error'].fillna(0)
    
    # ---------------------------------
    # 8) Compute portfolio values for each strategy
    # ---------------------------------
    initial_capital = 100.0
    # RL Portfolio Value
    df['pnl_RL'] = np.where(df['predicted_spike_RL'] == 1, 1, -1) * df['ret']
    df['portfolio_value_RL'] = initial_capital + df['pnl_RL'].cumsum()
    # Transformer Portfolio Value
    df['pnl_trans'] = np.where(df['predicted_spike_transformer'] == 1, 1, -1) * df['ret']
    df['portfolio_value_trans'] = initial_capital + df['pnl_trans'].cumsum()
    # LSTM Portfolio Value
    df['pnl_lstm'] = np.where(df['predicted_spike_lstm'] == 1, 1, -1) * df['ret']
    df['portfolio_value_lstm'] = initial_capital + df['pnl_lstm'].cumsum()
    
    # ---------------------------------
    # 9) Compute cumulative revenue from selling 1 unit of RTDA each hour.
    # (Revenue is defined as the negative cumulative sum of price_error.)
    df["cumulative_revenue_Rtda"] = -df["price_error"].cumsum()
    
    # Create spike indicator masks.
    spike_indicator_RL = (df['predicted_spike_RL'] == 1).astype(int)
    spike_indicator_trans = (df['predicted_spike_transformer'] == 1).astype(int)
    spike_indicator_lstm = (df['predicted_spike_lstm'] == 1).astype(int)
    
    # ---------------------------------
    # 10) Create a combined figure with 5 vertically stacked subplots:
    #     Row 1: Combined Portfolio Values (all 3 models)
    #     Row 2: RL Spike Indicator Bar
    #     Row 3: Transformer Spike Indicator Bar
    #     Row 4: LSTM Spike Indicator Bar
    #     Row 5: Cumulative Revenue from RTDA
    # ---------------------------------
    fig, axs = plt.subplots(5, 1, figsize=(16, 20), sharex=True,
                            gridspec_kw={'height_ratios': [2, 0.5, 0.5, 0.5, 2]})
    
    # Row 1: Combined Portfolio Values
    axs[0].plot(df.index, df['portfolio_value_RL'], color='black', linewidth=2, label='RL')
    axs[0].plot(df.index, df['portfolio_value_trans'], color='green', linewidth=2, label='Transformer')
    axs[0].plot(df.index, df['portfolio_value_lstm'], color='blue', linewidth=2, label='LSTM')
    # The pad parameter shifts the title downward (increase pad value as needed)
    axs[0].set_title("Combined Portfolio Values", fontsize=14, pad=20)
    axs[0].set_ylabel("Portfolio Value", fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Row 2: RL Spike Indicator Bar
    axs[1].fill_between(df.index, 0, spike_indicator_RL, color='black', step='pre')
    axs[1].set_ylim(0, 1)
    axs[1].set_yticks([])
    axs[1].grid(False)
    axs[1].set_ylabel("RL Spikes", fontsize=8, rotation=90, labelpad=10)
    
    # Row 3: Transformer Spike Indicator Bar
    axs[2].fill_between(df.index, 0, spike_indicator_trans, color='green', step='pre')
    axs[2].set_ylim(0, 1)
    axs[2].set_yticks([])
    axs[2].grid(False)
    axs[2].set_ylabel("Trans. Spikes", fontsize=8, rotation=90, labelpad=10)
    
    # Row 4: LSTM Spike Indicator Bar
    axs[3].fill_between(df.index, 0, spike_indicator_lstm, color='blue', step='pre')
    axs[3].set_ylim(0, 1)
    axs[3].set_yticks([])
    axs[3].grid(False)
    axs[3].set_ylabel("LSTM Spikes", fontsize=8, rotation=90, labelpad=10)
    
    # Row 5: Cumulative Revenue
    axs[4].plot(df.index, df["cumulative_revenue_Rtda"], color="red", linewidth=2, label="Cumulative Revenue (RTDA)")
    axs[4].set_xlabel("Date", fontsize=14)
    axs[4].set_ylabel("Cumulative Revenue", fontsize=12)
    axs[4].grid(True, linestyle='--', alpha=0.7)
    axs[4].legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
