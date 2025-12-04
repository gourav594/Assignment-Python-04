import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')

# TASK 1: Load data
def load_weather_data(csv_file):
    print("="*80 + "\nTASK 1: DATA ACQUISITION AND LOADING\n" + "="*80)
    df = pd.read_csv(csv_file)
    print(f"[OK] Data loaded: {csv_file}\n{df.shape[0]} records, {df.shape[1]} features")
    print("\n--- Sample Data ---\n", df.head())
    print("\n--- Info ---\n"), df.info()
    print("\n--- Stats ---\n", df.describe())
    missing = df.isnull().sum()
    print(f"\n--- Missing Values ---\n[OK] {missing.sum()} missing" if missing.sum() > 0 else "\n[OK] No missing")
    return df


# TASK 2: Clean data
def clean_weather_data(df):
    print("\n" + "="*80 + "\nTASK 2: DATA CLEANING AND PROCESSING\n" + "="*80)
    initial_missing = df.isnull().sum().sum()
    
    # Handle missing values
    print("\n--- Missing Values by Column ---")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing")
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Convert datetime columns if any
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"  [OK] {col} converted to datetime")
        except:
            print(f"  [FAIL] {col} could not be converted to datetime")
    
    # Filter relevant columns
    relevant_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Pressure', 'Cloud_Cover', 'Rainfall', 'Rain', 'Precip', 'Precipitation']
    available_cols = [col for col in relevant_cols if col in df.columns]
    if len(available_cols) > 0:
        print(f"\n--- Relevant Columns Selected ---\n  {', '.join(available_cols)}")
        df = df[available_cols]
    
    final_missing = df.isnull().sum().sum()
    print(f"\n[OK] Missing values: {initial_missing} -> {final_missing}")
    print("\n--- Data Ranges ---")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}")
    print("\n--- Categorical Distribution ---")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n  {col}:")
        print(df[col].value_counts().to_string())
    return df


# TASK 3: Statistical analysis
def statistical_analysis(df):
    print("\n" + "="*80 + "\nTASK 3: STATISTICAL ANALYSIS WITH NUMPY\n" + "="*80)
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = {col: {'mean': df[col].mean(), 'median': df[col].median(), 'std': df[col].std(), 
                   'min': df[col].min(), 'max': df[col].max()} for col in num_cols}
    
    print("\n" + "Metric".ljust(20) + "".join(col[:5].rjust(12) for col in num_cols))
    for stat in ['mean', 'median', 'std', 'min', 'max']:
        print(stat.ljust(20) + "".join(f"{stats[col][stat]:12.2f}" for col in num_cols))
    
    print("\n--- Rain Stats ---")
    rain_c = (df['Rain'] == 'rain').sum()
    print(f"Rainy: {rain_c} ({rain_c/len(df)*100:.1f}%)\nNo rain: {len(df)-rain_c}")
    return stats


# TASK 4: Visualizations
def create_visualizations(df):
    print("\n" + "="*80 + "\nTASK 4: VISUALIZATION WITH MATPLOTLIB\n" + "="*80)
    os.makedirs('plots', exist_ok=True)
    
    # Temp trend
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['Temperature'], linewidth=2, color='red', alpha=0.7)
    plt.xlabel('Index', fontweight='bold'); plt.ylabel('Temp (C)', fontweight='bold')
    plt.title('Temperature Trend', fontweight='bold')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('plots/01_temperature_trend.png', dpi=300); plt.close()
    print("  [OK] 01_temperature_trend.png")
    
    # Rain dist
    fig, ax = plt.subplots(figsize=(10, 6))
    rc = df['Rain'].value_counts()
    bars = ax.bar(rc.index, rc.values, color=['#ff6b6b', '#51cf66'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count', fontweight='bold'); ax.set_title('Rain Distribution', fontweight='bold')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout(); plt.savefig('plots/02_rain_distribution.png', dpi=300); plt.close()
    print("  [OK] 02_rain_distribution.png")
    
    # Humidity vs Temp
    fig, ax = plt.subplots(figsize=(10, 7))
    rain_mask = df['Rain'] == 'rain'
    ax.scatter(df[rain_mask]['Temperature'], df[rain_mask]['Humidity'], 
              alpha=0.6, s=60, color='blue', label='Rainy', edgecolors='black', linewidth=0.5)
    ax.scatter(df[~rain_mask]['Temperature'], df[~rain_mask]['Humidity'], 
              alpha=0.6, s=60, color='orange', label='No Rain', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Temp (C)', fontweight='bold'); ax.set_ylabel('Humidity (%)', fontweight='bold')
    ax.set_title('Humidity vs Temp', fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('plots/03_humidity_vs_temperature.png', dpi=300); plt.close()
    print("  [OK] 03_humidity_vs_temperature.png")
    
    # Wind dist
    plt.figure(figsize=(10, 6))
    plt.hist(df['Wind_Speed'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Wind Speed (km/h)', fontweight='bold'); plt.ylabel('Frequency', fontweight='bold')
    plt.title('Wind Speed Distribution', fontweight='bold'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig('plots/04_wind_speed_distribution.png', dpi=300); plt.close()
    print("  [OK] 04_wind_speed_distribution.png")
    
    # Combined
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].scatter(df['Temperature'], df['Pressure'], alpha=0.5, s=40, color='purple', edgecolors='black', linewidth=0.3)
    axes[0, 0].set_title('Pressure vs Temp', fontweight='bold'); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].hist(df['Cloud_Cover'], bins=25, color='lightblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Cloud Cover Distribution', fontweight='bold'); axes[0, 1].grid(axis='y', alpha=0.3)
    
    db = [df[df['Rain'] == 'rain']['Temperature'], df[df['Rain'] == 'no rain']['Temperature']]
    box = axes[1, 0].boxplot(db, tick_labels=['Rainy', 'No Rain'], patch_artist=True)
    for patch in box['boxes']: patch.set_facecolor('lightcoral')
    axes[1, 0].set_title('Temp by Rain', fontweight='bold'); axes[1, 0].grid(axis='y', alpha=0.3)
    
    db2 = [df[df['Rain'] == 'rain']['Humidity'], df[df['Rain'] == 'no rain']['Humidity']]
    box2 = axes[1, 1].boxplot(db2, tick_labels=['Rainy', 'No Rain'], patch_artist=True)
    for patch in box2['boxes']: patch.set_facecolor('lightgreen')
    axes[1, 1].set_title('Humidity by Rain', fontweight='bold'); axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout(); plt.savefig('plots/05_combined_visualization.png', dpi=300); plt.close()
    print("  [OK] 05_combined_visualization.png\n[OK] All plots created")


# TASK 5: Grouping & Aggregation
def grouping_and_aggregation(df):
    print("\n" + "="*80 + "\nTASK 5: GROUPING AND AGGREGATION\n" + "="*80)
    
    grouped = df.groupby('Rain')[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']].agg(['mean', 'median', 'std'])
    print("\nGrouped Statistics by Rain Condition:")
    print(grouped.to_string())
    
    corr = df[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']].corr()
    print("\n\nCorrelation Matrix:")
    print(corr.to_string())
    
    print("\n[OK] Grouping & correlation analysis complete")
    return grouped, corr


# TASK 6: Export and reporting
def export_and_report(df, stats):
    print("\n" + "="*80 + "\nTASK 6: EXPORT AND REPORTING\n" + "="*80)
    
    # Text report
    with open('weather_analysis_report.txt', 'w') as f:
        f.write("="*80 + "\nWEATHER ANALYSIS REPORT\n" + "="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Records: {len(df)}, Features: {len(df.columns)}\n")
        f.write(f"Features: {', '.join(df.columns)}\n\n")
        f.write("Statistical Summary:\n")
        for col, col_stats in stats.items():
            f.write(f"{col}: Mean={col_stats['mean']:.2f}, Median={col_stats['median']:.2f}, Std={col_stats['std']:.2f}\n")
        rain_c = (df['Rain'] == 'rain').sum()
        f.write(f"\nRainy days: {rain_c} ({rain_c/len(df)*100:.1f}%)\n")
    print("  [OK] weather_analysis_report.txt")
    
    print("\n[OK] Export completed")


def main():
    print("\n" + "="*80 + "\nWEATHER DATA VISUALIZER - COMPLETE ANALYSIS\n" + "="*80 + "\n")
    try:
        df = load_weather_data('data.csv')
        df_clean = clean_weather_data(df)
        stats = statistical_analysis(df_clean)
        create_visualizations(df_clean)
        grouping_and_aggregation(df_clean)
        export_and_report(df_clean, stats)
        print("\n" + "="*80 + "\n[OK] ALL TASKS COMPLETED\n" + "="*80 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}\n")


if __name__ == "__main__":
    main()
