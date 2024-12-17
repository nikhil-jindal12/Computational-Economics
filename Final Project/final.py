import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib

def process_training_data(df):
    """Process training data"""
    # Create a copy
    df = df.copy()
    
    # Remove rows with NaN or invalid selling prices
    df = df.dropna(subset=['sellingprice'])
    df = df[np.isfinite(df['sellingprice'])]
    
    # Remove extreme outliers (optional, adjust as needed)
    Q1 = df['sellingprice'].quantile(0.01)
    Q3 = df['sellingprice'].quantile(0.99)
    df = df[(df['sellingprice'] >= Q1) & (df['sellingprice'] <= Q3)]
    
    # Handle missing values first
    df = df.fillna({
        'odometer': df['odometer'].median(),
        'year': df['year'].median(),
        'transmission': df['transmission'].mode()[0],
        'make': 'unknown',
        'model': 'unknown',
        'body': 'unknown',
        'seller': 'unknown',
        'interior': 'unknown',
        'state': 'unknown',
        'saledate': 'unknown'
    })
    
    # Process categorical variables
    categorical_columns = ['transmission', 'model', 'make', 'body', 'seller', 'interior', 'state']
    encodings = {}
    
    for col in categorical_columns:
        # Create integer codes and store unique categories
        df[f'{col}_code'], encodings[col] = pd.factorize(df[col])
        df[f'{col}_code'] = df[f'{col}_code'].astype('int32')
        df.drop(columns=[col], inplace=True)
    
    # Drop unnecessary columns early
    df = df.drop(columns=['color', 'mmr', 'trim'])
    
    # Process odometer
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    df['odometer'] = df['odometer'].fillna(df['odometer'].median())
    df['odometer_category'] = df['odometer'].astype(float) // 10000
    mask = df['odometer'] % 10000 == 0
    df.loc[mask, 'odometer_category'] -= 1
    df['odometer_category'] = df['odometer_category'].astype(int)
    df.drop(columns=['odometer'], inplace=True)
    
    # Process quarter
    df['quarter'] = 0
    q1_mask = df['saledate'].str.lower().str.contains(r'jan|feb|mar', na=False)
    q2_mask = df['saledate'].str.lower().str.contains(r'apr|may|jun', na=False)
    q3_mask = df['saledate'].str.lower().str.contains(r'jul|aug|sep', na=False)
    q4_mask = df['saledate'].str.lower().str.contains(r'oct|nov|dec', na=False)
    
    df.loc[q1_mask, 'quarter'] = 1
    df.loc[q2_mask, 'quarter'] = 2
    df.loc[q3_mask, 'quarter'] = 3
    df.loc[q4_mask, 'quarter'] = 4
    df.drop(columns=['saledate'], inplace=True)
    
    # Process VIN
    df['modelyear'] = df['vin'].str[9].fillna('0')
    df['modelyear_code'], _ = pd.factorize(df['modelyear'])
    df.drop(columns=['vin', 'modelyear'], inplace=True)
    
    # Create feature interactions
    df['age_odometer'] = df['year'] * df['odometer_category']
    
    # Ensure all numeric columns are float32
    numeric_columns = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].astype('float32')
    
    return df, encodings

def process_test_data(df, encodings):
    """Process test data using training data encodings"""
    # Create a copy
    df = df.copy()
    
    # Handle missing values first
    df = df.fillna({
        'odometer': df['odometer'].median(),
        'year': df['year'].median(),
        'transmission': 'unknown',
        'make': 'unknown',
        'model': 'unknown',
        'body': 'unknown',
        'seller': 'unknown',
        'interior': 'unknown',
        'state': 'unknown',
        'saledate': 'unknown'
    })
    
    # Process categorical variables
    categorical_columns = ['transmission', 'model', 'make', 'body', 'seller', 'interior', 'state']
    
    for col in categorical_columns:
        # Use the encodings from training to map test data
        df[f'{col}_code'] = pd.Categorical(
            df[col], 
            categories=encodings[col]
        ).codes.astype('int32')
        df.drop(columns=[col], inplace=True)
    
    # Drop unnecessary columns early
    df = df.drop(columns=['color', 'mmr', 'trim'])
    
    # Process odometer
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    df['odometer'] = df['odometer'].fillna(df['odometer'].median())
    df['odometer_category'] = df['odometer'].astype(float) // 10000
    mask = df['odometer'] % 10000 == 0
    df.loc[mask, 'odometer_category'] -= 1
    df['odometer_category'] = df['odometer_category'].astype(int)
    df.drop(columns=['odometer'], inplace=True)
    
    # Process quarter
    df['quarter'] = 0
    q1_mask = df['saledate'].str.lower().str.contains(r'jan|feb|mar', na=False)
    q2_mask = df['saledate'].str.lower().str.contains(r'apr|may|jun', na=False)
    q3_mask = df['saledate'].str.lower().str.contains(r'jul|aug|sep', na=False)
    q4_mask = df['saledate'].str.lower().str.contains(r'oct|nov|dec', na=False)
    
    df.loc[q1_mask, 'quarter'] = 1
    df.loc[q2_mask, 'quarter'] = 2
    df.loc[q3_mask, 'quarter'] = 3
    df.loc[q4_mask, 'quarter'] = 4
    df.drop(columns=['saledate'], inplace=True)
    
    # Process VIN
    df['modelyear'] = df['vin'].str[9].fillna('0')
    df['modelyear_code'] = pd.Categorical(
        df['modelyear'], 
        categories=np.unique(df['modelyear'])
    ).codes
    df.drop(columns=['vin', 'modelyear'], inplace=True)
    
    # Create feature interactions
    df['age_odometer'] = df['year'] * df['odometer_category']
    
    # Ensure all numeric columns are float32
    numeric_columns = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].astype('float32')
    
    return df

def train_model(X_train, y_train):
    """Train the model"""
    # Create and train XGBoost model with optimal parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=7,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=3,
        subsample=0.9,
        random_state=42,
        enable_categorical=True  # Enable categorical feature support
    )
    
    # Convert data to float32 to ensure compatibility
    X_train = X_train.astype({col: 'float32' for col in X_train.columns})
    
    # Ensure y_train is clean and valid
    y_train = pd.Series(y_train)
    y_train = y_train.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Log transform with a small epsilon to avoid log(0)
    y_train_log = np.log1p(y_train)
    
    # Create pipeline with scaler
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model)
    ])
    
    # Align X_train with y_train after cleaning
    X_train = X_train.loc[y_train.index]
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train_log)
    
    return pipeline

def train_and_save_model(train_file, model_file='car_price_model.joblib', encodings_file='categorical_encodings.joblib'):
    """Train model and save to files"""
    # Load training data
    training_data = pd.read_csv(train_file)
    
    # Process training data
    processed_training_data, encodings = process_training_data(training_data)
    
    # Prepare training features and target
    X_train = processed_training_data.drop('sellingprice', axis=1)
    y_train = processed_training_data['sellingprice']
    
    # Train the model
    pipeline = train_model(X_train, y_train)
    
    # Save model and encodings
    joblib.dump(pipeline, model_file)
    joblib.dump(encodings, encodings_file)
    
    print(f"Model saved to {model_file}")
    print(f"Categorical encodings saved to {encodings_file}")
    
    return pipeline, encodings

def make_predictions(test_file, model_file='car_price_model.joblib', encodings_file='categorical_encodings.joblib', output_file='predictions.csv'):
    """Load model and make predictions on test data"""
    # Load model and encodings
    pipeline = joblib.load(model_file)
    encodings = joblib.load(encodings_file)
    
    # Load and process test data
    test_data = pd.read_csv(test_file)
    processed_test_data = process_test_data(test_data, encodings)
    
    # Make predictions
    predictions = np.expm1(pipeline.predict(processed_test_data))
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'vin': test_data['vin'],
        'sellingprice': predictions
    })
    
    # Save predictions to CSV
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")
    print("\nSample predictions:")
    print(submission.head())

def main():
    # Train and save the model
    train_and_save_model('final_train.csv')
    
    # Make predictions using the saved model
    make_predictions('final_test-1.csv')

if __name__ == "__main__":
    main()