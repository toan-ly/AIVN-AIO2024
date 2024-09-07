from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from data_preprocessing import *
from train import train_model
from evaluate import evaluate_model

def main():
    dataset_path = 'data/Housing.csv'
    df = load_data(dataset_path)
    encoded_df = encode_categorical(df)
    normalized_df = normalize(encoded_df)
    X_train, X_val, y_train, y_val = split_data(normalized_df)
    
    # Train and evaluate Decision Tree
    dt_model = train_model(DecisionTreeRegressor, X_train, y_train)
    print('Decision Tree:')
    evaluate_model(dt_model, X_val, y_val)

    # Train and evaluate Random Forest
    rf_model = train_model(RandomForestRegressor, X_train, y_train)
    print("\nRandom Forest:")
    evaluate_model(rf_model, X_val, y_val)
    
    # Train and evaluate AdaBoost
    ab_model = train_model(AdaBoostRegressor, X_train, y_train)
    print("\nAdaBoost:")
    evaluate_model(ab_model, X_val, y_val)
    
    # Train and evaluate Gradient Boosting
    gb_model = train_model(GradientBoostingRegressor, X_train, y_train)
    print("\nGradient Boosting:")
    evaluate_model(gb_model, X_val, y_val)
    
    
if __name__ == '__main__':
    main()