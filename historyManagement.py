import pandas as pd
import joblib
import os
import json
from datetime import datetime

MODEL_REGISTRY_FILE_REGRESSION = "Model_Training_History/History_Regression.csv"
MODEL_REGISTRY_FILE_CLASSIFICATION = "Model_Training_History/History_Classification.csv"

def load_or_create_registry(model_type):
    """Load existing model registry or create new one for both regression and classification"""
    
    if model_type.lower() == 'regression':
        registry_file = MODEL_REGISTRY_FILE_REGRESSION
        columns = [
            'model_name', 'scaler_name',
            'features_list', 'target_column',
            'mse', 'mae', 'rmse', 'r2',
            'parameters'
        ]
    elif model_type.lower() == 'classification':
        registry_file = MODEL_REGISTRY_FILE_CLASSIFICATION
        columns = [
            'model_name', 'scaler_name',
            'features_list', 'target_column',
            'classification_report_dict', 'confusion_matrix_df', 'auc_score_dict',
            'parameters'
        ]
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")
    
    try:
        return pd.read_csv(registry_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)
    
def save_model_performance_if_better(model_type, params):
    """Save model performance only if it performs better than existing version for the same scaler"""
    
    # Load existing registry
    registry_df = load_or_create_registry(model_type)
    
    # Define key columns for matching existing models
    key_columns = ['model_name', 'scaler_name', 'target_column']
    
    # Check if model with same key exists
    mask = (
        (registry_df['model_name'] == params['model_name']) &
        (registry_df['scaler_name'] == params['scaler_name']) &
        (registry_df['target_column'] == params['target_column'])
    )
    
    existing_model = registry_df[mask]
    
    # Determine registry file
    registry_file = MODEL_REGISTRY_FILE_REGRESSION if model_type.lower() == 'regression' else MODEL_REGISTRY_FILE_CLASSIFICATION
    
    if len(existing_model) > 0:
        # Model exists, check if current is better
        existing_row = existing_model.iloc[0]
        
        # Calculate number of features
        current_features = len(params['features_list'].split(',')) if params['features_list'] else 0
        existing_features = len(existing_row['features_list'].split(',')) if existing_row['features_list'] else 0
        
        if model_type.lower() == 'regression':
            # For regression: Check if current model is better
            current_mse = params['mse']
            current_r2 = params['r2']
            existing_mse = existing_row['mse']
            existing_r2 = existing_row['r2']
            
            # Performance comparison with tolerance for "same" performance
            mse_tolerance = abs(existing_mse) * 0.01  # 1% tolerance
            r2_tolerance = abs(existing_r2) * 0.01   # 1% tolerance
            
            # Check different improvement scenarios
            performance_better = (current_mse < existing_mse) and (current_r2 > existing_r2)
            performance_same = (abs(current_mse - existing_mse) <= mse_tolerance) and (abs(current_r2 - existing_r2) <= r2_tolerance)
            fewer_features = current_features < existing_features
            
            # Determine if model should be saved
            is_better = performance_better or (performance_same and fewer_features)
            
            if is_better:
                for col, value in params.items():
                    if col in registry_df.columns:
                        registry_df.loc[mask, col] = value
                    else:
                        print(f"Warning: Column '{col}' not found in registry")
                
                if performance_better:
                    print(f"✅ Regression model improved with better performance!")
                    print(f"   MSE: {existing_mse:.6f} → {current_mse:.6f} (lower is better)")
                    print(f"   R²:  {existing_r2:.6f} → {current_r2:.6f} (higher is better)")
                    print(f"   Features: {existing_features} → {current_features}")
                elif performance_same and fewer_features:
                    print(f"✅ Regression model improved with same performance but fewer features!")
                    print(f"   MSE: {existing_mse:.6f} ≈ {current_mse:.6f} (similar)")
                    print(f"   R²:  {existing_r2:.6f} ≈ {current_r2:.6f} (similar)")
                    print(f"   Features: {existing_features} → {current_features} (🎯 {existing_features - current_features} fewer features!)")
            else:
                print(f"❌ Regression model not improved:")
                print(f"   Current MSE: {current_mse:.6f}, Best MSE: {existing_mse:.6f}")
                print(f"   Current R²: {current_r2:.6f}, Best R²: {existing_r2:.6f}")
                print(f"   Current Features: {current_features}, Best Features: {existing_features}")
                return
                
        elif model_type.lower() == 'classification':
            # For classification: Check if current model is better
            current_report = json.loads(params['classification_report_dict']) if isinstance(params['classification_report_dict'], str) else params['classification_report_dict']
            # existing_report = json.loads(existing_row['classification_report_dict']) if isinstance(existing_row['classification_report_dict'], str) else existing_row['classification_report_dict']
            
            # Around line 108, replace the existing JSON parsing with this:
            try:
                if isinstance(existing_row['classification_report_dict'], str):
                    existing_report = json.loads(existing_row['classification_report_dict'])
                else:
                    existing_report = existing_row['classification_report_dict']
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️  Warning: Could not parse existing classification report: {e}")
                print("   Treating as new model entry...")
                # Treat as new model if we can't parse existing data
                new_row = pd.DataFrame([params])
                registry_df = pd.concat([registry_df, new_row], ignore_index=True)
                
                # Create directory and save
                os.makedirs(os.path.dirname(registry_file), exist_ok=True)
                registry_df.to_csv(registry_file, index=False)
                print(f"✨ New classification model added due to parsing error")
                print(f"💾 Registry updated: {registry_file}")
                return

            # Extract metrics
            current_accuracy = current_report.get('accuracy', 0)
            current_f1 = current_report.get('weighted avg', {}).get('f1-score', 0)
            if current_f1 == 0:  # Fallback to micro avg
                current_f1 = current_report.get('micro avg', {}).get('f1-score', 0)
            
            existing_accuracy = existing_report.get('accuracy', 0)
            existing_f1 = existing_report.get('weighted avg', {}).get('f1-score', 0)
            if existing_f1 == 0:  # Fallback to micro avg
                existing_f1 = existing_report.get('micro avg', {}).get('f1-score', 0)
            
            # Performance comparison with tolerance
            accuracy_tolerance = abs(existing_accuracy) * 0.01  # 1% tolerance
            f1_tolerance = abs(existing_f1) * 0.01             # 1% tolerance
            
            # Check different improvement scenarios
            performance_better = (current_accuracy > existing_accuracy) and (current_f1 > existing_f1)
            performance_same = (abs(current_accuracy - existing_accuracy) <= accuracy_tolerance) and (abs(current_f1 - existing_f1) <= f1_tolerance)
            fewer_features = current_features < existing_features
            
            # Determine if model should be saved
            is_better = performance_better or (performance_same and fewer_features)
            
            # Extract AUC for display
            current_auc_dict = json.loads(params['auc_score_dict']) if isinstance(params['auc_score_dict'], str) else params['auc_score_dict']
            existing_auc_dict = json.loads(existing_row['auc_score_dict']) if isinstance(existing_row['auc_score_dict'], str) else existing_row['auc_score_dict']
            
            current_auc_micro = current_auc_dict.get('micro-avg', 0)
            existing_auc_micro = existing_auc_dict.get('micro-avg', 0)
            
            if is_better:
                for col, value in params.items():
                    if col in registry_df.columns:
                        registry_df.loc[mask, col] = value
                    else:
                        print(f"Warning: Column '{col}' not found in registry")
                
                if performance_better:
                    print(f"✅ Classification model improved with better performance!")
                    print(f"   Accuracy: {existing_accuracy:.6f} → {current_accuracy:.6f}")
                    print(f"   F1-Score: {existing_f1:.6f} → {current_f1:.6f}")
                    print(f"   AUC (micro-avg): {existing_auc_micro:.6f} → {current_auc_micro:.6f}")
                    print(f"   Features: {existing_features} → {current_features}")
                elif performance_same and fewer_features:
                    print(f"✅ Classification model improved with same performance but fewer features!")
                    print(f"   Accuracy: {existing_accuracy:.6f} ≈ {current_accuracy:.6f} (similar)")
                    print(f"   F1-Score: {existing_f1:.6f} ≈ {current_f1:.6f} (similar)")
                    print(f"   AUC (micro-avg): {existing_auc_micro:.6f} ≈ {current_auc_micro:.6f}")
                    print(f"   Features: {existing_features} → {current_features} (🎯 {existing_features - current_features} fewer features!)")
            else:
                print(f"❌ Classification model not improved:")
                print(f"   Current Accuracy: {current_accuracy:.6f}, Best Accuracy: {existing_accuracy:.6f}")
                print(f"   Current F1: {current_f1:.6f}, Best F1: {existing_f1:.6f}")
                print(f"   Current AUC (micro-avg): {current_auc_micro:.6f}, Best AUC: {existing_auc_micro:.6f}")
                print(f"   Current Features: {current_features}, Best Features: {existing_features}")
                return
    else:
        # Model doesn't exist, add new row
        new_row = pd.DataFrame([params])
        registry_df = pd.concat([registry_df, new_row], ignore_index=True)
        
        current_features = len(params['features_list'].split(',')) if params['features_list'] else 0
        
        if model_type.lower() == 'regression':
            print(f"✨ New regression model added:")
            print(f"   MSE: {params['mse']:.6f}, R²: {params['r2']:.6f}")
            print(f"   Features: {current_features}")
        else:
            # Extract metrics for display
            report = json.loads(params['classification_report_dict']) if isinstance(params['classification_report_dict'], str) else params['classification_report_dict']
            auc_dict = json.loads(params['auc_score_dict']) if isinstance(params['auc_score_dict'], str) else params['auc_score_dict']
            
            accuracy = report.get('accuracy', 0)
            f1 = report.get('weighted avg', {}).get('f1-score', 0)
            auc_micro = auc_dict.get('micro-avg', 0)
            
            print(f"✨ New classification model added:")
            print(f"   Accuracy: {accuracy:.6f}, F1: {f1:.6f}, AUC (micro-avg): {auc_micro:.6f}")
            print(f"   Features: {current_features}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(registry_file), exist_ok=True)
    
    # Save updated registry
    registry_df.to_csv(registry_file, index=False)
    print(f"💾 Registry updated: {registry_file}")

def show_model_history(model_type, model_name=None, scaler_name=None, target_column=None):
    """Display model training history for regression or classification"""
    
    # Load existing registry
    registry_df = load_or_create_registry(model_type)
    
    if registry_df.empty:
        print(f"❌ No {model_type} models found in history.")
        return
    
    # Apply filters if provided
    filtered_df = registry_df.copy()
    
    if model_name:
        filtered_df = filtered_df[filtered_df['model_name'].str.contains(model_name, case=False, na=False)]
    
    if scaler_name:
        filtered_df = filtered_df[filtered_df['scaler_name'] == scaler_name]
    
    if target_column:
        filtered_df = filtered_df[filtered_df['target_column'] == target_column]
    
    if filtered_df.empty:
        print(f"❌ No {model_type} models found matching the criteria.")
        return
    
    if model_type.lower() == 'regression':
        # Prepare regression display data
        display_data = []
        
        for _, row in filtered_df.iterrows():
            # Count features
            features_list = row['features_list']
            if isinstance(features_list, str):
                num_features = len(features_list.split(',')) if features_list else 0
            else:
                num_features = 0
            
            display_data.append({
                'Model': row['model_name'],
                'Scaler': row['scaler_name'], 
                'Target': row['target_column'],
                'Features': num_features,
                'MSE': f"{row['mse']:.6f}",
                'MAE': f"{row['mae']:.6f}",
                'RMSE': f"{row['rmse']:.6f}",
                'R²': f"{row['r2']:.6f}"
            })
        
        # Convert to DataFrame and sort by target_column first, then R² (descending)
        display_df = pd.DataFrame(display_data)
        display_df['R²_numeric'] = filtered_df['r2'].values  # Keep numeric for sorting
        display_df = display_df.sort_values(['Target', 'R²_numeric'], ascending=[True, False])
        display_df = display_df.drop('R²_numeric', axis=1)  # Remove helper column
        
        print(f"\n📊 {model_type.upper()} Model Performance History")
        print("=" * 100)
        print(display_df.to_string(index=False))
        
    elif model_type.lower() == 'classification':
        # Prepare classification display data
        display_data = []
        
        for _, row in filtered_df.iterrows():
            # Count features
            features_list = row['features_list']
            if isinstance(features_list, str):
                num_features = len(features_list.split(',')) if features_list else 0
            else:
                num_features = 0
            
            # Parse classification report
            try:
                if isinstance(row['classification_report_dict'], str):
                    report = json.loads(row['classification_report_dict'])
                else:
                    report = row['classification_report_dict']
                
                accuracy = report.get('accuracy', 0)
                weighted_avg = report.get('weighted avg', {})
                precision = weighted_avg.get('precision', 0)
                recall = weighted_avg.get('recall', 0)
                f1_score = weighted_avg.get('f1-score', 0)
                
            except (json.JSONDecodeError, TypeError):
                accuracy = precision = recall = f1_score = 0
            
            display_data.append({
                'Model': row['model_name'],
                'Scaler': row['scaler_name'],
                'Target': row['target_column'],
                'Features': num_features,
                'Accuracy': f"{accuracy:.6f}",
                'Precision': f"{precision:.6f}",
                'Recall': f"{recall:.6f}",
                'F1-Score': f"{f1_score:.6f}"
            })
        
        # Convert to DataFrame and sort by target_column first, then F1 (descending)
        display_df = pd.DataFrame(display_data)
        
        # Extract numeric F1 for sorting
        f1_numeric = []
        for _, row in filtered_df.iterrows():
            try:
                if isinstance(row['classification_report_dict'], str):
                    report = json.loads(row['classification_report_dict'])
                else:
                    report = row['classification_report_dict']
                f1_numeric.append(report.get('weighted avg', {}).get('f1-score', 0))
            except:
                f1_numeric.append(0)
        
        display_df['F1_numeric'] = f1_numeric
        display_df = display_df.sort_values(['Target', 'F1_numeric'], ascending=[True, False])
        display_df = display_df.drop('F1_numeric', axis=1)  # Remove helper column
        
        print(f"\n📊 {model_type.upper()} Model Performance History")
        print("=" * 120)
        print(display_df.to_string(index=False))
    
    print(f"\n📈 Total models shown: {len(filtered_df)}")
    
    # Show best performing model
    if model_type.lower() == 'regression':
        best_model = filtered_df.loc[filtered_df['r2'].idxmax()]
        print(f"🏆 Best R² Score: {best_model['r2']:.6f} ({best_model['model_name']} + {best_model['scaler_name']} for {best_model['target_column']})")
    else:
        # Find best F1 score
        best_f1 = 0
        best_model_info = ""
        for _, row in filtered_df.iterrows():
            try:
                if isinstance(row['classification_report_dict'], str):
                    report = json.loads(row['classification_report_dict'])
                else:
                    report = row['classification_report_dict']
                f1 = report.get('weighted avg', {}).get('f1-score', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_info = f"{row['model_name']} + {row['scaler_name']} for {row['target_column']}"
            except:
                continue
        
        if best_f1 > 0:
            print(f"🏆 Best F1-Score: {best_f1:.6f} ({best_model_info})")

def show_model_parameters(model_type, model_name, scaler_name, target_column):
    """Display detailed parameters for a specific model"""
    
    # Load existing registry
    registry_df = load_or_create_registry(model_type)
    
    if registry_df.empty:
        print(f"❌ No {model_type} models found in history.")
        return
    
    # Find the specific model
    mask = (
        (registry_df['model_name'] == model_name) &
        (registry_df['scaler_name'] == scaler_name) &
        (registry_df['target_column'] == target_column)
    )
    
    model_row = registry_df[mask]
    
    if model_row.empty:
        print(f"❌ Model not found: {model_name} + {scaler_name} for {target_column}")
        print(f"Available combinations for {model_type}:")
        
        # Show available combinations
        available_combinations = registry_df[['model_name', 'scaler_name', 'target_column']].drop_duplicates()
        for _, row in available_combinations.iterrows():
            print(f"   • {row['model_name']} + {row['scaler_name']} for {row['target_column']}")
        return
    
    model_data = model_row.iloc[0]
    
    print(f"\n🔍 Model Parameters Details")
    print("=" * 80)
    print(f"📋 Model Name: {model_data['model_name']}")
    print(f"⚖️  Scaler: {model_data['scaler_name']}")
    print(f"🎯 Target: {model_data['target_column']}")
    
    # Show features
    features_list = model_data['features_list']
    if isinstance(features_list, str) and features_list:
        features = features_list.split(',')
        print(f"📊 Features ({len(features)}):")
        for i, feature in enumerate(features, 1):
            print(f"   {i:2d}. {feature.strip()}")
    else:
        print(f"📊 Features: Not specified")
    
    # Show performance metrics
    if model_type.lower() == 'regression':
        print(f"\n📈 Performance Metrics:")
        print(f"   MSE:  {model_data['mse']:.6f}")
        print(f"   MAE:  {model_data['mae']:.6f}")
        print(f"   RMSE: {model_data['rmse']:.6f}")
        print(f"   R²:   {model_data['r2']:.6f}")
        
    elif model_type.lower() == 'classification':
        print(f"\n📈 Performance Metrics:")
        
        # Parse and display classification report
        try:
            if isinstance(model_data['classification_report_dict'], str):
                report = json.loads(model_data['classification_report_dict'])
            else:
                report = model_data['classification_report_dict']
            
            print(f"   Accuracy: {report.get('accuracy', 0):.6f}")
            
            # Show weighted averages
            weighted_avg = report.get('weighted avg', {})
            if weighted_avg:
                print(f"   Weighted Avg Precision: {weighted_avg.get('precision', 0):.6f}")
                print(f"   Weighted Avg Recall:    {weighted_avg.get('recall', 0):.6f}")
                print(f"   Weighted Avg F1-Score:  {weighted_avg.get('f1-score', 0):.6f}")
            
            # Show per-class metrics
            print(f"\n   📊 Per-Class Metrics:")
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg'] and isinstance(metrics, dict):
                    print(f"      {class_name}:")
                    print(f"         Precision: {metrics.get('precision', 0):.4f}")
                    print(f"         Recall:    {metrics.get('recall', 0):.4f}")
                    print(f"         F1-Score:  {metrics.get('f1-score', 0):.4f}")
                    print(f"         Support:   {metrics.get('support', 0)}")
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"   ❌ Error parsing classification report: {e}")
        
        # Show AUC scores
        try:
            if isinstance(model_data['auc_score_dict'], str):
                auc_dict = json.loads(model_data['auc_score_dict'])
            else:
                auc_dict = model_data['auc_score_dict']
            
            print(f"\n   🎯 AUC Scores:")
            for class_name, auc_score in auc_dict.items():
                if class_name == 'micro-avg':
                    print(f"      Micro-Average: {auc_score:.6f}")
                elif class_name == 'macro-avg':
                    print(f"      Macro-Average: {auc_score:.6f}")
                else:
                    print(f"      {class_name}: {auc_score:.6f}")
                    
        except (json.JSONDecodeError, TypeError) as e:
            print(f"   ❌ Error parsing AUC scores: {e}")
    
    # Show model hyperparameters
    print(f"\n⚙️  Model Hyperparameters:")
    try:
        if isinstance(model_data['parameters'], str):
            parameters = json.loads(model_data['parameters'])
        else:
            parameters = model_data['parameters']
        
        if isinstance(parameters, dict):
            for param_name, param_value in parameters.items():
                print(f"   {param_name}: {param_value}")
        else:
            print(f"   {parameters}")
            
    except (json.JSONDecodeError, TypeError) as e:
        print(f"   ❌ Error parsing parameters: {e}")
        print(f"   Raw parameters: {model_data['parameters']}")
    
    print("=" * 80)


