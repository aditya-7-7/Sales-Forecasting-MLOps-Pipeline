from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
import optuna

from feature_engineering.feature_pipeline import FeatureEngineer
from data_validation.validators import DataValidator
from ml_models.ensemble_model import EnsembleModel
from utils.mlflow_utils import MLflowManager

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = '/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator = DataValidator(config_path)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'sales',
                    date_col: str = 'date', group_cols: Optional[List[str]] = None,
                    categorical_cols: Optional[List[str]] = None):
        
        logger.info("Preparing data for training")
        
        # Skip validation for aggregated data that doesn't include product_id
        # The validation expects product-level data but we're training on store-level aggregates
        # Validate only essential columns
        required_cols = ['date', target_col]
        if group_cols:
            required_cols.extend(group_cols)
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")
        
        # Feature engineering
        df_features = self.feature_engineer.create_all_features(
            df, target_col=target_col, date_col=date_col, group_cols=group_cols, categorical_cols=categorical_cols
        )
        
        # Skip target encoding to avoid data leakage
        # Target encoding can cause overfitting in time series
        
        # Split data chronologically for time series
        df_sorted = df_features.sort_values(by=date_col)
        
        train_size = int(len(df_sorted) * (1 - self.training_config['test_size'] - self.training_config['validation_size']))
        val_size = int(len(df_sorted) * self.training_config['validation_size'])
        
        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size + val_size]
        test_df = df_sorted[train_size + val_size:]
        
        # Remove any rows with NaN in target column
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])
        
        logger.info(f"Train data split: {len(train_df)}, Validation data split: {len(val_df)}, Test data split: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, target_col: str,
                           exclude_cols: List[str] = ['date']):
        logger.info("Preprocessing features")
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
        
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        # Encode categorical variables
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train.loc[:, col] = self.encoders[col].fit_transform(X_train[col].astype(str))
            else:
                X_train.loc[:, col] = self.encoders[col].transform(X_train[col].astype(str))
            
            X_val.loc[:, col] = self.encoders[col].transform(X_val[col].astype(str))
            X_test.loc[:, col] = self.encoders[col].transform(X_test[col].astype(str))
            
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     use_optuna: bool = True):
        
        logger.info("Training XGBoost model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }
                
                params['early_stopping_rounds'] = 50
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))   
            best_params = study.best_params
            logger.info(f"Best parameters found: {best_params}")
            
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.models['xgboost'] = model
            return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      use_optuna: bool = True) -> lgb.LGBMRegressor:
        
        logger.info("Training LightGBM model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
        else:
            best_params = self.model_config['lightgbm']['params']
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(50)])
        
        self.models['lightgbm'] = model
        return model
    
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, target_col: str = 'sales',
                        use_optuna: bool = True):
        
        results = {}
        
        # Start MLflow run
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "ensemble", "use_optuna": str(use_optuna)}
        )
        
        try:
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col=target_col
            )
            
            # Log data stats
            self.mlflow_manager.log_params({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "n_features": len(self.feature_cols)
            })
            
            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna=use_optuna)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)
            
            self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
            self.mlflow_manager.log_model(xgb_model, "xgboost", 
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Top XGBoost features:\n{feature_importance.to_string(index=False)}")
            self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})" 
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})
            
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'predictions': xgb_pred
            }
            
            # Train LightGBM
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
            lgb_pred = lgb_model.predict(X_test)
            lgb_metrics = self.calculate_metrics(y_test, lgb_pred)
            
            self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
            self.mlflow_manager.log_model(lgb_model, "lightgbm",
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance for LightGBM
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top LightGBM features:\n{lgb_importance.to_string()}")
            
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': lgb_metrics,
                'predictions': lgb_pred
            }
            
            # Weighted ensemble based on individual model performance (using validation R2)
            xgb_val_pred = xgb_model.predict(X_val)
            lgb_val_pred = lgb_model.predict(X_val)
            
            xgb_val_r2 = r2_score(y_val, xgb_val_pred)
            lgb_val_r2 = r2_score(y_val, lgb_val_pred)
            
            # Calculate weights based on R2 scores with minimum weight constraint
            # This prevents a poorly performing model from being completely ignored
            min_weight = 0.2
            xgb_weight = max(min_weight, xgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
            lgb_weight = max(min_weight, lgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
            
            # Normalize weights
            total_weight = xgb_weight + lgb_weight
            xgb_weight /= total_weight
            lgb_weight /= total_weight
            
            logger.info(f"Ensemble weights - XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}")
            
            # Create ensemble model
            ensemble_weights = {
                'xgboost': xgb_weight,
                'lightgbm': lgb_weight
            }
            
            # Use a single weighted ensemble based on validation performance
            ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
            
            # Create the ensemble model object
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model
            }
                
            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)
            
            # Save ensemble model
            self.models['ensemble'] = ensemble_model
            
            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)
            
            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_model(ensemble_model, "ensemble", 
                                            input_example=X_train.iloc[:5])
            
            results['ensemble'] = {
                    'model': ensemble_model,
                    'metrics': ensemble_metrics,
                    'predictions': ensemble_pred
            }
        
            # Generate visualizations
            logger.info("Generating model comparison visualizations...")
            try:
                self._generate_and_log_visualizations(results, test_df, target_col=target_col)
            except Exception as viz_error:
                logger.error(f"Visualization generation failed: {viz_error}")
            
            # Save artifacts
            self.save_artifacts()
            
            # Get current run ID for verification
            current_run_id = mlflow.active_run().info.run_id
            
            self.mlflow_manager.end_run()
            
            # Sync artifacts to S3
            from utils.mlflow_s3_utils import MLflowS3Manager
            
            logger.info("Syncing artifacts to S3...")
            try:
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(current_run_id)
                logger.info("Successfully synced artifacts to S3")
                
                # Verify S3 artifacts after sync
                from utils.s3_verification import verify_s3_artifacts, log_s3_verification_results
                
                logger.info("Verifying S3 artifact storage...")
                verification_results = verify_s3_artifacts(
                        run_id=current_run_id,
                        expected_artifacts=[
                            'models/', 
                            'scalers.pkl', 
                            'encoders.pkl', 
                            'feature_cols.pkl',
                            'visualizations/',
                            'reports/'
                        ])
                log_s3_verification_results(verification_results)
                
                if not verification_results["success"]:
                        logger.error("S3 artifact verification failed after sync. Check logs for details")
            except Exception as e:
                logger.error(f"Failed to sync artifacts to S3: {e}")
    
        except Exception as e:        
            self.mlflow_manager.end_run(status="FAILED")
            raise e
        
        return results
        
    def save_artifacts(self):
        # Save scalers and encoders
        joblib.dump(self.scalers, '/tmp/scalers.pkl')
        joblib.dump(self.encoders, '/tmp/encoders.pkl')
        joblib.dump(self.feature_cols, '/tmp/feature_cols.pkl')
        
        # Save individual models in the expected format
        import os
        os.makedirs('/tmp/models/xgboost', exist_ok=True)
        os.makedirs('/tmp/models/lightgbm', exist_ok=True)
        os.makedirs('/tmp/models/ensemble', exist_ok=True)
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], '/tmp/models/xgboost/xgboost_model.pkl')
        
        if 'lightgbm' in self.models:
            joblib.dump(self.models['lightgbm'], '/tmp/models/lightgbm/lightgbm_model.pkl')
            
        if 'ensemble' in self.models:
            joblib.dump(self.models['ensemble'], '/tmp/models/ensemble/ensemble_model.pkl')
            
        self.mlflow_manager.log_artifacts('/tmp/')
        
        logger.info("Artifacts saved successfully")
    
    def _generate_and_log_visualizations(self, results, 
                                        test_df: pd.DataFrame, 
                                        target_col: str = 'sales') -> None:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            from ml_models.model_visualization import ModelVisualizer
            import tempfile
            import os
            
            logger.info("Starting visualization generation...")
            visualizer = ModelVisualizer()
            
            # Extract metrics
            metrics_dict = {}
            for model_name, model_results in results.items():
                if 'metrics' in model_results:
                    metrics_dict[model_name] = model_results['metrics']
            
            # Prepare predictions data
            predictions_dict = {}
            for model_name, model_results in results.items():
                if 'predictions' in model_results and model_results['predictions'] is not None:
                    pred_df = test_df[['date']].copy()
                    pred_df['prediction'] = model_results['predictions']
                    predictions_dict[model_name] = pred_df
            
            # Extract feature importance if available
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                if model_name in ['xgboost', 'lightgbm'] and 'model' in model_results:
                    model = model_results['model']
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        feature_importance_dict[model_name] = importance_df
            
            # Create temporary directory for visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating visualizations in temporary directory: {temp_dir}")
                
                # Generate all visualizations
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir
                )
                
                logger.info(f"Generated {len(saved_files)} visualization files: {list(saved_files.keys())}")
                
                # Log each visualization to MLflow
                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")
                        logger.info(f"Logged visualization: {viz_name} from {file_path}")
                    else:
                        logger.warning(f"Visualization file not found: {file_path}")
                
                # Also create a combined HTML report
                self._create_combined_html_report(saved_files, temp_dir)
                
                # Log the combined report
                combined_report = os.path.join(temp_dir, 'model_comparison_report.html')
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")
                    logger.info("Logged combined HTML report")
                    
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            # Don't fail the entire training if visualization fails