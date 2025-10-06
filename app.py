from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
from src.data_processor import ExoplanetDataProcessor
from src.models import ExoplanetClassifier, ModelEnsemble
from src.habitability import HabitabilityScorer
from src.light_curve_viz import LightCurveGenerator, generate_discovery_story
from src.planet_characterization import PlanetCharacterizer
from src.realtime_tess import TESSRealtimeFetcher
from src.uncertainty_quantification import UncertaintyQuantifier
from src.adversarial_testing import AdversarialTestGenerator
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['MODELS_FOLDER'] = 'models'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Global variables for model and preprocessor
current_model = None
preprocessor = ExoplanetDataProcessor()
uncertainty_quantifier = None  # Will be initialized after training
model_stats = {
    'trained': False,
    'model_type': None,
    'metrics': {},
    'feature_importance': None
}


def load_saved_model():
    """Load previously saved model and preprocessor from disk."""
    global current_model, preprocessor

    model_path = os.path.join(app.config['MODELS_FOLDER'], 'exoplanet_model.pkl')
    preprocessor_path = os.path.join(app.config['MODELS_FOLDER'], 'preprocessor.pkl')

    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        try:
            print("Loading saved model from disk...")
            current_model = ExoplanetClassifier.load_model(model_path)
            preprocessor.load_preprocessor(preprocessor_path)
            print("Successfully loaded saved model!")
            return True
        except Exception as e:
            print(f"Error loading saved model: {e}")
            return False
    return False


# Try to load saved model on startup
load_saved_model()


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Upload and process dataset."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and analyze data
        df = pd.read_csv(filepath)

        # Auto-detect TESS data and harmonize to Kepler format
        is_tess = False
        if 'toi' in df.columns or 'tfopwg_disp' in df.columns:
            print("\nDetected TESS data - harmonizing to Kepler format...")
            is_tess = True

            # Use the TESSRealtimeFetcher harmonization
            from realtime_tess import TESSRealtimeFetcher
            fetcher = TESSRealtimeFetcher()
            df = fetcher.harmonize_toi_to_kepler_format(df)

            # Save harmonized data
            harmonized_path = filepath.replace('.csv', '_harmonized.csv')
            df.to_csv(harmonized_path, index=False)
            filepath = harmonized_path
            print(f"Saved harmonized data to {harmonized_path}")

        # Get basic statistics
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'filepath': filepath,
            'is_tess': is_tess
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train ML model on uploaded data."""
    global current_model, preprocessor, model_stats, uncertainty_quantifier

    try:
        data = request.json
        filepath = data.get('filepath')
        model_type = data.get('model_type', 'random_forest')
        target_column = data.get('target_column', 'koi_disposition')
        test_size = data.get('test_size', 0.2)
        use_smote = data.get('use_smote', True)
        hyperparams = data.get('hyperparams', {})

        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Data file not found'}), 400

        # Load and preprocess data
        df = preprocessor.load_data(filepath, target_col=target_column)

        # Select features
        features = preprocessor.select_features(df)

        # Encode target
        df = preprocessor.encode_target(df, target_column)

        # Preprocess
        X, y = preprocessor.preprocess(df, fit=True)

        # Check if we have enough classes
        n_classes = len(np.unique(y))
        if n_classes < 2:
            return jsonify({
                'error': 'Training requires at least 2 classes (planets and non-planets). '
                        'This dataset only has 1 class. '
                        'TESS data is typically for PREDICTION only. '
                        'Please train on Kepler data first, then use TESS data for predictions.',
                'n_classes': n_classes
            }), 400

        # Handle class imbalance
        if use_smote:
            X, y = preprocessor.handle_imbalance(X, y, method='smote')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Create and train model
        current_model = ExoplanetClassifier(model_type=model_type)
        current_model.train(X_train, y_train, X_val, y_val, **hyperparams)

        # Evaluate
        metrics = current_model.evaluate(X_test, y_test)

        # Get predictions for visualization
        y_pred = current_model.predict(X_test)
        y_proba = current_model.predict_proba(X_test)

        # For neural networks, find optimal threshold using Youden's J statistic
        if model_type == 'neural_net':
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
            # Youden's J statistic: maximize TPR - FPR
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            current_model.optimal_threshold = float(optimal_threshold)

            print(f"\n=== OPTIMAL THRESHOLD CALCULATION ===")
            print(f"Optimal threshold for Neural Network: {optimal_threshold:.4f}")
            print(f"At this threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")

            # Re-calculate predictions with optimal threshold
            y_pred = current_model.predict(X_test)
            metrics = current_model.evaluate(X_test, y_test)

            print(f"Updated metrics with optimal threshold:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1_score']:.4f}")
            print(f"=====================================\n")

        # FEATURE 2: Calibrate uncertainty quantification
        # Calibrate uncertainty quantification (skip for neural networks - can't be cloned by sklearn)
        if model_type != 'neural_net':
            print("\nCalibrating uncertainty quantification...")
            uncertainty_quantifier = UncertaintyQuantifier(alpha=0.1)
            uncertainty_quantifier.calibrate(current_model.model, X_train, y_train, cv=5)
            print("Uncertainty quantification calibrated!")
        else:
            print("\nSkipping uncertainty quantification for neural networks")

        # Get feature importance
        feature_importance = None
        if model_type != 'neural_net':
            # Use the actual feature columns after preprocessing (may be fewer due to removed NaN columns)
            feature_importance = current_model.get_feature_importance(preprocessor.feature_columns)

        # Save model and preprocessor
        current_model.save_model(os.path.join(app.config['MODELS_FOLDER'], 'exoplanet_model.pkl'))
        preprocessor.save_preprocessor(os.path.join(app.config['MODELS_FOLDER'], 'preprocessor.pkl'))

        # Update stats (store test data for advanced visualizations)
        # Note: X_train and y_train are stored but not serialized to JSON
        model_stats = {
            'trained': True,
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'num_features': len(features),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist(),
            'X_train': X_train,  # DataFrame - not JSON serializable
            'y_train': y_train   # Series - not JSON serializable
        }

        # Create JSON-safe version for response
        model_stats_json = {
            'trained': True,
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'num_features': len(features),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        return jsonify({
            'success': True,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_stats': model_stats_json
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Training failed:\n{error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace}), 500


@app.route('/api/download_model', methods=['GET'])
def download_model():
    """Download the currently trained model and preprocessor as a zip file."""
    global current_model, preprocessor

    if current_model is None:
        return jsonify({'error': 'No trained model available'}), 400

    try:
        import zipfile
        import tempfile
        from io import BytesIO

        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, 'model.pkl')
            current_model.save_model(model_path)

            # Save preprocessor
            preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')
            preprocessor.save_preprocessor(preprocessor_path)

            # For neural networks, also save the h5 file
            h5_path = None
            if current_model.model_type == 'neural_net':
                h5_path = os.path.join(temp_dir, 'neural_network.h5')
                current_model.model.save(h5_path)

            # Create zip file in memory
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(model_path, 'model.pkl')
                zipf.write(preprocessor_path, 'preprocessor.pkl')
                if h5_path and os.path.exists(h5_path):
                    zipf.write(h5_path, 'neural_network.h5')

                # Add model info
                info = {
                    'model_type': current_model.model_type,
                    'metrics': current_model.metrics if hasattr(current_model, 'metrics') else {},
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                info_path = os.path.join(temp_dir, 'model_info.json')
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
                zipf.write(info_path, 'model_info.json')

            # Reset file pointer to beginning
            memory_file.seek(0)

        # Send the in-memory file (temp_dir is now cleaned up)
        return send_file(
            memory_file,
            as_attachment=True,
            download_name=f'exoplanet_model_{current_model.model_type}.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    """Upload and load a pretrained model."""
    global current_model, preprocessor, model_stats

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.zip'):
            return jsonify({'error': 'Only ZIP files are supported'}), 400

        import zipfile
        import tempfile

        # Save uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'uploaded_model.zip')
            file.save(zip_path)

            # Extract zip
            extract_dir = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_dir)

            # Load model info
            info_path = os.path.join(extract_dir, 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                model_info = {'model_type': 'unknown'}

            # Load model
            model_path = os.path.join(extract_dir, 'model.pkl')
            if not os.path.exists(model_path):
                return jsonify({'error': 'Invalid model file - model.pkl not found'}), 400

            current_model = ExoplanetClassifier()
            current_model.load_model(model_path)

            # Load preprocessor
            preprocessor_path = os.path.join(extract_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                preprocessor.load_preprocessor(preprocessor_path)

            # Get feature importance for non-neural net models
            feature_importance = None
            if current_model.model_type != 'neural_net' and hasattr(preprocessor, 'feature_columns') and preprocessor.feature_columns is not None:
                try:
                    feature_importance = current_model.get_feature_importance(preprocessor.feature_columns)
                except Exception as e:
                    print(f"Warning: Could not get feature importance: {e}")

            # Update model stats
            model_stats = {
                'trained': True,
                'model_type': current_model.model_type,
                'metrics': current_model.metrics if hasattr(current_model, 'metrics') else model_info.get('metrics', {}),
                'feature_importance': feature_importance,
                'loaded_from_file': True,
                'upload_timestamp': pd.Timestamp.now().isoformat()
            }

            return jsonify({
                'success': True,
                'model_type': current_model.model_type,
                'metrics': model_stats.get('metrics', {}),
                'message': 'Model loaded successfully'
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data."""
    global current_model, preprocessor, uncertainty_quantifier

    try:
        if current_model is None:
            # Try to load saved model
            model_path = os.path.join(app.config['MODELS_FOLDER'], 'exoplanet_model.pkl')
            preprocessor_path = os.path.join(app.config['MODELS_FOLDER'], 'preprocessor.pkl')

            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                current_model = ExoplanetClassifier()
                current_model.load_model(model_path)
                preprocessor.load_preprocessor(preprocessor_path)
            else:
                return jsonify({'error': 'No trained model available'}), 400

        data = request.json

        # Handle file upload or manual input
        if 'filepath' in data:
            filepath = data['filepath']
            # Ensure filepath exists and is readable
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filepath}'}), 400
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        elif 'data' in data:
            df = pd.DataFrame([data['data']])
        else:
            return jsonify({'error': 'No data provided'}), 400

        # Preprocess
        X, _ = preprocessor.preprocess(df, fit=False)

        # Predict
        predictions = current_model.predict(X)
        probabilities = current_model.predict_proba(X)

        # Limit batch predictions to avoid timeout
        max_predictions = 100

        # Initialize visualizers and characterizers
        characterizer = PlanetCharacterizer()
        hab_scorer = HabitabilityScorer()

        results = []
        for i, (pred, proba) in enumerate(zip(predictions[:max_predictions], probabilities[:max_predictions])):
            result = {
                'index': i,
                'prediction': 'Exoplanet' if pred == 1 else 'Not Exoplanet',
                'confidence': float(proba[1]),
                'probability_exoplanet': float(proba[1]),
                'probability_not_exoplanet': float(proba[0])
            }

            #  FEATURE 2: Add uncertainty quantification
            if uncertainty_quantifier is not None:
                try:
                    _, _, intervals = uncertainty_quantifier.predict_with_intervals(
                        current_model, X.iloc[i:i+1]
                    )
                    interval_lower, interval_upper = intervals[0]
                    result['uncertainty'] = {
                        'interval_lower': float(interval_lower),
                        'interval_upper': float(interval_upper),
                        'interval_width': float(interval_upper - interval_lower),
                        'coverage': f"{(1 - uncertainty_quantifier.alpha) * 100:.0f}%"
                    }
                except Exception as e:
                    print(f"Uncertainty quantification error: {e}")

            # Add planet characterization for all detected exoplanets
            if pred == 1 and proba[1] > 0.5:
                planet_data = df.iloc[i].to_dict()

                # Extract TESS metadata if available
                tess_metadata = {}
                tess_columns = ['tess_toi', 'tess_tid', 'tess_ra', 'tess_dec',
                               'tess_toi_created', 'tess_rowupdate', 'tess_pl_orbper',
                               'tess_pl_rade', 'tess_tfopwg_disp']
                for col in tess_columns:
                    if col in planet_data and pd.notna(planet_data[col]):
                        # Remove 'tess_' prefix for cleaner display
                        clean_key = col.replace('tess_', '')
                        tess_metadata[clean_key] = planet_data[col]

                if tess_metadata:
                    result['tess_metadata'] = tess_metadata

                # Extract Kepler metadata if available
                kepler_metadata = {}
                kepler_columns = ['kepid', 'kepoi_name', 'kepler_name', 'ra', 'dec',
                                 'koi_pdisposition', 'koi_score', 'koi_period', 'koi_prad']
                for col in kepler_columns:
                    if col in planet_data and pd.notna(planet_data[col]):
                        kepler_metadata[col] = planet_data[col]

                if kepler_metadata:
                    result['kepler_metadata'] = kepler_metadata

                # Characterize the planet
                try:
                    characteristics = characterizer.characterize_planet(planet_data)
                    result['characteristics'] = characteristics
                except Exception as e:
                    print(f"Characterization error: {e}")

                # Calculate habitability for ALL exoplanets
                try:
                    hab_result = hab_scorer.calculate_habitability_score(planet_data)
                    result['habitability'] = hab_result
                except Exception as e:
                    print(f"Habitability scoring error: {e}")

                # Add row index for on-demand visualization loading
                result['row_index'] = i

            results.append(result)

        # Add summary statistics for large batches
        summary = {
            'total_samples': len(predictions),
            'samples_returned': len(results),
            'total_exoplanets': int(predictions.sum()),
            'total_non_exoplanets': int((predictions == 0).sum()),
            'avg_confidence': float(probabilities[:, 1].mean()),
            'filepath': data.get('filepath')  # Include filepath for on-demand visualization
        }

        return jsonify({
            'predictions': results,
            'summary': summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize_planet/<int:row_index>', methods=['POST'])
def visualize_planet(row_index):
    """Generate visualizations for a specific planet on-demand."""
    try:
        data = request.json
        filepath = data.get('filepath')

        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Invalid filepath'}), 400

        # Load the data
        df = pd.read_csv(filepath)

        if row_index >= len(df):
            return jsonify({'error': 'Invalid row index'}), 400

        planet_data = df.iloc[row_index].to_dict()

        # Get prediction confidence from request
        confidence = data.get('confidence', 0.5)

        # Initialize generators
        light_curve_gen = LightCurveGenerator()
        hab_scorer = HabitabilityScorer()

        result = {}

        # Generate visualizations
        try:
            result['light_curve_plot'] = light_curve_gen.create_light_curve_plot(
                planet_data,
                prediction_confidence=confidence
            )
            result['phase_folded_plot'] = light_curve_gen.create_phase_folded_plot(planet_data)
            result['comparison_chart'] = light_curve_gen.create_comparison_chart(planet_data)
        except Exception as e:
            import traceback
            print(f"Visualization error: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Visualization error: {str(e)}'}), 500

        # Generate discovery story
        try:
            hab_result = hab_scorer.calculate_habitability_score(planet_data)
            result['discovery_story'] = generate_discovery_story(
                planet_data,
                hab_result,
                confidence
            )
        except Exception as e:
            print(f"Discovery story error: {e}")

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_stats', methods=['GET'])
def get_model_stats():
    """Get current model statistics."""
    global model_stats, current_model

    if current_model is None:
        # Try to load saved model
        model_path = os.path.join(app.config['MODELS_FOLDER'], 'exoplanet_model.pkl')
        if os.path.exists(model_path):
            current_model = ExoplanetClassifier()
            current_model.load_model(model_path)
            model_stats['trained'] = True
            model_stats['metrics'] = current_model.metrics
            model_stats['model_type'] = current_model.model_type

    # Create JSON-safe copy without DataFrames
    stats_json = {}
    for key, value in model_stats.items():
        # Skip DataFrames and Series
        if hasattr(value, 'to_dict') or key in ['X_train', 'y_train', 'X_test', 'y_test_raw']:
            continue
        stats_json[key] = value

    return jsonify(stats_json)


@app.route('/api/visualization/feature_importance', methods=['GET'])
def get_feature_importance_viz():
    """Get feature importance visualization."""
    global model_stats

    if not model_stats.get('feature_importance'):
        return jsonify({'error': 'No feature importance data available'}), 400

    importance_data = model_stats['feature_importance'][:15]  # Top 15 features

    fig = go.Figure(go.Bar(
        x=[item['importance'] for item in importance_data],
        y=[item['feature'] for item in importance_data],
        orientation='h'
    ))

    fig.update_layout(
        title='Top 15 Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500
    )

    return jsonify({'plot': fig.to_json()})


@app.route('/api/visualization/confusion_matrix', methods=['GET'])
def get_confusion_matrix_viz():
    """Get confusion matrix visualization."""
    global model_stats

    if not model_stats.get('metrics', {}).get('confusion_matrix'):
        return jsonify({'error': 'No confusion matrix data available'}), 400

    cm = model_stats['metrics']['confusion_matrix']

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Not Exoplanet', 'Exoplanet'],
        y=['Not Exoplanet', 'Exoplanet'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500
    )

    return jsonify({'plot': fig.to_json()})


@app.route('/api/visualization/metrics', methods=['GET'])
def get_metrics_viz():
    """Get model metrics visualization."""
    global model_stats

    if not model_stats.get('metrics'):
        return jsonify({'error': 'No metrics data available'}), 400

    metrics = model_stats['metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1_score', 0),
        metrics.get('roc_auc', 0)
    ]

    fig = go.Figure(go.Bar(
        x=metric_names,
        y=metric_values,
        text=[f'{v:.3f}' for v in metric_values],
        textposition='auto'
    ))

    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=400
    )

    return jsonify({'plot': fig.to_json()})


@app.route('/api/visualization/roc', methods=['GET'])
def get_roc_curve():
    """Get ROC curve visualization."""
    global model_stats

    if not model_stats.get('y_test') or not model_stats.get('y_proba'):
        return jsonify({'error': 'No test data available. Please train a model first.'}), 400

    from sklearn.metrics import roc_curve, auc

    y_test = np.array(model_stats['y_test'])
    y_proba = np.array(model_stats['y_proba'])

    # Debug output
    print(f"ROC - y_test shape: {y_test.shape}, y_proba shape: {y_proba.shape}")
    print(f"ROC - y_test unique values: {np.unique(y_test)}")
    print(f"ROC - y_proba sample: {y_proba[:5]}")

    # Get probabilities for positive class
    y_proba_pos = y_proba[:, 1] if len(y_proba.shape) > 1 and y_proba.shape[1] == 2 else y_proba

    print(f"ROC - y_proba_pos min: {y_proba_pos.min()}, max: {y_proba_pos.max()}, mean: {y_proba_pos.mean()}")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_pos)
    roc_auc = auc(fpr, tpr)

    print(f"ROC - fpr points: {len(fpr)}, tpr points: {len(tpr)}, AUC: {roc_auc}")

    fig = go.Figure()

    # ROC Curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='#64c8ff', width=2)
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True
    )

    return jsonify({'plot': fig.to_json()})


@app.route('/api/visualization/precision_recall', methods=['GET'])
def get_precision_recall_curve():
    """Get Precision-Recall curve visualization."""
    global model_stats

    if not model_stats.get('y_test') or not model_stats.get('y_proba'):
        return jsonify({'error': 'No test data available. Please train a model first.'}), 400

    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_test = np.array(model_stats['y_test'])
    y_proba = np.array(model_stats['y_proba'])

    # Debug output
    print(f"PR - y_test shape: {y_test.shape}, y_proba shape: {y_proba.shape}")
    print(f"PR - y_test unique values: {np.unique(y_test)}")

    # Get probabilities for positive class
    y_proba_pos = y_proba[:, 1] if len(y_proba.shape) > 1 and y_proba.shape[1] == 2 else y_proba

    print(f"PR - y_proba_pos min: {y_proba_pos.min()}, max: {y_proba_pos.max()}, mean: {y_proba_pos.mean()}")

    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_pos)
    avg_precision = average_precision_score(y_test, y_proba_pos)

    print(f"PR - precision points: {len(precision)}, recall points: {len(recall)}, AP: {avg_precision}")

    fig = go.Figure()

    # Add filled area
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR curve (AP = {avg_precision:.3f})',
        line=dict(color='#64c8ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(100, 200, 255, 0.2)'
    ))

    # Add baseline (random classifier would have precision = fraction of positive class)
    baseline_precision = np.sum(y_test) / len(y_test)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline_precision, baseline_precision],
        mode='lines',
        name=f'Random Classifier (AP = {baseline_precision:.3f})',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis_range=[0, 1.05],
        xaxis_range=[0, 1.05],
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0')
    )

    return jsonify({'plot': fig.to_json()})


@app.route('/api/visualization/learning_curves', methods=['GET'])
def get_learning_curves():
    """Get learning curves visualization."""
    global model_stats, current_model

    # Check if training data exists (handle DataFrames properly)
    if 'X_train' not in model_stats or 'y_train' not in model_stats:
        return jsonify({'error': 'No training data available'}), 400

    if current_model is None:
        return jsonify({'error': 'No model available'}), 400

    # Learning curves not supported for neural networks (can't be cloned by sklearn)
    if current_model.model_type == 'neural_net':
        return jsonify({'error': 'Learning curves not available for Neural Networks'}), 400

    from sklearn.model_selection import learning_curve

    X_train = model_stats['X_train']
    y_train = model_stats['y_train']

    # Calculate learning curves
    # Note: n_jobs=1 to avoid Windows multiprocessing issues
    train_sizes, train_scores, val_scores = learning_curve(
        current_model.model, X_train, y_train,
        cv=5, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    # Training score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='#64c8ff', width=2)
    ))

    # Validation score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Cross-validation Score',
        line=dict(color='#90ee90', width=2)
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(100, 200, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Training ±1 std'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(144, 238, 144, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Validation ±1 std'
    ))

    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Examples',
        yaxis_title='Accuracy Score',
        yaxis_range=[0, 1.05],
        height=500,
        showlegend=True
    )

    return jsonify({'plot': fig.to_json()})


@app.route('/api/habitability', methods=['POST'])
def calculate_habitability():
    """Calculate habitability score for a planet."""
    try:
        data = request.json
        scorer = HabitabilityScorer()

        result = scorer.calculate_habitability_score(data)

        # Remove emojis for Windows compatibility
        result['classification'] = result['classification'].replace('', '[HIGHLY HABITABLE]')
        result['classification'] = result['classification'].replace('', '[HABITABLE]')
        result['classification'] = result['classification'].replace('', '[INTERESTING]')
        result['classification'] = result['classification'].replace('', '[MARGINAL]')
        result['classification'] = result['classification'].replace('', '[NOT HABITABLE]')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_csv', methods=['POST'])
def export_csv():
    """Export predictions to CSV."""
    try:
        data = request.json
        predictions = data.get('predictions', [])

        if not predictions:
            return jsonify({'error': 'No predictions to export'}), 400

        # Build CSV data
        rows = []
        for i, pred in enumerate(predictions):
            row = {
                'Sample': i + 1,
                'Prediction': pred.get('prediction', ''),
                'Confidence': f"{pred.get('confidence', 0) * 100:.2f}%",
                'Exoplanet_Probability': f"{pred.get('probability_exoplanet', 0) * 100:.2f}%"
            }

            # Add characteristics if available
            if 'characteristics' in pred:
                char = pred['characteristics']
                row['Size_Category'] = char.get('size', {}).get('category', '')
                row['Temperature_Zone'] = char.get('temperature', {}).get('category', '')
                row['Star_Type'] = char.get('star', {}).get('category', '')
                row['Composition'] = char.get('composition', {}).get('composition', '')

            # Add habitability if available
            if 'habitability' in pred:
                hab = pred['habitability']
                row['Habitability_Score'] = f"{hab.get('overall_score', 0):.2f}"
                row['Habitability_Class'] = hab.get('classification', '')

            rows.append(row)

        # Convert to DataFrame and CSV
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)

        # Return CSV as downloadable file
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': 'exoplanet_predictions.csv'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/leaderboard', methods=['POST'])
def get_leaderboard():
    """Get top habitable planets from predictions."""
    try:
        data = request.json
        predictions = data.get('predictions', [])

        # Extract planets with habitability scores
        habitable_planets = []
        for i, pred in enumerate(predictions):
            if pred.get('prediction') == 'Exoplanet' and 'habitability' in pred:
                hab = pred['habitability']
                char = pred.get('characteristics', {})

                habitable_planets.append({
                    'index': i + 1,
                    'name': f"Sample {i + 1}",
                    'score': hab.get('overall_score', 0),
                    'classification': hab.get('classification', ''),
                    'size': char.get('size', {}).get('category', 'Unknown'),
                    'temperature': char.get('temperature', {}).get('category', 'Unknown'),
                    'confidence': pred.get('confidence', 0)
                })

        # Sort by habitability score (descending)
        habitable_planets.sort(key=lambda x: x['score'], reverse=True)

        # Get top 10
        top_10 = habitable_planets[:10]

        return jsonify({
            'success': True,
            'leaderboard': top_10,
            'total_habitable': len(habitable_planets)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch_recent_tess', methods=['POST'])
def fetch_recent_tess():
    """
    FEATURE 1: Real-Time TESS Data
    Fetch recently discovered TESS planets from NASA API
    """
    try:
        data = request.json or {}
        days = data.get('lookback_days', data.get('days', 30))

        print(f"\nFetching TESS TOIs from last {days} days...")

        fetcher = TESSRealtimeFetcher()
        recent_tois = fetcher.get_recent_tois(days=days)

        if recent_tois is None or len(recent_tois) == 0:
            return jsonify({
                'success': False,
                'message': 'No recent TOIs found or API unavailable',
                'count': 0
            })

        # Harmonize for prediction
        print("Harmonizing TOI data...")
        harmonized_data = fetcher.harmonize_toi_to_kepler_format(recent_tois)
        print(f"Harmonized data shape: {harmonized_data.shape}")
        print(f"Harmonized columns: {list(harmonized_data.columns)}")

        # Preserve original TESS metadata
        metadata_columns = ['toi', 'tid', 'ra', 'dec', 'toi_created', 'rowupdate',
                           'pl_orbper', 'pl_rade', 'tfopwg_disp']
        for col in metadata_columns:
            if col in recent_tois.columns:
                harmonized_data[f'tess_{col}'] = recent_tois[col].values

        # Save temporarily
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'recent_tess_tois.csv')
        harmonized_data.to_csv(temp_file, index=False)
        print(f"Saved to {temp_file}")

        # Make predictions if model is trained
        predictions = None

        # Try to load saved model if not in memory
        if current_model is None:
            load_saved_model()

        if current_model is not None and preprocessor is not None:
            try:
                # Prepare data for prediction (align columns with training data)
                X_pred = harmonized_data.copy()

                # Remove non-feature columns
                cols_to_drop = ['kepid', 'koi_disposition', 'label']
                X_pred = X_pred.drop(columns=[col for col in cols_to_drop if col in X_pred.columns], errors='ignore')

                print(f"TESS data columns after drop: {list(X_pred.columns)}")
                print(f"TESS data shape: {X_pred.shape}")

                # Get feature names from preprocessor
                if hasattr(preprocessor, 'feature_columns') and preprocessor.feature_columns:
                    training_features = preprocessor.feature_columns
                    print(f"Training features: {len(training_features)} features")
                    print(f"First 10: {training_features[:10]}")

                    # Add missing columns with median/mean values
                    for col in training_features:
                        if col not in X_pred.columns:
                            X_pred[col] = 0  # Default value for missing features
                            print(f"Added missing column: {col}")

                    # Select only training features in the same order
                    X_pred = X_pred[training_features]
                    print(f"Aligned data shape: {X_pred.shape}")

                # Transform using the preprocessor scaler
                X_transformed = preprocessor.scaler.transform(X_pred)
                print(f"Transformed shape: {X_transformed.shape}")

                # Make predictions
                y_pred = current_model.predict(X_transformed)
                y_proba = current_model.predict_proba(X_transformed)[:, 1]

                # Create predictions list with full characterization
                predictions = []
                characterizer = PlanetCharacterizer()
                hab_scorer = HabitabilityScorer()

                for i in range(len(harmonized_data)):
                    pred_value = int(y_pred[i])
                    proba_value = float(y_proba[i])

                    result = {
                        'index': i,
                        'prediction': 'Exoplanet' if pred_value == 1 else 'Not Exoplanet',
                        'confidence': proba_value,
                        'probability_exoplanet': proba_value,
                        'probability_not_exoplanet': 1.0 - proba_value,
                        'label': 'PLANET' if pred_value == 1 else 'NOT PLANET'
                    }

                    # Add full characterization for exoplanets
                    if pred_value == 1 and proba_value > 0.5:
                        planet_data = harmonized_data.iloc[i].to_dict()

                        # Extract TESS metadata
                        tess_metadata = {}
                        tess_columns = ['tess_toi', 'tess_tid', 'tess_ra', 'tess_dec',
                                       'tess_toi_created', 'tess_rowupdate', 'tess_pl_orbper',
                                       'tess_pl_rade', 'tess_tfopwg_disp']
                        for col in tess_columns:
                            if col in planet_data and pd.notna(planet_data[col]):
                                clean_key = col.replace('tess_', '')
                                tess_metadata[clean_key] = planet_data[col]

                        if tess_metadata:
                            result['tess_metadata'] = tess_metadata

                        # Extract Kepler metadata if available (for non-TESS data)
                        kepler_metadata = {}
                        kepler_columns = ['kepid', 'kepoi_name', 'kepler_name', 'ra', 'dec',
                                         'koi_pdisposition', 'koi_score', 'koi_period', 'koi_prad']
                        for col in kepler_columns:
                            if col in planet_data and pd.notna(planet_data[col]):
                                kepler_metadata[col] = planet_data[col]

                        if kepler_metadata:
                            result['kepler_metadata'] = kepler_metadata

                        # Characterize the planet
                        try:
                            characteristics = characterizer.characterize_planet(planet_data)
                            result['characteristics'] = characteristics
                        except Exception as e:
                            print(f"Characterization error for TOI {i}: {e}")

                        # Calculate habitability
                        try:
                            hab_result = hab_scorer.calculate_habitability_score(planet_data)
                            result['habitability'] = hab_result
                        except Exception as e:
                            print(f"Habitability scoring error for TOI {i}: {e}")

                        # Add row index for on-demand visualization
                        result['row_index'] = i

                    predictions.append(result)

                print(f"Made {len(predictions)} predictions on TESS data")

            except Exception as pred_error:
                import traceback
                print(f"ERROR making predictions: {pred_error}")
                print(traceback.format_exc())
                predictions = None

        # Add summary statistics
        summary = None
        if predictions:
            exoplanet_count = sum(1 for p in predictions if p['prediction'] == 'Exoplanet')
            avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
            summary = {
                'total_samples': len(predictions),
                'samples_returned': len(predictions),
                'total_exoplanets': exoplanet_count,
                'total_non_exoplanets': len(predictions) - exoplanet_count,
                'avg_confidence': avg_confidence,
                'filepath': temp_file
            }

        return jsonify({
            'success': True,
            'count': len(recent_tois),
            'days': days,
            'filepath': temp_file,
            'predictions': predictions,
            'summary': summary,
            'message': f'Fetched {len(recent_tois)} TOIs from last {days} days'
        })

    except Exception as e:
        import traceback
        print(f"\nERROR in fetch_recent_tess:")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/generate_adversarial', methods=['POST'])
def generate_adversarial():
    """
    FEATURE 3: Adversarial Testing
    Generate fake planets to test model robustness
    """
    try:
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 20)

        print(f"\nGenerating adversarial test cases ({n_samples} per type)...")

        generator = AdversarialTestGenerator()
        generator.generate_all_adversarial_cases(n_samples=n_samples)

        adversarial_df = generator.to_dataframe()

        # Save for prediction
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'adversarial_test_suite.csv')
        adversarial_df.to_csv(temp_file, index=False)

        # Summary by type
        summary = adversarial_df.groupby('type').size().to_dict()
        types = list(summary.keys())

        return jsonify({
            'success': True,
            'count': len(adversarial_df),
            'filepath': temp_file,
            'types': types,
            'summary': summary,
            'message': f'Generated {len(adversarial_df)} adversarial test cases'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/test_adversarial', methods=['POST'])
def test_adversarial():
    """
    Test trained model against adversarial cases
    """
    global current_model, preprocessor

    try:
        if not model_stats['trained'] or current_model is None:
            return jsonify({'error': 'No trained model available'}), 400

        data = request.json
        filepath = data.get('filepath')

        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Adversarial file not found'}), 400

        print("\nTesting model against adversarial cases...")

        # Load adversarial data
        adversarial_df = pd.read_csv(filepath)
        print(f"Loaded {len(adversarial_df)} adversarial test cases")
        print(f"Columns: {list(adversarial_df.columns[:10])}")

        # Separate features from metadata (the generator uses 'type' and 'label')
        metadata_cols = ['type', 'label']
        feature_cols = [col for col in adversarial_df.columns if col not in metadata_cols]

        X_adv = adversarial_df[feature_cols]
        print(f"Feature columns in adversarial data: {len(feature_cols)}")

        # Check if preprocessor has feature columns
        if not hasattr(preprocessor, 'feature_columns') or preprocessor.feature_columns is None:
            return jsonify({'error': 'Preprocessor not properly initialized. Please train a model first.'}), 400

        print(f"Expected feature columns: {len(preprocessor.feature_columns)}")

        # Align adversarial data with training features
        # Add missing columns with default values
        for col in preprocessor.feature_columns:
            if col not in X_adv.columns:
                X_adv[col] = 0  # Default value for missing features

        # Select only the training features in the correct order
        X_adv = X_adv[preprocessor.feature_columns]
        print(f"Aligned adversarial data shape: {X_adv.shape}")

        # Preprocess (using existing fitted preprocessor)
        X_processed = preprocessor.scaler.transform(
            preprocessor.imputer.transform(X_adv)
        )

        # Predict
        predictions = current_model.predict(X_processed)
        probabilities = current_model.predict_proba(X_processed)[:, 1]

        # Add predictions to dataframe
        adversarial_df['prediction'] = predictions
        adversarial_df['confidence'] = probabilities

        # Calculate rejection rate (should reject adversarial cases)
        rejection_rate = float((predictions == 0).sum() / len(predictions))

        # Group by type
        results_by_type = {}
        for adv_type in adversarial_df['type'].unique():
            subset = adversarial_df[adversarial_df['type'] == adv_type]
            results_by_type[adv_type] = {
                'total': int(len(subset)),
                'rejected': int((subset['prediction'] == 0).sum()),
                'accepted': int((subset['prediction'] == 1).sum()),
                'rejection_rate': float((subset['prediction'] == 0).sum() / len(subset)),
                'avg_confidence': float(subset['confidence'].mean())
            }

        # Find failures (incorrectly accepted)
        failures = adversarial_df[adversarial_df['prediction'] == 1].sort_values('confidence', ascending=False)
        failures_list = []
        for idx, row in failures.head(5).iterrows():
            failures_list.append({
                'name': f"{row['type']} #{idx}",
                'type': row['type'],
                'confidence': float(row['confidence']),
                'label': int(row['label'])
            })

        # Calculate counts for frontend
        rejected_count = int((predictions == 0).sum())
        accepted_count = int((predictions == 1).sum())
        acceptance_rate = float(accepted_count / len(predictions))

        # Create sample results for display
        sample_results = []
        for idx, row in adversarial_df.iterrows():
            sample_results.append({
                'type': row['type'],
                'prediction': int(row['prediction']),
                'confidence': float(row['confidence'])
            })

        return jsonify({
            'success': True,
            'total_cases': len(adversarial_df),
            'rejected_count': rejected_count,
            'accepted_count': accepted_count,
            'rejection_rate': rejection_rate,
            'acceptance_rate': acceptance_rate,
            'by_type': results_by_type,
            'sample_results': sample_results,
            'failures': failures_list,
            'message': f'Tested {len(adversarial_df)} adversarial cases'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
