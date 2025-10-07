# ====================================================================
# TULSI PLANT DISEASE DETECTION - COMPLETE END-TO-END PROJECT
# Classes: Bacterial, Fungi, Pests, Healthy
# ====================================================================

# STEP 0 — Import All Required Libraries
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, 
                                   Dense, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

print("✅ All libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")

# ====================================================================
# STEP 1 — Dataset Setup and Extraction
# ====================================================================

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# Dataset path configuration
zip_path = 'dataset.zip'
dataset_root = "./dataset"

# Extract dataset
if not os.path.exists(dataset_root):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_root)
    print(f"✅ Extracted '{zip_path}' → {dataset_root}")
else:
    print(f"✅ Dataset already extracted at {dataset_root}")

# ====================================================================
# STEP 2 — Dataset Analysis and Preprocessing
# ====================================================================

def list_class_folders(root_dir):
    """Returns sorted list of class folders"""
    folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return sorted(folders)

def count_images(folder):
    """Count images in each class folder"""
    counts = {}
    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(valid_extensions)
            ])
    return counts

# Handle nested folder structure
classes = list_class_folders(dataset_root)
if len(classes) == 1:
    inner_dir = os.path.join(dataset_root, classes[0])
    inner_classes = list_class_folders(inner_dir)
    if len(inner_classes) >= 2:
        dataset_root = inner_dir
        classes = inner_classes

if len(classes) < 2:
    raise RuntimeError(f"❌ Found {len(classes)} class folders. Expected >= 2.")

print("📂 Disease Classes Found:", classes)
print("📊 Expected classes: ['bacterial', 'fungi', 'healthy', 'pests']")

# Count images before split
counts_before = count_images(dataset_root)
print("\n📈 Image Distribution:")
for cls, count in counts_before.items():
    print(f"  {cls}: {count} images")

# Visualization of class distribution
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(counts_before.keys(), counts_before.values(), 
               color=colors[:len(counts_before)])
plt.title('Tulsi Plant Disease Dataset - Class Distribution', fontsize=16, fontweight='bold')
plt.ylabel('Number of Images', fontsize=12)
plt.xlabel('Disease Classes', fontsize=12)

# Add value labels on bars
for bar, count in zip(bars, counts_before.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ====================================================================
# STEP 3 — Advanced Data Splitting with Stratification
# ====================================================================

def create_stratified_split(dataset_root, classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create stratified train/validation/test split"""
    
    train_dir = "data_split/train"
    val_dir = "data_split/validation"
    test_dir = "data_split/test"
    
    # Create directories
    for split_dir in [train_dir, val_dir, test_dir]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
    
    split_info = {}
    
    for cls in classes:
        cls_path = os.path.join(dataset_root, cls)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_extensions)]
        
        # Shuffle images
        np.random.shuffle(images)
        
        # Calculate split points
        total_images = len(images)
        train_split = int(total_images * train_ratio)
        val_split = int(total_images * (train_ratio + val_ratio))
        
        # Split images
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
        for img in test_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
        
        split_info[cls] = {
            'train': len(train_images),
            'validation': len(val_images),
            'test': len(test_images),
            'total': total_images
        }
    
    return train_dir, val_dir, test_dir, split_info

# Create splits if they don't exist
if not os.path.exists("data_split"):
    train_dir, val_dir, test_dir, split_info = create_stratified_split(dataset_root, classes)
    print("✅ Dataset split completed!")
    
    # Display split information
    split_df = pd.DataFrame(split_info).T
    print("\n📊 Dataset Split Summary:")
    print(split_df)
else:
    train_dir, val_dir, test_dir = "data_split/train", "data_split/validation", "data_split/test"
    print("✅ Using existing dataset split")

# ====================================================================
# STEP 4 — Advanced Data Augmentation and Generators
# ====================================================================

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = len(classes)

# Create class name to index mapping
class_names = sorted(classes)
class_indices = {name: idx for idx, name in enumerate(class_names)}
print(f"🏷️ Class Mapping: {class_indices}")

# Advanced data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1,
    fill_mode='nearest'
)

# Validation and test data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"🔄 Training samples: {train_generator.samples}")
print(f"🔄 Validation samples: {validation_generator.samples}")
print(f"🔄 Test samples: {test_generator.samples}")

# ====================================================================
# STEP 5 — Visualization of Augmented Images
# ====================================================================

def show_augmented_images(generator, num_images=8):
    """Display augmented images from generator"""
    plt.figure(figsize=(15, 8))
    
    batch_images, batch_labels = next(generator)
    
    for i in range(min(num_images, len(batch_images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(batch_images[i])
        
        # Get class name from label
        label_idx = np.argmax(batch_labels[i])
        class_name = class_names[label_idx]
        plt.title(f'Class: {class_name}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Augmented Training Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("🖼️ Sample Augmented Images:")
show_augmented_images(train_generator)

# ====================================================================
# STEP 6 — Custom CNN Architecture
# ====================================================================

def create_custom_cnn():
    """Create custom CNN for tulsi disease detection"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# ====================================================================
# STEP 7 — Transfer Learning Models
# ====================================================================

def create_vgg16_model():
    """Create VGG16 based transfer learning model"""
    base_model = VGG16(weights='imagenet', include_top=False, 
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_mobilenet_model():
    """Create MobileNetV2 based transfer learning model"""
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# ====================================================================
# STEP 8 — Model Training Setup
# ====================================================================

def compile_and_train_model(model, model_name, epochs=50):
    """Compile and train the model with callbacks"""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_{model_name}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Display model summary
    print(f"\n🏗️ {model_name} Architecture:")
    model.summary()
    
    # Train model
    print(f"\n🚀 Training {model_name}...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ====================================================================
# STEP 9 — Train Multiple Models
# ====================================================================

# Create and train models
models_to_train = {
    'Custom_CNN': create_custom_cnn(),
    'VGG16_Transfer': create_vgg16_model(),
    'MobileNet_Transfer': create_mobilenet_model()
}

trained_models = {}
training_histories = {}

for model_name, model in models_to_train.items():
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    
    trained_model, history = compile_and_train_model(model, model_name, epochs=30)
    trained_models[model_name] = trained_model
    training_histories[model_name] = history

# ====================================================================
# STEP 10 — Training Results Visualization
# ====================================================================

def plot_training_history(histories, metrics=['accuracy', 'loss']):
    """Plot training history for multiple models"""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, metric in enumerate(metrics):
        for j, (model_name, history) in enumerate(histories.items()):
            color = colors[j % len(colors)]
            
            # Training metric
            axes[i].plot(history.history[metric], 
                        label=f'{model_name} - Train', 
                        color=color, linestyle='-')
            
            # Validation metric
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[i].plot(history.history[val_metric], 
                            label=f'{model_name} - Val', 
                            color=color, linestyle='--')
        
        axes[i].set_title(f'Model {metric.capitalize()}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot training histories
plot_training_history(training_histories)

# ====================================================================
# STEP 11 — Model Evaluation and Comparison
# ====================================================================

def evaluate_model(model, model_name, test_generator):
    """Evaluate model and return metrics"""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Reset test generator
    test_generator.reset()
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    
    print(f"✅ {model_name} Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'classification_report': report,
        'confusion_matrix': cm
    }

# Evaluate all models
evaluation_results = {}
for model_name, model in trained_models.items():
    evaluation_results[model_name] = evaluate_model(model, model_name, test_generator)

# ====================================================================
# STEP 12 — Results Visualization and Analysis
# ====================================================================

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def create_comparison_report(evaluation_results):
    """Create model comparison report"""
    comparison_data = []
    
    for model_name, results in evaluation_results.items():
        report = results['classification_report']
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

# Plot confusion matrices for all models
for model_name, results in evaluation_results.items():
    plot_confusion_matrix(results['confusion_matrix'], class_names, model_name)

# Create and display comparison report
comparison_df = create_comparison_report(evaluation_results)
print("\n🏆 MODEL PERFORMANCE COMPARISON:")
print("="*60)
print(comparison_df.round(4))

# Visualize model comparison
plt.figure(figsize=(12, 8))
x = np.arange(len(comparison_df))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, comparison_df[metric], width, 
            label=metric, color=colors[i], alpha=0.8)

plt.xlabel('Models', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('Model Performance Comparison - Tulsi Disease Detection', 
          fontsize=14, fontweight='bold')
plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
print(f"\n🏅 Best performing model: {best_model_name}")
print(f"🎯 Best accuracy: {comparison_df['Accuracy'].max():.4f}")

# ====================================================================
# STEP 13 — Prediction Function and Testing
# ====================================================================

def predict_disease(image_path, model, class_names):
    """Predict disease from image path"""
    # Load and preprocess image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_predictions': dict(zip(class_names, predictions[0]))
    }

def visualize_predictions(image_paths, model, class_names, num_images=8):
    """Visualize predictions on sample images"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, img_path in enumerate(image_paths[:num_images]):
        if i >= 8:
            break
            
        # Load image
        img = load_img(img_path)
        axes[i].imshow(img)
        
        # Make prediction
        result = predict_disease(img_path, model, class_names)
        
        # Set title with prediction
        title = f"Pred: {result['predicted_class']}\n"
        title += f"Conf: {result['confidence']:.2%}"
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Disease Predictions - {best_model_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Test predictions on sample images
sample_images = []
for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    images = os.listdir(class_path)[:2]  # Take 2 images per class
    sample_images.extend([os.path.join(class_path, img) for img in images])

print(f"\n🔍 Testing predictions with {best_model_name}:")
visualize_predictions(sample_images, best_model, class_names)

# ====================================================================
# STEP 14 — Save Best Model and Create Deployment Function
# ====================================================================

# Save the best model
best_model.save('tulsi_disease_detection_best_model.h5')
print(f"💾 Best model saved as 'tulsi_disease_detection_best_model.h5'")

# Save model configuration
model_config = {
    'best_model': best_model_name,
    'class_names': class_names,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'accuracy': comparison_df.loc[comparison_df['Model'] == best_model_name, 'Accuracy'].iloc[0],
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

import json
with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print("📋 Model configuration saved as 'model_config.json'")

# ====================================================================
# STEP 15 — Create Deployment-Ready Prediction System
# ====================================================================

class TulsiDiseaseDetector:
    """Complete Tulsi Disease Detection System"""
    
    def __init__(self, model_path, config_path=None):
        self.model = load_model(model_path)
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.class_names = self.config['class_names']
                self.img_height = self.config['img_height']
                self.img_width = self.config['img_width']
        else:
            self.class_names = class_names
            self.img_height = IMG_HEIGHT
            self.img_width = IMG_WIDTH
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    
    def predict(self, image_path):
        """Make prediction and return detailed results"""
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Create detailed results
        all_predictions = {}
        for i, class_name in enumerate(self.class_names):
            all_predictions[class_name] = float(predictions[0][i])
        
        return {
            'predicted_disease': predicted_class,
            'confidence': float(confidence),
            'confidence_percentage': f"{confidence*100:.1f}%",
            'all_class_probabilities': all_predictions,
            'is_healthy': predicted_class.lower() == 'healthy',
            'needs_treatment': predicted_class.lower() in ['bacterial', 'fungi', 'pests']
        }
    
    def get_treatment_recommendation(self, prediction_result):
        """Get treatment recommendations based on prediction"""
        disease = prediction_result['predicted_disease'].lower()
        confidence = prediction_result['confidence']
        
        recommendations = {
            'healthy': "✅ Plant appears healthy! Continue regular care and monitoring.",
            'bacterial': "🦠 BACTERIAL INFECTION detected.\n" + 
                        "• Apply copper-based bactericide\n" + 
                        "• Improve air circulation\n" + 
                        "• Avoid overhead watering\n" + 
                        "• Remove infected leaves",
            'fungi': "🍄 FUNGAL INFECTION detected.\n" + 
                    "• Apply fungicide spray\n" + 
                    "• Reduce humidity around plant\n" + 
                    "• Ensure good drainage\n" + 
                    "• Remove affected parts",
            'pests': "🐛 PEST INFESTATION detected.\n" + 
                    "• Apply neem oil or insecticidal soap\n" + 
                    "• Check for insects regularly\n" + 
                    "• Use yellow sticky traps\n" + 
                    "• Quarantine if necessary"
        }
        
        base_recommendation = recommendations.get(disease, "Unknown condition detected.")
        
        if confidence < 0.6:
            base_recommendation += f"\n\n⚠️ Note: Prediction confidence is {confidence*100:.1f}%. Consider consulting an expert for confirmation."
        
        return base_recommendation

# Create detector instance
detector = TulsiDiseaseDetector('tulsi_disease_detection_best_model.h5', 'model_config.json')

# Test the complete system
print("\n🌿 TULSI DISEASE DETECTION SYSTEM - READY FOR DEPLOYMENT!")
print("="*70)

# Test on a sample image
if sample_images:
    test_image = sample_images[0]
    result = detector.predict(test_image)
    recommendation = detector.get_treatment_recommendation(result)
    
    print(f"\n📸 Test Image: {os.path.basename(test_image)}")
    print(f"🏥 Diagnosis: {result['predicted_disease']}")
    print(f"🎯 Confidence: {result['confidence_percentage']}")
    print(f"📋 Treatment Recommendation:")
    print(recommendation)

print("\n" + "="*70)
print("🎉 TULSI DISEASE DETECTION PROJECT COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nProject Summary:")
print(f"✅ Models Trained: {len(trained_models)}")
print(f"✅ Best Model: {best_model_name}")
print(f"✅ Best Accuracy: {comparison_df['Accuracy'].max():.2%}")
print(f"✅ Classes Detected: {', '.join(class_names)}")
print("✅ Deployment System: Ready")
print("✅ Treatment Recommendations: Integrated")
print("\nFiles Generated:")
print("📁 tulsi_disease_detection_best_model.h5")
print("📁 model_config.json")
print("📁 Multiple model checkpoints")