# The third phase involves collecting and annotating floor plan images with architectural elements. After preprocessing, a CNN model is trained for spatial recognition. Viability constraints, such as rules for door placement and pathway dynamics, are defined. The trained model is integrated with the decision tree logic to modify floor plans based on spatial recognition. The modifications are validated against architectural constraints, and testing and iteration help refine the model, decision tree, and architectural maze floor plan.
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Step 2: Organize Annotated Data
# (images in one folder, annotations in another)

# Define paths to the dataset
dataset_path = 'Existing Floorplan.jpg'
images_path = os.path.join(dataset_path, 'Floor Plan Dataset')
annotations_path = os.path.join(dataset_path, 'Floor Plan Annotations')

# Step 3: Data Preprocessing
# Load images and annotations, resize images, normalize pixel values, and prepare labels
def preprocess_data(images_path, annotations_path):
    images = []
    labels = []

    for annotation_file in os.listdir(annotations_path):
        image_file = annotation_file.replace('.xml', '.jpg')
        image_path = os.path.join(images_path, image_file)

        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Adjust size as needed
        image = image / 255.0  # Normalize pixel values

        # Load and preprocess the annotation (replace this with your annotation processing logic)
        annotation = process_annotation(os.path.join(annotations_path, annotation_file))

        images.append(image)
        labels.append(annotation)

    return np.array(images), np.array(labels)


# Step 4: Split Dataset
# Split the dataset into training and validation sets
def split_dataset(images, labels):
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2,
                                                                          random_state=42)
    return train_images, val_images, train_labels, val_labels


# Step 5: Build Convolutional Neural Network (CNN) Model
# Define a simple CNN model using Keras
def build_model(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# Step 6: Compile Model
# Compile the model with an appropriate loss function, optimizer, and metrics
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# Step 7: Train Model
# Train the model using the annotated dataset
def train_model(model, train_images, train_labels, val_images, val_labels, num_epochs=10):
    # Add data augmentation for improved generalization
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Checkpoint to save the best model during training
    checkpoint = ModelCheckpoint('spatial_model_best.h5', save_best_only=True)

    # Train the model
    model.fit(datagen.flow(train_images, train_labels, batch_size=32),
              epochs=num_epochs, validation_data=(val_images, val_labels), callbacks=[checkpoint])


# Step 8: Evaluate Model
# Evaluate the trained model on the validation set
def evaluate_model(model, val_images, val_labels):
    model.evaluate(val_images, val_labels)

# Step 9: Save Model
# Save the trained model for later use
def save_model(model, filename='spatial_model.h5'):
    model.save(filename)

# Step 10: Main Execution
# Execute the steps in sequence

# Step A: Organize Annotated Data
images, labels = preprocess_data(images_path, annotations_path)

# Step B: Split Dataset
train_images, val_images, train_labels, val_labels = split_dataset(images, labels)

# Step C: Build Convolutional Neural Network (CNN) Model
input_shape = (224, 224, 3)  # Adjust according to your image size and channels
num_classes = len(np.unique(labels))
model = build_model(input_shape, num_classes)

# Step D: Compile Model
compile_model(model)

# Step E: Train Model
train_model(model, train_images, train_labels, val_images, val_labels)

# Step F: Evaluate Model
evaluate_model(model, val_images, val_labels)

# Step G: Save Model
save_model(model)

# Step 11: Phase 3 - Floor Plan Modification

# Define architectural constraints (Replace with your specific constraints)
architectural_constraints = {
    "Confusing Intersections": {"Threshold": 3},
    "Dynamic Pathways": {"Threshold": 2},
    "Misleading Signage": {"Threshold": 1},
    "Hidden Passages": {"Threshold": 2},
    "Variable Path Width": {"Threshold": 2},
}


# Update Decision Tree Logic
def update_decision_tree_with_constraints(constraints):
    for constraint, config in constraints.items():
        threshold = config["Threshold"]

        # Add or modify decision nodes based on the constraints
        # Modify decision_tree dictionary as needed

        # Example: Adding a decision node for confusing intersections
        if constraint == "Confusing Intersections":
            decision_tree["Node X"] = {"Intersection": {"Easy": "Standard Intersection",
                                                        "Moderate": "Confusing Intersection",
                                                        "Difficult": "Highly Confusing Intersection"}}


# Integrate Viability Checks with Model Output
def integrate_constraints_with_model_output(phase_1_output):
    spatial_category, intensity, quantitative_factor = phase_1_output

    # Check if the modification adheres to architectural constraints
    for constraint, config in architectural_constraints.items():
        threshold = config["Threshold"]

        # Check if the modification violates the constraint
        if intensity == "Difficult" and quantitative_factor > threshold:
            # Adjust the decision tree logic to address the violation
            # Example: If violating the "Confusing Intersections" constraint, suggest an alternative pathway
            decision_tree["Node X"]["Intersection"]["Difficult"] = "Alternative Pathway"


# Validation and Testing
def validate_floor_plans():
    # Implement validation logic based on architectural constraints
    # This may involve simulations, visual inspections, or other methods
    pass


# Iterate on the Model and Decision Tree Integration
def iterate_on_integration():
    # Implement iteration logic based on feedback and performance
    pass


# Documentation
def document_architectural_constraints():
    # Document the architectural constraints, their translation into decision tree logic, and implemented viability checks
    # This documentation is crucial for understanding the rules governing the modification process
    pass


# Step 12: Main Execution - Phase 3

# Step H: Update Decision Tree Logic with Architectural Constraints
update_decision_tree_with_constraints(architectural_constraints)

# Step I: Integrate Viability Checks with Model Output
phase_1_output = ["Walking", "Difficult", 3]  # Replace with actual output from Phase 1
integrate_constraints_with_model_output(phase_1_output)

# Step J: Validation and Testing
validate_floor_plans()

# Step K: Iterate on the Model and Decision Tree Integration
iterate_on_integration()

# Step L: Documentation
document_architectural_constraints()
