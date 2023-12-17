# Project Overview

This research project converges the realms of architecture, machine learning, and gamification to revolutionize maze design. Beyond its architectural significance, the project serves as a medium to unveil unethical practices prevalent in corporate architecture firms, immersing participants in an experiential simulation through an architectural maze. The innovative approach involves training a machine learning model to generatively modify the existing decision tree and floor-plan with each new qualitative dataset. Augmented reality is leveraged to visualize and immerse users in the dynamically adaptive maze, creating an environment where the spatial manifestations intricately mirror nuanced decisions derived from qualitative data.

## Phase 1: Data Collection, Machine Learning Model Training, and Prediction

**Data Collection and Annotation:**
- *Input:* Raw qualitative experiences
- *Output:* Labeled dataset with spatial categories and quantitative subcategories

**Preprocessing:**
- *Input:* Labeled dataset
- *Output:* Preprocessed textual data for model training

**Model Selection and Training:**
- *Input:* Preprocessed textual data
- *Output:* Trained text classification model

**Model Evaluation:**
- *Input:* Trained model, evaluation dataset
- *Output:* Model performance metrics (accuracy, precision, recall, F1 score)

**Prediction:**
- *Input:* Trained model, new qualitative experiences
- *Output:* Predicted spatial categories and quantitative subcategories

**Save Predictions to CSV:**
- *Input:* Predicted data
- *Output:* CSV file with predictions for further analysis

## Phase 2: Generative Decision Tree Alteration

**Analyze Existing Decision Tree:**
- *Input:* Existing decision tree structure
- *Output:* Structured data representation of the decision tree

**Integration Logic:**
- *Input:* Structured decision tree data
- *Output:* Integration logic rules for decision tree modification

**Update Decision Tree:**
- *Input:* Existing decision tree, machine learning model predictions
- *Output:* Updated decision tree structure

## Phase 3: Floor Plan Modification

**Data Collection and Annotation for Floor Plans:**
- *Input:* Raw floor plan images
- *Output:* Labeled dataset with annotations for architectural elements

**Preprocessing:**
- *Input:* Labeled floor plan images
- *Output:* Preprocessed floor plan images for model training

**Model Selection and Training:**
- *Input:* Preprocessed floor plan images
- *Output:* Trained CNN model for spatial recognition

**Viability Constraints:**
- *Input:* Trained CNN model, predefined architectural constraints
- *Output:* Decision tree modifications based on constraints

**Integration with Decision Tree:**
- *Input:* Updated decision tree, spatial recognition model output
- *Output:* Further decision tree modifications based on spatial recognition

**Validate Modifications:**
- *Input:* Modified floor plans
- *Output:* Validation results confirming adherence to architectural constraints

**Testing and Iteration:**
- *Input:* Model, decision tree, floor plans
- *Output:* Refined model, decision tree, and modified floor plans based on iterative testing

**Documentation:**
- *Input:* Decision tree modifications, architectural constraints
- *Output:* Documented floor plan modification process

## Connections Between Phases

**Phase 1 to Phase 2:**
Output of Phase 1 (Trained Text Classification Model) is used to generate predictions in Phase 2.

**Phase 2 to Phase 3:**
Output of Phase 2 (Updated Decision Tree) is used in Phase 3 for decision-making based on architectural constraints.

## Overall Output

The final output is a modified floor plan for the maze adhering to architectural constraints, generated through the integration of a spatial recognition model and a generative decision tree.
