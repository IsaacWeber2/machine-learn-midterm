import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, KFold # for cross validation
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Loading data, we should be able to change name of data set for different data, as long as it has enough patients
print("Loading the dataset...")
filepath = 'dataset-original.csv'   # You can modify the name of the dataset to test different data
data = pd.read_csv(filepath, low_memory=False, index_col=0)

# get types of cancer from index
cancer_types = data.index.str.extract('([A-Za-z]+)', expand=False)

# Data cleaning
data = data.loc[:, ~data.columns.duplicated()]                  # Drop duplicated columns
data = data.map(lambda x: 1 if x > 1 else (0 if x < 0 else x))  # Since the data in the expanded dataset is binary, correct outliers by setting them to 1 if x > 1 or 0 if x < 0
data = data.loc[:, data.sum(axis=0) > 0]                        # Remove all irrelevant columns which contain only zeroes
# Work in progress

all_correct = 0
all_total = 0


def select_features(data_type1, data_type2, num_features=1000):

    # Remove rare mutations (occurring less than 5 times in total)
    combined_data = pd.concat([data_type1, data_type2])
    mutation_counts = combined_data.sum(axis=0)
    frequent_mutations = mutation_counts[mutation_counts >= 5].index.tolist()
    
    # Filter data to include only frequent mutations
    data_type1_filtered = data_type1[frequent_mutations]
    data_type2_filtered = data_type2[frequent_mutations]
    freq_type1 = (data_type1_filtered > 0).mean(axis=0)
    freq_type2 = (data_type2_filtered > 0).mean(axis=0)
    freq_diff = abs(freq_type1 - freq_type2)
    
    #Selecting top features based on frequency difference
    top_features = freq_diff.sort_values(ascending=False).iloc[:num_features].index.tolist()
    
    print(f"Selected {len(top_features)} features out of {len(frequent_mutations)} frequent mutations")
    return top_features


# Building this so it is easy to change the cancer type when making changes
def run_perceptron_model(cancer_type1, cancer_type2, accuracies):
    global all_correct, all_total
    
    print(f"\n--- Perceptron Model: {cancer_type1} vs {cancer_type2} ---")
    
    #Filter based on chosen cancer type
    mask = (cancer_types == cancer_type1) | (cancer_types == cancer_type2)
    filtered_data = data[mask]
    filtered_labels = cancer_types[mask]
    
    # 0 for first, 1 for second
    labels = np.where(filtered_labels == cancer_type1, 0, 1)
    type1_indices = filtered_data[filtered_labels == cancer_type1].index
    type2_indices = filtered_data[filtered_labels == cancer_type2].index
    
    # feature selection
    data_type1 = filtered_data.loc[type1_indices]
    data_type2 = filtered_data.loc[type2_indices]
    selected_features = select_features(data_type1, data_type2, num_features=1000)
    filtered_data = filtered_data[selected_features]  # Keep only selected features
    


    # Select 200 samples for training and 200 for testing from each type
    # Split data for testing
    train_indices = np.concatenate([type1_indices[:200], type2_indices[:200]])
    test_indices = np.concatenate([type1_indices[200:400], type2_indices[200:400]])
    X_train = filtered_data.loc[train_indices]
    X_test = filtered_data.loc[test_indices]
    y_train = labels[np.isin(filtered_data.index, train_indices)]
    y_test = labels[np.isin(filtered_data.index, test_indices)]
    
    # using standard scaler for the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the perceptron
    # setting our iterations and learning rate 
    perceptron = Perceptron(max_iter=100, eta0=0.001, random_state=42) # increased the iteration to improve the fit
    perceptron.fit(X_train_scaled, y_train)

    # Cross-validation using k-fold
    kfold = KFold(n_splits=5, random_state=42, shuffle=True) # divide data into 5 parts
    # Perform cross-validation
    scores = cross_val_score(perceptron, X_test_scaled, y_test, cv=kfold)

    # Calculate and make predictions
    test_predictions = perceptron.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, test_predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f'Cross-validation scores: {scores}')
    accuracies.append((f"{cancer_type1} vs {cancer_type2}", accuracy * 100))


    # accuracy for each type
    type1_correct = sum((y_test == 0) & (test_predictions == 0))
    type1_total = sum(y_test == 0)
    type2_correct = sum((y_test == 1) & (test_predictions == 1))
    type2_total = sum(y_test == 1)
    
    print(f"{cancer_type1} Accuracy: {type1_correct / type1_total * 100:.2f}%")
    print(f"{cancer_type2} Accuracy: {type2_correct / type2_total * 100:.2f}%")
    correct_predictions = sum(y_test == test_predictions)
    total_predictions = len(y_test)
    all_correct += correct_predictions
    all_total += total_predictions



accuracies = []
run_perceptron_model('brca', 'luad', accuracies)
run_perceptron_model('brca', 'prad', accuracies)
run_perceptron_model('luad', 'prad', accuracies)

# Now plot the accuracies
def plot_accuracies(accuracies):
    labels, values = zip(*accuracies)  # Unzips the list of tuples into two lists
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='blue')
    plt.xlabel('Comparison Groups')
    plt.ylabel('Accuracy (%)')
    plt.title('Perceptron Model Accuracies')
    plt.ylim([min(values) - 5, 100])  # Sets the y-axis limits
    plt.show()

plot_accuracies(accuracies)
    
'''
# Run the three perceptron models
#print out results individually and combined
run_perceptron_model('brca', 'luad')
run_perceptron_model('brca', 'prad')
run_perceptron_model('luad', 'prad')
'''
print("\nAll models completed!")
print("\n--- Combined Results ---")
overall_accuracy = (all_correct / all_total) * 100
print(f"Total correct predictions: {all_correct} out of {all_total}")
print(f"Overall accuracy across all models: {overall_accuracy:.2f}%")