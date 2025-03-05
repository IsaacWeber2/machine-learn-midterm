import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Loading data, we should be able to change name of data set for different data, as long as it has enough patients
print("Loading the dataset...")
data = pd.read_csv('dataset-original.csv')
data = data.set_index('Unnamed: 0')

# get types of cancer from index
cancer_types = data.index.str.extract('([A-Za-z]+)', expand=False)

all_correct = 0
all_total = 0

# Building this so it is easy to change the cancer type when making changes
def run_perceptron_model(cancer_type1, cancer_type2):
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

	  #Eta0 = Learning Rate
    #Found max iters around 100 led to more accurate results. Trying values 200+ led to less accuracy
    #Random States can change the accuracy slightly. For example 1 was 68.82% and 42 is 69.97%
    #Eta0 at .000001 (One micro) Is the smallest value before seeing no changes in accuracy
    #Convergence for max_iters happens at 7 and above. Having less than 7 leads to smaller accuracy. 7+ has no change in accuracy 
    #Best Settings Found so Far: max_iter=10, eta0=.000001, random_state=21
    #Worst Random State setting found is 25
    perceptron = Perceptron(max_iter=10, eta0=.000001, random_state=21) 
    perceptron.fit(X_train_scaled, y_train)
    
    # Calculate and make predictions
    test_predictions = perceptron.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, test_predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # accuracy for each tupe
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

# Run the three perceptron models
#print out resutls indavidualy and combined
run_perceptron_model('brca', 'luad')
run_perceptron_model('brca', 'prad')
run_perceptron_model('luad', 'prad')
print("\nAll models completed!")
print("\n--- Combined Results ---")
overall_accuracy = (all_correct / all_total) * 100
print(f"Total correct predictions: {all_correct} out of {all_total}")
print(f"Overall accuracy across all models: {overall_accuracy:.2f}%")
