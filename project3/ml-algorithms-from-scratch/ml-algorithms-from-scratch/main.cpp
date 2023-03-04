#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <math.h>

using namespace std;

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Source: https://www.johndcook.com/blog/cpp_phi/
double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

vector<vector<double>> logisticRegression(const vector<vector<double>>& data_matrix, const vector<int>& labels) {
    int data_size = static_cast<int>(data_matrix.size()); // training size
    int p_vars = static_cast<int>(data_matrix[0].size()); // predictor variables
    
    vector<double> weights(p_vars, 1.0);
    double learning_rate = 0.001;
    
    // Store coefficients
    vector<double> errors(data_size);
    vector<double> prob_vector(data_size);
    vector<double> std_errors(p_vars);
    vector<double> z_values(p_vars);
    vector<double> p_values(p_vars);
    
    // perform gradient descent
    for (int i = 0; i < 500000; i++) {
        // Calculate predicted probabilities for each observation
        for (int obs = 0; obs < data_size; obs++) {
            double weighted_sum = 0.0;
            // Calculate the weighted sum of input features
            for (int pred = 0; pred < p_vars; pred++) {
                weighted_sum += weights[pred] * data_matrix[obs][pred];
            }
            // Apply the sigmoid function to get a proper probability value
            prob_vector[obs] = sigmoid(weighted_sum);
        }
        
        // Calculate errors for each observation
        for (int obs = 0; obs < data_size; obs++) {
            // Error is the difference between actual label and predicted probability
            errors[obs] = labels[obs] - prob_vector[obs];
        }
        
        // update weights
        for (int pred = 0; pred < p_vars; pred++) {
            double sum = 0.0;
            for (int obs = 0; obs < data_size; obs++) {
                // Calculate the sum of errors * input feature values
                sum += data_matrix[obs][pred] * errors[obs];
            }
            // Update the weight value by adding the scaled sum
            weights[pred] += learning_rate * sum / data_size;
        }
    }
    
    // calculate coefficients for each predictor variable
    for (int pred = 0; pred < p_vars; pred++) {
        double sum_sq_errors = 0.0;
        // Iterate over each observation
        for (int obs = 0; obs < data_size; obs++) {
            double weighted_sum = 0.0;
            // Calculate the weighted sum of input features
            for (int pred = 0; pred < p_vars; pred++) {
                weighted_sum += weights[pred] * data_matrix[obs][pred];
            }
            // Apply the sigmoid function to get a proper probability value
            double prob = sigmoid(weighted_sum);
            // Calculate the sum of squared errors for this predictor variable
            sum_sq_errors += prob * (1.0 - prob) * data_matrix[obs][pred] * data_matrix[obs][pred];
        }
        // Calculate standard error, z-value, and p-value for this predictor variable
        std_errors[pred] = sqrt(sum_sq_errors / data_size);
        z_values[pred] = weights[pred] / std_errors[pred];
        // p-value = 2 * (1 - |0.5 - Φ(z)|)
        p_values[pred] = 2.0 * (1 - abs(0.5 - phi(z_values[pred])));
    }
    
    // combine coefficients into reasonable output data type (vector<vector<double>>) to represent coefficients for each predictor variable (including intercept)
    vector<vector<double>> coefficients(p_vars, vector<double>(4));
    for (int pred = 0; pred < p_vars; pred++) {
        coefficients[pred][0] = weights[pred]; // aka estimate
        coefficients[pred][1] = std_errors[pred];
        coefficients[pred][2] = z_values[pred];
        coefficients[pred][3] = p_values[pred];
    }
    
    return coefficients;
}

double accuracy(const vector<int>& predicted, const vector<int>& actual) {
    int length = static_cast<int>(predicted.size());
    int correct = 0;
    for (int i = 0; i < length; i++) {
        if (predicted[i] == actual[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / length;
}

double sensitivity(const vector<int>& predicted, const vector<int>& actual) {
    int length = static_cast<int>(predicted.size());
    int predicted_positives = 0;
    int actual_positives = 0;
    for (int i = 0; i < length; i++) {
        if (actual[i] == 1) {
            actual_positives++;
            if (predicted[i] == 1) {
                predicted_positives++;
            }
        }
    }
    if (actual_positives == 0) {
        return 0.0;
    } else {
        return static_cast<double>(predicted_positives) / actual_positives;
    }
}

double specificity(const vector<int>& predicted, const vector<int>& actual) {
    int length = static_cast<int>(predicted.size());
    int predicted_negatives = 0;
    int actual_negatives = 0;
    for (int i = 0; i < length; i++) {
        if (actual[i] == 0) {
            actual_negatives++;
            if (predicted[i] == 0) {
                predicted_negatives++;
            }
        }
    }
    if (actual_negatives == 0) {
        return 0.0;
    } else {
        return static_cast<double>(predicted_negatives) / actual_negatives;
    }
}

int main(int argc, const char * argv[]) {
    ifstream inFS;
    string line;
    string key_in, pclass_in, survived_in, sex_in, age_in;
    
    const int MAX_LEN = 1050;
    vector<int> key(MAX_LEN);
    vector<int> pclass(MAX_LEN);
    vector<int> survived(MAX_LEN);
    vector<int> sex(MAX_LEN);
    vector<int> age(MAX_LEN);

    cout << "Opening file titanic_project.csv..." << endl;
    inFS.open("titanic_project.csv");
    if (!inFS.is_open())
    {
        cout << "Could not open file titanic_project.csv for reading." << endl;
        return 1;
    }
    
    cout << "Populating vectors..." << endl;
    getline(inFS, line);
    
    int num_observations = 0;
    while (inFS.good())
    {
        getline(inFS, key_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');
        
        if (key_in.front() == '"') {
            key_in.erase(0, 1);
            key_in.erase(key_in.size() - 1);
        }
        
        key.at(num_observations) = stoi(key_in);
        pclass.at(num_observations) = stoi(pclass_in);
        survived.at(num_observations) = stoi(survived_in);
        sex.at(num_observations) = stoi(sex_in);
        age.at(num_observations) = stoi(age_in);
        
        num_observations++;
    }

    key.resize(num_observations);
    pclass.resize(num_observations);
    survived.resize(num_observations);
    sex.resize(num_observations);
    age.resize(num_observations);

    cout << "Closing file titanic_project.csv..." << endl;
    inFS.close();
    
    cout << "Running logistic regression and calculating coefficients, please wait..." << endl;
    
    // use only first 800 observations and 1 predictor variable (sex) + intercept
    int train_size = 800;
    int p_vars = 2;
    vector<vector<double>> train_data(train_size, vector<double>(p_vars));
    vector<int> train_labels(train_size);
    for (int i = 0; i < train_size; i++) {
        train_data[i][0] = 1.0; // intercept
        train_data[i][1] = static_cast<double>(sex[i]); // train on sex variable
        train_labels[i] = survived[i];
    }
    
    // run logistic regression, time it, and return coefficients
    auto start_time = chrono::high_resolution_clock::now();
    vector<vector<double>> coefficients = logisticRegression(train_data, train_labels);
    auto end_time = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "\nTraining Time: " << duration.count() / 1000.0 << " Seconds" << endl;
    
    // Output coefficients
    cout << "\nEstimate\t\tStd. Error\t\tz value\t\tPr(>|z|)" << endl;
    cout << coefficients[0][0] << "\t\t" << coefficients[0][1] << "\t\t" << coefficients[0][2] << "\t\t" << coefficients[0][3] << "\t\t(Intercept)" << endl;
    cout << coefficients[1][0] << "\t\t" << coefficients[1][1] << "\t\t" << coefficients[1][2] << "\t\t" << coefficients[1][3] << "\t\tsex" << endl;
    
    // use the remaining data to predict values
    vector<vector<double>> test_data(num_observations - train_size, vector<double>(3));
    vector<int> test_labels(num_observations - train_size);
    for (int obs = 800; obs < num_observations; obs++) {
        test_data[obs - 800][0] = sex[obs];
        test_data[obs - 800][1] = age[obs];
        test_data[obs - 800][2] = pclass[obs];
        test_labels[obs - 800] = survived[obs];
    }
    
    int test_size = static_cast<int>(test_data.size());
    vector<double> weights = coefficients[0];
    vector<double> prob_vector(test_size);

    // calculate predicted probabilities for each observation from the test dataset
    for (int obs = 0; obs < test_size; obs++) {
        double prob = 0.0;
        for (int i = 0; i < weights.size(); i++) {
            prob += weights[i] * test_data[obs][i];
        }
        prob_vector[obs] = sigmoid(prob);
    }

    // convert predicted probabilities to binary prediction (0 or 1)
    vector<int> predictions(test_size);
    for (int obs = 0; obs < test_size; obs++) {
        if (prob_vector[obs] >= 0.5) {
            predictions[obs] = 1;
        } else {
            predictions[obs] = 0;
        }
    }
    
    // calculate evaluation metrics
    double acc = accuracy(predictions, test_labels);
    double sens = sensitivity(predictions, test_labels);
    double spec = specificity(predictions, test_labels);

    // Output evaluation metrics
    cout << "\nAccuracy: " << acc << endl;
    cout << "Sensitivity: " << sens << endl;
    cout << "Specificity: " << spec << endl;
    
    return 0;
}
