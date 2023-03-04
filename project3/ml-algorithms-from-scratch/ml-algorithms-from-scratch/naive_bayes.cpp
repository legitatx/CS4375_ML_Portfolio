#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <math.h>
#include <string>
#include <unordered_map>
#include <map>

using namespace std;

class NaiveBayes {
private:
    unordered_map<int, vector<vector<double>>> summaries;
    unordered_map<int, double> class_probabilities;
    int num_columns;

public:
    NaiveBayes(int num_cols) {
        num_columns = num_cols;
    }

    void fit(vector<vector<double>> dataset, vector<int> classes) {
        int num_rows = static_cast<int>(dataset.size());

        // Split dataset by class values
        unordered_map<int, vector<vector<double>>> class_map;
        for (int i = 0; i < num_rows; i++) {
            class_map[classes[i]].push_back(dataset[i]);
        }

        // Calculate mean, stdev and count for each column in each class
        for (auto const& entry : class_map) {
            int class_value = entry.first;
            vector<vector<double>> rows = entry.second;
            class_probabilities[class_value] = (double)rows.size() / num_rows;

            vector<vector<double>> class_summaries(num_columns, vector<double>(2));
            for (auto const& row : rows) {
                for (int j = 0; j < num_columns; j++) {
                    class_summaries[j][0] += row[j];
                }
            }
            for (int j = 0; j < num_columns; j++) {
                int count = static_cast<int>(class_summaries[j].size());
                double mean = class_summaries[j][0] / count;
                class_summaries[j][0] = mean;

                double variance = 0.0;
                for (auto const& row : rows) {
                    variance += pow(row[j] - mean, 2);
                }
                double stdev = sqrt(variance / (count - 1));
                class_summaries[j][1] = stdev;
            }
            summaries[class_value] = class_summaries;
        }
    }

    double calculateProbability(double x, double mean, double stdev) {
        double exponent = exp(-pow(x - mean, 2) / (2 * pow(stdev, 2)));
        return (1 / (sqrt(2 * M_PI) * stdev)) * exponent;
    }
    
    vector<int> predict(vector<vector<double>> test_data) {
        vector<int> predictions;
        for (vector<double> row : test_data) {
            double best_probability = -1;
            int best_class = -1;

            for (auto const& entry : class_probabilities) {
                int class_value = entry.first;
                double class_probability = entry.second;

                for (int i = 0; i < num_columns; i++) {
                    double mean = summaries[class_value][i][0];
                    double stdev = summaries[class_value][i][1];
                    double x = row[i];
                    double probability = calculateProbability(x, mean, stdev);
                    class_probability *= probability;
                }

                if (best_class == -1 || class_probability > best_probability) {
                    best_probability = class_probability;
                    best_class = class_value;
                }
            }
            predictions.push_back(best_class);
        }
        return predictions;
    }

    vector<vector<double>> get_coefficients(const vector<string>& predictor_names) {
        vector<vector<double>> coeffs;
        for (auto const& entry : summaries) {
            vector<double> class_coeffs = {static_cast<double>(entry.first)};
            for (int i = 0; i < predictor_names.size(); i++) {
                class_coeffs.push_back(entry.second[i][0]);
                class_coeffs.push_back(entry.second[i][1]);
            }
            coeffs.push_back(class_coeffs);
        }

        return coeffs;
    }
};

double accuracy(const vector<int>& predicted, const vector<int>& actual) {
    int n = static_cast<int>(predicted.size());
    if (n != static_cast<int>(actual.size())) {
        throw invalid_argument("predicted and actual vectors must have the same size");
    }
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predicted[i] == actual[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / n;
}

double sensitivity(const vector<int>& predicted, const vector<int>& actual) {
    int n = static_cast<int>(predicted.size());
    if (n != static_cast<int>(actual.size())) {
        throw invalid_argument("predicted and actual vectors must have the same size");
    }
    int true_positives = 0;
    int actual_positives = 0;
    for (int i = 0; i < n; i++) {
        if (actual[i] == 1) {
            actual_positives++;
            if (predicted[i] == 1) {
                true_positives++;
            }
        }
    }
    if (actual_positives == 0) {
        return 0.0;
    } else {
        return static_cast<double>(true_positives) / actual_positives;
    }
}

double specificity(const vector<int>& predicted, const vector<int>& actual) {
    int n = static_cast<int>(predicted.size());
    if (n != static_cast<int>(actual.size())) {
        throw invalid_argument("predicted and actual vectors must have the same size");
    }
    int true_negatives = 0;
    int actual_negatives = 0;
    for (int i = 0; i < n; i++) {
        if (actual[i] == 0) {
            actual_negatives++;
            if (predicted[i] == 0) {
                true_negatives++;
            }
        }
    }
    if (actual_negatives == 0) {
        return 0.0;
    } else {
        return static_cast<double>(true_negatives) / actual_negatives;
    }
}

int main(int argc, const char * argv[]) {
    ifstream inFS;
    string line;
    string key_in, pclass_in, survived_in, sex_in, age_in;
    NaiveBayes* nb = new NaiveBayes(3);
    
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
    
    cout << "Running Naive Bayes and calculating coefficients, please wait..." << endl;
    
    // use only first 800 observations
    int train_size = 800;
    vector<vector<double>> train_data(train_size, vector<double>(3));
    vector<int> train_labels(train_size);
    for (int i = 0; i < train_size; i++) {
        train_data[i][0] = age[i];
        train_data[i][1] = pclass[i];
        train_data[i][2] = sex[i];
        train_labels[i] = survived[i];
    }

    // run logistic regression, time it, and return coefficients
    auto start_time = chrono::high_resolution_clock::now();
    nb->fit(train_data, train_labels);
    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    cout << "\nTraining Time: " << duration.count() << " Microseconds\n" << endl;
    
    // output the coefficients
    map<int, string> class_labels = {{0, "Negative"}, {1, "Positive"}};
    vector<string> predictor_names = {"age", "pclass", "sex"};
    
    vector<vector<double>> coeffs = nb->get_coefficients(predictor_names);
    for (auto const& class_coeffs : coeffs) {
        cout << "Class = " << class_labels[class_coeffs[0]] << ":" << endl;
        for (int i = 1; i < class_coeffs.size(); i += 2) {
            cout << "  Coefficient " << predictor_names[(i - 1) / 2] << ": Mean = " << class_coeffs[i] << ", "
                 << "Stdev = " << class_coeffs[i + 1] << endl;
        }
    }

    // use the remaining data to predict values
    vector<vector<double>> test_data(num_observations - train_size, vector<double>(3));
    vector<int> test_labels(num_observations - train_size);
    for (int i = 800; i < num_observations; i++) {
        test_data[i - 800][0] = age[i];
        test_data[i - 800][1] = pclass[i];
        test_data[i - 800][2] = sex[i];
        test_labels[i - 800] = survived[i];
    }
    
    vector<int> predictions = nb->predict(test_data);

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
