#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

double vectorSum(vector<double> vec) 
{
    double sum = 0;
    for (int i = 0; i < vec.size(); i++) 
    {
        sum += vec[i];
    }
    return sum;
}

double vectorMean(vector<double> vec) 
{
    double sum = 0;
    for (int i = 0; i < vec.size(); i++) 
    {
        sum += vec[i];
    }
    return sum / vec.size();
}

double vectorMedian(vector<double> vec) 
{
    sort(vec.begin(), vec.end());
    if (vec.size() % 2 == 0) 
    {
        // even array take two middle values and average to get median
        return (vec[vec.size() / 2] + vec[(vec.size() - 1) / 2]) / 2;
    } 
    else 
    {
        // odd array has median element at middle of list always
        return vec[vec.size() / 2];
    }
}

tuple<double, double> vectorRange(vector<double> vec)
{
    // subtract greatest value from least value will give range
    double minElement = INFINITY;
    double maxElement = -INFINITY;
    for (int i = 0; i < vec.size(); i++)
    {
        // compare min
        if (vec[i] < minElement)
        {
            minElement = vec[i];
        }
        // compare max
        if (vec[i] > maxElement)
        {
            maxElement = vec[i];
        }
    }
    return {minElement, maxElement};
}

void print_stats(vector<double> vec)
{
    auto [min, max] = vectorRange(vec);
    cout << "\n Sum = " << vectorSum(vec) << endl;
    cout << "\n Mean = " << vectorMean(vec) << endl;
    cout << "\n Median = " << vectorMedian(vec) << endl;
    cout << "\n Range = " << min << " " << max << endl;
    cout << "-------------" << endl;
}

double covar(vector<double> vec1, vector<double> vec2, auto size)
{
    double sigma = 0;
    double mean1 = vectorMean(vec1);
    double mean2 = vectorMean(vec2);
    for (int i = 0; i < size; i++)
    {
        sigma += (vec1[i] - mean1) * (vec2[i] - mean2);
    }
    return sigma / (size - 1);
}

double cor(vector<double> vec1, vector<double> vec2, auto size) {
    double sigmaVec1 = sqrt(covar(vec1, vec1, vec1.size()));
    double sigmaVec2 = sqrt(covar(vec2, vec2, vec2.size()));
    double covariance = covar(vec1, vec2, size);
    return covariance / (sigmaVec1 * sigmaVec2);
}

int main(int argc, char** argv)
{
    ifstream inFS;
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    cout << "Opening file Boston.csv" << endl;
    inFS.open("Boston.csv");
    if (!inFS.is_open()) 
    {
        cout << "Could not open file Boston.csv for reading" << endl;
        return 1;
    }

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good()) 
    {
        // prevent weird crash where rm_in is blank char and medv is 11 (eof?)
        if (numObservations == 506) {
            break;
        }
        
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');
        
        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);
        
        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;
    cout << "Closing file Boston.csv" << endl;
    inFS.close();

    cout << "Number of records: " << numObservations << endl;
    
    cout << "\nStats for rm" << endl;
    print_stats(rm);
    cout << "\nStats for medv" << endl;
    print_stats(medv);
    cout << "\nCovariance = " << covar(rm, medv, numObservations) << endl;
    cout << "Correlation = " << cor(rm, medv, numObservations) << endl;
    cout << "\nProgram terminated.";

    return 0;
}
