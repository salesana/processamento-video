#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip> // For formatted output

using namespace std;

int main() {
    string file_path;

    // Ask user for the file path
    cout << "Enter the path to the .bin file: ";
    cin >> file_path;

    // Open the binary file
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open file '" << file_path << "'." << endl;
        return -1;
    }

    vector<double> buffer;
    const size_t VECTOR_SIZE = 100 * 100; // Assuming 100x100 feature vectors

    cout << "Reading file '" << file_path << "'..." << endl;

    // Read and print each vector
    int vector_count = 0;
    while (!file.eof()) {
        vector<double> vec(VECTOR_SIZE); // A single vector of size 10,000
        file.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(double));

        // Check if valid data was read
        if (file.gcount() > 0) {
            ++vector_count;
            cout << "Vector " << vector_count << ": [";
            for (size_t i = 0; i < vec.size(); ++i) {
                cout << fixed << setprecision(4) << vec[i]; // Print each value with 4 decimal places
                if (i < vec.size() - 1) cout << ", ";
                if (i % 10 == 9) cout << endl; // BBreak after every 10 elements for readability
            }
            cout << "]" << endl;
        }
    }

    file.close();
    cout << "Finished reading. Total vectors: " << vector_count << endl;

    return 0;
}
