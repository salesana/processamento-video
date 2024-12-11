/* InfoMinds - Grupo 3 

Ana Paula Sales  -                          11201811710
Caio Henrique Falcheti Nunes                11201920936
Edson Felipe                                11201922149
Gabriel César Nápoles Campos dos Santos     11201722756

Data de última modificação: 09/12/2024

Exemplo de Chamada desta classe:

$ Cmake .
$ make 
$ ./takephotos

*/

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <ctime>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Realiza o salvamento do par (Id,name) no arquivo Id_names.csv
void saveToCSV(const string &filename, const vector<pair<int, string>> &data) {
    ofstream file(filename);
    if (file.is_open()) {
        file << "id,name\n";
        for (const auto &entry : data) {
            file << entry.first << "," << entry.second << "\n";
        }
        file.close();
    }
}

// Realiza a leitura dos pares (Id,name) a partir do arquivo Id_names.csv
vector<pair<int, string>> readFromCSV(const string &filename) {
    vector<pair<int, string>> data;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        getline(file, line); 
        while (getline(file, line)) {
            size_t commaPos = line.find(',');
            int id = stoi(line.substr(0, commaPos));
            string name = line.substr(commaPos + 1);
            data.emplace_back(id, name);
        }
        file.close();
    }
    return data;
}

int main() {

    // Especifica onde os arquivos serão salvos
    string csvFile = "./train/id-names.csv";
    string csvFileToRecog = "./train/Recog/Classifiers/id-names.csv";
    string facesPath = "./train/faces";

    vector<pair<int, string>> idNames;

    // Carrega ou inicializa o .csv
    if (fs::exists(csvFile)) {
        idNames = readFromCSV(csvFile);
    } else {
        saveToCSV(csvFile, idNames);
    }

    // Verifica se o caminho do diretório apontado existe 
    if (!fs::exists(facesPath)) {
        // Cria a estrutura de diretorios especificada, caso ela não exista
        fs::create_directories(facesPath);  
    }

    cout << "Welcome!\n\nPlease put in your ID.\n";
    cout << "If this is your first time, choose a random ID between 1-10000\n";

    int id;
    string name;
    cout << "ID: ";
    cin >> id;

    auto it = find_if(idNames.begin(), idNames.end(),
                      [id](const pair<int, string> &entry) { return entry.first == id; });

    //Verifica a partir do ID se o usuário já está cadastrado
    if (it != idNames.end()) {
        name = it->second;
        cout << "Welcome Back! " << name << "!!\n";
    } else {
        cout << "Please Enter your name: ";
        cin.ignore();
        getline(cin, name);

        // Cria o diretorio para o usuario no caminho "/train/faces"
        string personDir = facesPath + "/" + to_string(id);
        fs::create_directory(personDir);

        idNames.emplace_back(id, name);
        saveToCSV(csvFile, idNames);
        saveToCSV(csvFileToRecog, idNames);
    }

    //inicia a etapa de captura. Sugere-se a captura de 10 a 30 fotos
    cout << "\nLet's capture! Press 's' to take a picture, and 'q' to quit.\n";
    // verifica 
    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cerr << "Error: Could not open the camera.\n";
        return -1;
    }

    CascadeClassifier faceClassifier("Classifiers/haarface.xml");
    if (faceClassifier.empty()) {
        cerr << "Error: Could not load classifier cascade.\n";
        return -1;
    }

    int photosTaken = 0;
    Mat frame, gray;

    while (true) {
        //inicia a captura de fotos segundo comando do proprio usuário
        camera >> frame;
        if (frame.empty()) break;
        
        //Transformação para escala de cinza
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceClassifier.detectMultiScale(gray, faces, 1.1, 5);

        for (const auto &face : faces) {
            //Realiza o redimencionamento da foto capturada.
            rectangle(frame, face, Scalar(0, 0, 255), 2);
            Mat faceRegion = gray(face);

            if (waitKey(1) == 's' && mean(faceRegion)[0] > 50) {
                Mat resizedFace;
                resize(faceRegion, resizedFace, Size(220, 220));

                // Salva as fotos no diretório especificado 
                string filename = facesPath + "/" + to_string(id) + "/face_" + to_string(photosTaken++) + ".jpg";
                imwrite(filename, resizedFace);
                cout << photosTaken << " -> Photos taken!\n";
            }
        }

        imshow("Face Capture", frame);
        if (waitKey(1) == 'q') break;
    }
    // Encerra o acesso à camera do dispositivo
    camera.release();
    destroyAllWindows();
    return 0;
}

