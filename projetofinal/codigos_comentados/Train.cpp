/* InfoMinds - Grupo 3 

Ana Paula Sales  -                          11201811710
Caio Henrique Falcheti Nunes                11201920936
Edson Felipe                                11201922149
Gabriel César Nápoles Campos dos Santos     11201722756

Data de última modificação: 09/12/2024

Exemplo de Chamada desta classe:

$ Cmake .
$ make 
$ ./Train

*/

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

namespace fs = std::filesystem;

std::vector<int> labels;
std::vector<cv::Mat> faces;

// Realiza o carregamento dos dados de treino
void createTrain(const std::string& datasetPath) {
    for (const auto& idDir : fs::directory_iterator(datasetPath)) {
        if (!fs::is_directory(idDir)) continue;

        std::string id = idDir.path().filename().string();
        for (const auto& imgFile : fs::directory_iterator(idDir.path())) {
            try {
                // Acessa a imagem e aplica escala de cinza
                cv::Mat face = cv::imread(imgFile.path().string());
                cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);

                // Adiona os dados TRATADOS ao vetor de treino
                faces.push_back(face);
                labels.push_back(std::stoi(id));
            } catch (...) {
                std::cerr << "Error processing: " << imgFile.path() << std::endl;
            }
        }
    }
}

int main() {
    // instancia a classe LBPHFaceRecognizer em um objeto lbph
    cv::Ptr<cv::face::LBPHFaceRecognizer> lbph = cv::face::LBPHFaceRecognizer::create();
    lbph->setThreshold(50.0); // Set threshold

    // Realiza o carregamento dos dados de treinamento 
    std::cout << "Loading training data..." << std::endl;
    createTrain("faces");

    // Verifica a existência dos dados de treinamento no diretorio /train/faces
    if (faces.empty() || labels.empty()) {
        std::cerr << "No training data found!" << std::endl;
        return -1;
    }

    std::cout << "Training Started" << std::endl;
    lbph->train(faces, labels);

    // Salva o modelo treinado no arquivo e caminho especificado
    std::string modelPath = "./Recog/TrainedLBPH.yml";
    std::string modelPathAlt = "./Recog/Classifiers/TrainedLBPH.yml";
    lbph->save(modelPath);
    lbph->save(modelPathAlt);

    std::cout << "Training Complete!" << std::endl;

    return 0;
}
