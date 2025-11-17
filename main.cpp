#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Carga nombres de clases desde coco.names
vector<string> loadClassNames(const string& classesFile) {
    vector<string> classNames;
    ifstream ifs(classesFile.c_str());
    if (!ifs.is_open()) {
        cerr << "[ERROR] No se pudo abrir el archivo de clases: " << classesFile << endl;
        return classNames;
    }

    string line;
    while (getline(ifs, line)) {
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }
    return classNames;
}

// Obtiene los nombres de las capas de salida de la red YOLO
vector<string> getOutputsNames(const Net& net) {
    static vector<string> names;
    if (names.empty()) {
        // Nombres de todas las capas
        vector<string> layersNames = net.getLayerNames();
        // Índices de capas de salida
        vector<int> outLayers = net.getUnconnectedOutLayers();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1]; // -1 por índice base 0
        }
    }
    return names;
}

int main(int argc, char** argv) {
    // Parámetros de entrada / salida por defecto
    string inputPath = "input_video.mp4";
    string outputPath = "output_video_cpp.mp4";
    string cfgPath = "yolov3-tiny.cfg";
    string weightsPath = "yolov3-tiny.weights";
    string namesPath = "coco.names";

    if (argc >= 2) {
        inputPath = argv[1];  // 1er arg: video de entrada
    }
    if (argc >= 3) {
        outputPath = argv[2]; // 2do arg: video de salida
    }

    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;

    cout << "[INFO] Cargando clases desde: " << namesPath << endl;
    vector<string> classNames = loadClassNames(namesPath);
    if (classNames.empty()) {
        cerr << "[ERROR] No se pudieron cargar las clases. Saliendo." << endl;
        return -1;
    }

    cout << "[INFO] Cargando modelo YOLO..." << endl;
    Net net;
    try {
        net = readNetFromDarknet(cfgPath, weightsPath);
    } catch (const cv::Exception& e) {
        cerr << "[ERROR] No se pudo cargar la red. Revisa las rutas de cfg/pesos." << endl;
        cerr << e.what() << endl;
        return -1;
    }

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    cout << "[INFO] Modelo cargado correctamente." << endl;

    // Abrir video de entrada
    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "[ERROR] No se pudo abrir el video de entrada: " << inputPath << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0 || std::isnan(fps)) {
        fps = 30.0; // por defecto
    }
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Video de salida
    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    VideoWriter writer(outputPath, fourcc, fps, Size(width, height));

    if (!writer.isOpened()) {
        cerr << "[ERROR] No se pudo abrir el video de salida: " << outputPath << endl;
        return -1;
    }

    cout << "[INFO] Procesando video. Presiona 'q' para salir." << endl;

    Mat frame;
    int frameIndex = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break; // fin del video
        }
        frameIndex++;

        // Crear blob para YOLO
        Mat blob;
        blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
        net.setInput(blob);

        // Forward
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        // Procesar salidas
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int classId = classIdPoint.x;

                // Filtrar solo personas
                if (confidence > confThreshold && classNames[classId] == "person") {
                    float centerX = data[0] * frame.cols;
                    float centerY = data[1] * frame.rows;
                    float widthBox = data[2] * frame.cols;
                    float heightBox = data[3] * frame.rows;
                    int left = static_cast<int>(centerX - widthBox / 2);
                    int top = static_cast<int>(centerY - heightBox / 2);

                    boxes.emplace_back(left, top,
                                       static_cast<int>(widthBox),
                                       static_cast<int>(heightBox));
                    confidences.push_back(static_cast<float>(confidence));
                    classIds.push_back(classId);
                }
            }
        }

        // Non-Max Suppression
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        int personCount = static_cast<int>(indices.size());

        // Dibujar cajas
        for (int idx : indices) {
            Rect box = boxes[idx];
            rectangle(frame, box, Scalar(0, 255, 0), 2);
        }

        // Escribir texto con conteo
        string label = "Personas: " + to_string(personCount);
        putText(frame, label, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);

        cout << "Frame " << frameIndex << ": " << personCount << " personas" << endl;

        // Guardar al video de salida
        writer.write(frame);

        // Mostrar ventana
        imshow("People Counter - C++", frame);
        char c = (char)waitKey(1);
        if (c == 'q' || c == 27) { // 'q' o ESC
            cout << "[INFO] Tecla de salida detectada. Terminando." << endl;
            break;
        }
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    cout << "[INFO] Procesamiento terminado. Video guardado en: " << outputPath << endl;
    return 0;
}
