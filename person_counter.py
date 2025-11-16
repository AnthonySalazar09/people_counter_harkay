import cv2
import numpy as np
import argparse
import os

def load_yolo(cfg_path, weights_path, names_path):

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"No se encontró el archivo de clases: {names_path}")

    with open(names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    return net, output_layers, classes


def detect_people(frame, net, output_layers, classes,
                  conf_threshold=0.5, nms_threshold=0.4):
    """
    Detecta personas en un frame usando YOLO y devuelve las cajas filtradas.
    """
    (H, W) = frame.shape[:2]

    # Preprocesar imagen para la red
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=(416, 416),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    # Iterar sobre detecciones
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:] 
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]

            # Filtrar solo la clase "person" con suficiente confianza
            if confidence > conf_threshold and classes[class_id] == "person":
                # Escalar caja a tamaño original del frame
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    filtered_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            filtered_boxes.append(boxes[i])

    return filtered_boxes


def process_video(input_path, output_path,
                  cfg_path="yolov3-tiny.cfg",
                  weights_path="yolov3-tiny.weights",
                  names_path="coco.names",
                  show_window=True):
  
    print("[INFO] Cargando modelo YOLO...")
    net, output_layers, classes = load_yolo(cfg_path, weights_path, names_path)
    print("[INFO] Modelo cargado correctamente.")

    # Abrir video de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video de entrada: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    print("[INFO] Procesando video... presiona 'q' para salir (si show_window=True).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Detectar personas en el frame
        boxes = detect_people(frame, net, output_layers, classes)
        person_count = len(boxes)

        # Dibujar cajas y texto
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = f"Personas: {person_count}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        print(f"Frame {frame_index}: {person_count} personas")

        out.write(frame)

        if show_window:
            cv2.imshow("Person Counter - Harkay Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Tecla 'q' presionada. Saliendo.")
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Procesamiento terminado. Video guardado en: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector de personas con YOLOv3-tiny y OpenCV para prueba HARKAY."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="input_video.mp4",
        help="Ruta del video de entrada."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_video.mp4",
        help="Ruta del video de salida donde se guardará el resultado."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="yolov3-tiny.cfg",
        help="Ruta al archivo de configuración YOLO."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov3-tiny.weights",
        help="Ruta al archivo de pesos YOLO."
    )
    parser.add_argument(
        "--names",
        type=str,
        default="coco.names",
        help="Ruta al archivo con nombres de clases."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Si se especifica, no se mostrará la ventana de video."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    show = not args.no_show

    process_video(
        input_path=args.input,
        output_path=args.output,
        cfg_path=args.cfg,
        weights_path=args.weights,
        names_path=args.names,
        show_window=show
    )
