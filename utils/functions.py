import numpy as np
import cv2
from math import sqrt

def create_model(config, weights, use_gpu=False):
    """
    Esta función se encarga de crear el modelo que detecta
    las personas en una imagen. Para esto se hace uso de 
    YOLOv3.

    Parámetros:
        config: Archivo de configuración de la red YOLOv3
        weights: Archivo de pesos de la red YOLOv3
    
    Salida:
        Retorna el modelo de YOLOv3
            
    """
    model = cv2.dnn.readNetFromDarknet(config, weights)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU
    
    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    
    return model

def get_output_layers(model):
    """
    Esta función obtiene las capas de salida de la red para poder realizar la 
    predicción en YOLOv3

    Parámetros:
        model: Modelo creado de YOLOv3, usar función create_model

    Salida:
        Retorna las capas de salida para la predicción
    """
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in model.getUnconnectedOutLayers()]
    return output_layers

def blob_from_image(image, target_size):
    """
    Esta función se encarga de crear un blob sobre la imagen o frame de video
    que se quiera predecir, se hace escalamiento [0-255] y de dimensión (416 x 416)
    ya que así lo requiere el modelo de YOLOv3

    Parámetros:
        image: Imagen de entrada RGB
        target_size: Dimensión final de la imagen (416 x 416)

    Salida:
        Retorna un blob en base a la imagen o frame de entrada        
    """
    if not isinstance(target_size, tuple): 
        raise Exception("target_size must be a tuple (width, height)")
    
    blob = cv2.dnn.blobFromImage(image, 
                                 1/255.,
                                 target_size,
                                 [0,0,0],
                                 1,
                                 crop=False)
    
    return blob

def predict(blob, model, output_layers):
    """
    Esta función realiza la predicción para detectar las personas en la escena,
    retorna la clase y los valores del bounding box

    Parámetros:
        blob: Blob generado a partir de la imagen original
        model: Modelo YOLOv3
        output_layers: Capas finales para realizar la predicción de una nueva imagen o frame
    
    Salida:
        Retorna la clase y los valores del bounding box
    """
    model.setInput(blob)
    outputs = model.forward(output_layers)

    return outputs
    
def non_maximum_suppression(image, outputs, confidence_threshold=0.6, nms_threshold=0.4):
    """
    Esta función realiza la supresión de no máxmimos, es decir que elimina aquellos bounding
    boxes que están superpuestos en base al threshold de supression y además se pone una confidencia
    para la detección de objetos en la escena

    Parámetros:
        image: Imagen original o frame de video
        outputs: Predicción realizada por el método 'predict'
        confidence_threshold: Umbral de confidencia para que considere una detección como aceptada
        nms_threshold: Umbral para eliminación de supresión de no máximos

    Salida:
        Retorna tres valores.
        1. Los nuevos bounding boxes donde están las personas.
        2. Los indices de los bounding boxes que cumplen con los parámetros de confidencia
        3. Los indices de las clases a las que pertenence dicho objeto
    """
    class_ids = []
    confidences = []
    boxes = []

    img_height, img_width = image.shape[:2]
    
    #detecting bounding boxing
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                cx = int(detection[0] * img_width)
                cy = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)
                left = int(cx - width / 2)
                top = int(cy - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    return boxes, nms_indices, class_ids

def get_domain_boxes(classes, class_ids, nms_indices, boxes, domain_class):
    """
    Esta función obtiene los bounding boxes finales que sean del mismo tipo que nuestra clase de dominio específica,
    en este caso debe ser 'person'

    Parámetros:
        classes: Clases original de COCO Dataset ya que con esas se entrenó el modelo de YOLOv3
        class_ids: Indices de la clase obtenidas luego de aplicar supresión de no máximos
        nms_indices: Indices de los bounding boxes luego de apllicar supersión de no máximos
        boxes: Bouding boxes finales luego de aplicar supresión de no máximos
        domain_class: Clase objetivo la cual deseamos los bounding boxes finales.
    
    Salida:
        Retorna la lista final de bounding boxes según la clase que deseemos, cada bounding box
        contiene las coordenadas (left, top), el ancho y altura (width, height); y el punto central
        del bounding box (cx, cy)
    """

    domain_boxes = []
    for index in nms_indices:
        idx = index[0]
        class_name = classes[class_ids[idx]]
        if class_name == domain_class:
            box = boxes[idx]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cx = left + int(width / 2)
            cy = top + int(height / 2)
            domain_boxes.append((left, top, width, height, cx, cy))
    
    return domain_boxes

def people_distances_bird_eye_view(boxes, distance_allowed):
    """
    Esta función detecta si las personas respetan la distancia social.

    Parámetros:
        boxes: Bounding boxes obtenidos luego de aplicar la función get_domain_boxes
        distance_allowed: Distancia mínima permitida para indicar si una persona respeta o no la distancia social

    Salida:
        Retorna una tupla que contiene 2 listas:
        1. La primera lista contiene la información de los puntos (personas) que respetan la distancia social.
        2. La segunda lista contiene la información de los puntos (personas) que no respetan la distancia social.
    """
    people_bad_distances = []
    people_good_distances = []
    # Tomamos los valores center,bottom
    result = __map_points_to_bird_eye_view([[box[4],box[1]+box[3]] for box in boxes])[0]
    # Creamos nuevos bounding boxes con valores mapeados de bird eye view (8 elementos por item)
    # left, top, width, height, cx, cy, bev_cy, bev_cy
    new_boxes = [box + tuple(result) for box, result in zip(boxes, result)]

    for i in range(0, len(new_boxes)-1):
        for j in range(i+1, len(new_boxes)):
            cxi,cyi = new_boxes[i][6:]
            cxj,cyj = new_boxes[j][6:]
            distance = eucledian_distance([cxi,cyi], [cxj,cyj])
            if distance < distance_allowed:
                people_bad_distances.append(new_boxes[i])
                people_bad_distances.append(new_boxes[j])

    people_good_distances = list(set(new_boxes) - set(people_bad_distances))
    people_bad_distances = list(set(people_bad_distances))
    
    return (people_good_distances, people_bad_distances)

def draw_new_image_with_boxes(image, people_good_distances, people_bad_distances, distance_allowed, draw_lines=False):
    """
    Esta función se encarga de pintar los bounding boxes y la línea entre instancias del mismo tipo para tener mejor
    entendimiento de a que distancia se encuentran.

    Parámetros:
        image: Imagen original sobre la cual se dibujarán los bounding boxes y líneas
        people_good_distances: Lista de bounding boxes que sí respetan la distancia social
        people_bad_distances: Lista de bounding boxes que no respetan la distancia social
        distance_allowed: Valor de la distancia social mínima permitida
        draw_lines: Flag (True/False) para dibujar la línea entre dos puntos, sólo para las personas que no respetan 
                    la distancia social permitida.

    Salida:
        Retorna la nueva imagen con bounding boxes dibujados y si las líneas en caso se hayan habilitado.
    """
    green = (0, 255, 0)
    red = (255, 0, 0)
    new_image = image.copy()
    
    for person in people_bad_distances:
        left, top, width, height = person[:4]
        cv2.rectangle(new_image, (left, top), (left + width, top + height), red, 2)
    
    for person in people_good_distances:
        left, top, width, height = person[:4]
        cv2.rectangle(new_image, (left, top), (left + width, top + height), green, 2)
    
    if draw_lines:
        for i in range(0, len(people_bad_distances)-1):
            for j in range(i+1, len(people_bad_distances)):
                cxi,cyi,bevxi,bevyi = people_bad_distances[i][4:]
                cxj,cyj,bevxj,bevyj = people_bad_distances[j][4:]
                distance = eucledian_distance([bevxi, bevyi], [bevxj, bevyj])
                if distance < distance_allowed:
                    cv2.line(new_image, (cxi, cyi), (cxj, cyj), red, 2)
            
    return new_image

def __matrix_bird_eye_view():
    """
    Esta función retorna los valores ya obtenidos de la matríz de homografía en pasos previos.
    """
    return np.array([[ 1.14199333e+00,  6.94076400e+00,  8.88203441e+02],
       [-5.13279159e-01,  7.26783411e+00,  1.02467130e+03],
       [ 9.79674124e-07,  1.99580075e-03,  1.00000000e+00]])

def __map_points_to_bird_eye_view(points):
    """
    Esta función realiza el mapeo de puntos de la vista original hacia la vista Bird Eye

    Parámetros:
        points: Lista bidimensional de los puntos que se desean transformar hacia Bird Eye
        
    Salida:
        Retorna los nuevos puntos pertenecientes a la vista Bird Eye
    """
    if not isinstance(points, list):
        raise Exception("poinst must be a list of type [[x1,y1],[x2,y2],...]")
    
    matrix_transformation = __matrix_bird_eye_view()
    new_points = np.array([points], dtype=np.float32)
    
    return cv2.perspectiveTransform(new_points, matrix_transformation)
    
def eucledian_distance(point1, point2):
    """
    Esta función realiza el cálculo de la distancia euclediana entre un par de puntos

    Parámetros:
        point1: Primer punto
        point2: Segundo punto

    Salida:
        Distancia euclediana entre el par de puntos dados
    """
    x1,y1 = point1
    x2,y2 = point2
    return sqrt((x1-x2)**2 + (y1-y2)**2)