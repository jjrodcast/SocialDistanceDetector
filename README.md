# SocialDistanceDetector

Proyecto Final de fin de curso de *_Desarrollo de Aplicaciones con Visión Artificial_* del Diplomado de Inteligencia Artificial de la PUCP.

## Objetivo 🚀

Este proyecto contiene una implementación para detectar en una imagen o en un video las personas que cumplen o no el distanciamiento social necesario que les ayudes a prevenir contagiarse del COVID-19.

<p align="center"> 
    <img src='results/Res4.png' alt="Resultado">
</p>

## Procedimiento 🛠️

El proceso general para detectar el distanciamiento social es el siguiente:

1. **Leer** la imagen o frame de video.
2. **Aplicar modelo XYZ** para detectar ABC.
3. **Aplicar modelo XYZ** para ABC.
4. **Repetir el proceso** para todos los rostros en la imagen.


## Pre-Requisitos 📋

(Actualizar)
Este proyecto inicialmente ha sido diseñado para poder ser ejecutado en Colab y para una optimización se activó el entorno de ejecución en GPU.
Colab ya trae instaladas muchas de las librerías utilizadas en este proyecto, solo fue necesario instalar adicionalmente:
- mtcnn (Detector de rostros: Multi-Task Cascaded Convolutional Neural Network)

```
!pip install actualizar ---------------
```

## Archivos necesarios para la ejecución 🛠️

(Falta actualizar descripciones)

📌 **MODELOS:**

* **_models/yolov3-320_** : Model CNN entrenado desde cero con Keras.

* **_models/yolov3-608_** : Modelo CNN pre-entrenado de ResNet18 en Pytorch, del cual se mantuvo la estructura de la red y se volvieron a entrenar los pesos en todas las capas.

📌 **UTILITARIOS:**

* **_utils/functions.py_** : Utilitario para incrementar en un porcentaje dato el cuadro delimitador de un rostro en una imagen.

* **_utils/view.py_** : Utilitario para detectar las caras en las imágenes o frames de videos.

📌 **ARCHIVO PRINCIPAL:**

* **_Social Distance Detector v3.ipynb_** : Notebook con las pruebas end-to-end para generar sobre imágenes y videos las predicciones de si una persona está usando o no una mascarilla.

📌 **ARCHIVOS MULTIMEDIA:**

* **_multimedia/calibration_frame.jpg_** : Imagen

* **_multimedia/TownCentre-test.mp4_** : Video


## Proceso de Ejecución ⚙️ 

* Levantar el notebook principal en Colab
* Cargar los archivos necesarios al notebook
* Validar que el Tipo de Entorno de Ejecución está en **GPU**
* Ejecutar todo el notebook

## Documentación de apoyo 📚

Landing AI Creates an AI Tool to Help Customers Monitor Social Distancing in the Workplace: https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/


## Autores ✒️

* **Jorge Rodríguez Castillo** - [Linkedin](https://www.linkedin.com/in/jorge-rodr%C3%ADguez-castillo/) - [Github](https://github.com/jjrodcast)
* **Keven Fernández Carrillo** - [Linkedin](https://www.linkedin.com/in/keven-fern%C3%A1ndez-carrillo-50b07aa2/) - [Github](https://github.com/KevenRFC)
* **David Fosca Gamarra** - [Linkedin](https://www.linkedin.com/in/davidfoscagamarra/) - [Github](https://github.com/DavidFosca)

## Licencia 📄

Este proyecto está bajo la Licencia **GNU General Public License v3.0** - mira el archivo [LICENSE.md](LICENSE.md) para más detalles.


