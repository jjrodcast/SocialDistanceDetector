import numpy as np
import cv2

def __generate_partial_image(picture, partial_image, position):
    if not isinstance(position, tuple):
        raise Exception("position must be a tuple representing x,y coordinates")
    
    image_height, image_width = partial_image.shape[:2]
    x, y = position
    picture[x: x + image_height, y: y + image_width] = partial_image

def __generate_text(image, text, target_size, font_scale, color, thickness):
    cv2.putText(
        image,
        text,
        target_size,
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=font_scale,
        color=color,
        thickness=thickness
    )

def __generate_logo(path_image, target_size=(280,100)):
    img_logo = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
    img_logo = cv2.resize(img_logo, target_size)
    return img_logo

def generate_bird_eye_view(good, bad):
    red = (255,0,0)
    green = (0,255,0)
    target_size = (600, 1000)

    # Background size
    background = np.zeros((3000, 4500, 3), dtype=np.uint8)

    # Points that respect the distance
    for point in good:
        cv2.circle(background, tuple(point), 25, green, -1)
    
    # Points that don't respect the distance
    for point in bad:
        cv2.circle(background, tuple(point), 25, red, -1)


    # ROI of bird eye view
    cut_posx_min, cut_posx_max = (2000, 3400)
    cut_posy_min, cut_posy_max = ( 200, 2800)

    bird_eye_view = background[cut_posy_min:cut_posy_max, 
                                cut_posx_min:cut_posx_max, 
                                :]

    # Bird Eye View resize
    bird_eye_view_resize = cv2.resize(bird_eye_view, target_size)

    return bird_eye_view_resize

def generate_picture():
    text_color = (38, 82, 133)
    target_size = (1250, 2600, 3)
    background = np.ones(target_size, dtype=np.uint8) * 150
    background[0:120,:] = 255
    background[1200:,:] = 255

    # Generate Logo
    path_logo = 'multimedia/LogoPUCP.png'
    img_logo = __generate_logo(path_logo)
    __generate_partial_image(background, img_logo, position=(10, 25))

    # Generate Title Original
    __generate_text(image=background,
                text="Detector de Distanciamiento Social",
                target_size=(400, 90),
                font_scale=3,
                color=text_color,
                thickness=4)

    # Generate Title Bird Eye View
    __generate_text(image=background,
                text="Bird's Eye View",
                target_size=(1975, 90),
                font_scale=3,
                color=text_color,
                thickness=4)

    # Generate Bottom Title
    __generate_text(image=background,
                text="Proyecto Final del curso de Computer Vision del Diplomado de Inteligencia Artificial",
                target_size=(10, 1230),
                font_scale=1.2,
                color=text_color,
                thickness=2)

    picture = cv2.copyMakeBorder(background,2,2,2,2, cv2.BORDER_CONSTANT)
    return picture

def generate_content_view(picture, image, bird_eye_view):
    
    content = picture.copy()

    # Orginal View
    __generate_partial_image(content, image, position=(120, 0))

    # Bird Eye View
    __generate_partial_image(content, bird_eye_view, position=(160, 1960))

    return content