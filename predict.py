import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

def load_and_prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def predict_face(model, img_array):
    predictions = model.predict(img_array)
    return "La imagen contiene un rostro." if predictions[0][0] > 0.5 else "La imagen no contiene un rostro."

def main():
    model_path = 'modelo/face_detected.h5'
    img_path = 'modelo/202595.jpg'

    model = load_model(model_path)
    img_array = load_and_prepare_image(img_path)

    prediction = predict_face(model, img_array)
    print(prediction)

    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(prediction)
    plt.show()

if __name__ == "__main__":
    main()
