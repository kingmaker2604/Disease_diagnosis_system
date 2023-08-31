from rest_framework.response import Response
from rest_framework.decorators import api_view
import base64
from keras.models import load_model #for this it will be required
import cv2
import numpy as np


@api_view(['POST'])
def getPredForXrayImg(req):
    base64_string = str(req.data['image'])
    #print(base64_string)
    if not base64_string:
        return Response({'error': 'Image not provided'}, status=400)

    image_bytes = base64.b64decode(base64_string.split(",")[1])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)


    # img = Image.open(BytesIO(image_bytes))
    # print(os.getcwd())
    model = load_model('xray/xray_model.h5')
    #img = cv2.imread(image)a

    # Resize the image to match the input shape of the model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    gray = gray.astype(np.float32)/255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)


    # Make a prediction on the image
    pred = model.predict(gray)

    # Print the predicted labels and their corresponding probabilities
    all_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
                'Fibrosis', 'Pleural_Thickening', 'Hernia']
    for label, prob in zip(all_labels, pred[0]):
        print('{}: {:.2%}'.format(label, prob))

    
    resp = zip(all_labels,pred[0])

    return Response(resp)

# Create your views here.