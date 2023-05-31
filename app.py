from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from load_model import *
import base64
import cv2
import numpy as np
import torch
import os
import face_recognition

font = cv2.FONT_HERSHEY_DUPLEX        
text_color = (0, 0, 255)

model, feature_extractor = ModelLoader('pre_model/fined_tuning')
emotions = {
    0:"Angry",
    1:"Disgust",
    2:"Fear",
    3:"Happy",
    4:"Sad",
    5:"Surprise",
    6:"Neutral"
    }

app = FastAPI()
templates = Jinja2Templates(directory=".")
@app.get("/")
def form_post(request: Request):
    return RedirectResponse(url="/dynamic")

@app.get("/dynamic")
def form_post(request: Request):
    return templates.TemplateResponse('web.html', context={'request': request})

@app.post("/dynamic")
async def dynamic(request: Request, photo: UploadFile = File()):
    img = photo.file.read()
    # encoding the image
    nparr = np.frombuffer(img, np.uint8)
    color_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # width, height, _ = color_img.shape
    gray_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
    
    face_locations = face_recognition.face_locations(img)
    print('So luong khuon mat:', len(face_locations))
    for locatation in face_locations:
        top, right, bottom, left = locatation
        cv2.rectangle(color_img, (left, top), (right, bottom), (0, 255, 0), 4)
        face = img[top:bottom, left:right]
        
        #predict
        inputs = feature_extractor(face, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            softmax_vals = torch.softmax(logits, dim=-1)
        predicted_label = logits.argmax(-1).item()
        confidence = softmax_vals[0][predicted_label]

        result =  emotions[predicted_label]
        print(result)
        
        #put labels
        (w, h), _ = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # font_scale = min(width / w, height/ h) 
        img = cv2.rectangle(color_img, (left, bottom), (left + w, bottom + 3*h), (0, 255, 0), -1)
        img = cv2.putText(color_img, result, (left, bottom + h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        img = cv2.putText(color_img, str(round(float(confidence),2)), (left, bottom + 2*h + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)



    buffer = cv2.imencode('.jpg', color_img)[1]

    i = 1  # số thứ tự bắt đầu
    filename = photo.filename
    name = filename.split('.')[0]
    extension = filename.split('.')[1]
    while os.path.exists(os.path.join('result', filename)):
        filename = name + f" ({i})." + extension # đặt tên mới cho tệp
        i += 1
        
    with open("result/" + filename, "wb") as f:
        f.write(buffer)
    photo.file.close()

    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return templates.TemplateResponse("web.html", {"request": request, "photo": encoded_image})






    
