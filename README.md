# **Skin Disease Classifier**  

This project is a Flask-based web application that uses a **Convolutional Neural Network (CNN)** to classify five types of skin diseases. The model is trained on the **HAM10000 dataset** and provides an interactive **web-based UI (index.html)** for users to upload and classify skin images.

---

## **Project Overview**  

The system detects the following skin conditions:  

- **Actinic Keratoses (akiec)**  
- **Basal Cell Carcinoma (bcc)**  
- **Benign Keratosis-like Lesions (bkl)**  
- **Dermatofibroma (df)**  
- **Melanoma (mel)**  

The dataset is preprocessed, and images are resized and normalized before training. A **Flask API** is used to handle image uploads, run model inference, and display predictions through a **web interface**.

---

## **Features**  

- **User-friendly Web Interface** (index.html for uploading images)  
- **Flask-based API** for handling model inference  
- **CNN Model for Skin Disease Classification**  
- **SQLite Database** for storing predictions  
- **Image Preprocessing** using Torchvision transforms  
- **Cloud-Deployment Ready**  

---

## **Dataset & Preprocessing**  

- **Dataset**: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
- **Preprocessing Steps**:  
  - Resize images to **128x128 pixels**  
  - Normalize using `torchvision.transforms`  
  - Augment images (rotation, flipping)  
- **Train-Test Split**: **80% training, 20% validation**  

---

## **Installation Guide**  

### **Set Up Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # (Mac/Linux)
venv\Scripts\activate  # (Windows)
```

### **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **Download the Trained Model**  
If the model is not included, download it using:  
```bash
gdown --id YOUR_MODEL_FILE_ID -O cnn_skin_model.pth
```

### **Run the Flask Web Application**  
```bash
python app.py
```
The application will be accessible at:  
**`http://127.0.0.1:5000/`**  

---

## **Using the Web Application**  

### **1. Upload an Image via Web UI (index.html)**  
- Open `http://127.0.0.1:5000/` in a browser  
- Click **Choose File** to upload an image  
- Click **Predict** to classify the skin condition  
- The **predicted disease and confidence score** will be displayed  

### **2. Use the API with Postman or CURL**  

#### **Make a POST request to `/predict`**  
```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/predict
```

#### **Example JSON Response**  
```json
{
  "disease": "Melanoma",
  "confidence": "92.50%"
}
```

### **3. View Prediction History**  
```bash
curl http://127.0.0.1:5000/history
```

---

## **Model Training**  

The Convolutional Neural Network (CNN) architecture consists of:  
- **Two convolutional layers** with ReLU activation  
- **MaxPooling layers** for downsampling  
- **Fully connected layers** for classification  

Training results:

| Epoch | Loss  | Accuracy  |
|--------|------|-----------|
| 1      | 41.31 | 34.50% |
| 5      | 15.33 | 77.75% |
| 10     | 4.39  | 94.50% |

The CNN model achieved **94.50% accuracy** on the validation set.



---

## **Contributors**  

- Developer: **Dev Thonangi**  
- Dataset Source: **HAM10000**  
- Frameworks Used: **PyTorch, Torchvision, Flask, HTML**  

---

## **License**  

This project is licensed under the **MIT License**.  

---

## **Next Steps**  

- Improve accuracy with additional data augmentation  
- Deploy the model on cloud services  
- Enhance the **web UI** for a better user experience  
