from flask import Flask, request, jsonify, render_template
import cv2
import joblib
import numpy as np
import os
import time

from skimage.feature import hog, local_binary_pattern
from skimage import morphology


import threading


from sorter_training_ml import retrain_svm

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# shared state
chosen_number = 0
waiting = False
status = "starting"

model = "model_1"
model_classes = 3

retrain_svm(model, model_classes)

svm_clf = joblib.load(f"models/{model}_svm_model.joblib")
pca = joblib.load(f"models/{model}_pca_transform.joblib")

mask_processed = False


def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))

    hog_feat = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2),
                   orientations=9, block_norm='L2-Hys', feature_vector=True)

    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 8 + 3),
                             range=(0, 8 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return np.hstack([hog_feat, hist])




def crop_mask():
    global mask_processed

    img = cv2.imread('uploads/initial_image.jpg')
    # hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 20, 45])
    upper_green = np.array([125, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    cv2.imwrite("uploads/surface_mask_raw.jpg", mask)

    # kernel = np.ones((15, 15), np.uint8)

    # morphex_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    selem = np.ones((15, 15), np.uint8)
    morphex_mask = morphology.binary_closing(mask, selem)

    cv2.imwrite("uploads/surface_mask.jpg", morphex_mask.astype(np.uint8) * 255)
    mask_processed = True



def detail_measure(image, ksize=3):

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return laplacian.var()

def blockwise_detail(image, block_size=64):
    h, w = image.shape[:2]
    details = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            score = detail_measure(block)
            details.append(((x,y), score))
    return details

def process_image():
    global chosen_number, waiting, status, mask_processed
    
    while not mask_processed:
        time.sleep(0.1)
    
    correction = True

    img_init = cv2.imread("uploads/initial_image.jpg")
    img = cv2.imread("uploads/received_image.jpg")

    mask = cv2.imread("uploads/surface_mask.jpg", cv2.IMREAD_GRAYSCALE)

    if correction:
        img = img.astype(np.float32)
        img_init = img_init.astype(np.float32)
        inverted_mask = cv2.bitwise_not(mask)

        for c in range(3):
            init_mean, _ = cv2.meanStdDev(img_init[:, :, c], mask=mask)
            img_mean, _ = cv2.meanStdDev(img[:, :, c], mask=mask)
            diff = init_mean[0][0] - img_mean[0][0]
            img[:, :, c] += diff
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_init = img_init.astype(np.uint8)

        cv2.imwrite("uploads/received_image_correction.jpg", img)



    diff = cv2.absdiff(img_init, img)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    diff[:, :, 1] = (diff[:, :, 1] * 0.2).astype(np.uint8)

    # mean = cv2.mean(diff, mask=mask)
    # print(f"Mean difference: {mean}")

    # meanmax = max(mean)

    cv2.imwrite("uploads/difference_image.jpg", diff)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)


    mean, std = cv2.meanStdDev(diff, mask=mask)

    print(f"Mean: {mean}, Std: {std}")
    mean = mean[0][0]
    std = std[0][0]

    if std < 5:
        print("No object detected.")
        chosen_number = 0
        waiting = False
        status = "idle"
        return

    obj_thresh = cv2.threshold(diff, mean+0.05*(255-mean), 255, cv2.THRESH_BINARY)[1]
    # obj_thresh = cv2.threshold(diff, mean, 255, cv2.THRESH_BINARY)[1]
    # obj_thresh = cv2.cvtColor(obj_thresh, cv2.COLOR_BGR2GRAY)
    obj_thresh[obj_thresh > 0] = 255

    # morphex obj
    # kernel = np.ones((7, 7), np.uint8)
    # obj_thresh = cv2.morphologyEx(obj_thresh, cv2.MORPH_CLOSE, kernel)
    selem = np.ones((7, 7), np.uint8)
    obj_thresh = morphology.binary_closing(obj_thresh, selem)

    cv2.imwrite("uploads/received_image_masked.jpg", obj_thresh)


    # make square that contains biggest area in obj_thresh
    contours, _ = cv2.findContours(obj_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No object contour detected.")
        chosen_number = 0
        waiting = False
        status = "idle"
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    img_masked = img[y:y+h, x:x+w]

    cv2.imwrite("uploads/received_image_processed.jpg", img_masked)

    print("Object detected.")

    # increase contrast, grayscale, resize to new size
    img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    img_masked = cv2.equalizeHist(img_masked)
    img_masked = cv2.resize(img_masked, (128, 128))

    cv2.imwrite("uploads/model_input.jpg", img_masked)


    feat = extract_features("uploads/model_input.jpg")
    feat_reduced = pca.transform(feat.reshape(1, -1))

    scores = svm_clf.decision_function(feat_reduced)
    print("Decision scores:", scores)

    probs = svm_clf.predict_proba(feat_reduced)
    print("Class probabilities:", probs[0])

    max_prob = np.max(probs)
    chosen_number = int(np.argmax(probs) + 1)

    if max_prob > 0.75:
        print(f"Confidently predicted class: {chosen_number} with probability: {max_prob}")
        dest_folder = os.path.join(model, str(chosen_number))
        os.makedirs(dest_folder, exist_ok=True)
        os.rename("uploads/model_input.jpg", os.path.join(dest_folder, f"{int(time.time())}.jpg"))
        status = "idle"
    
    else:
        print(f"Unconfident prediction: {chosen_number} with probability: {max_prob}")
        status = "user_input"


@app.route('/init', methods=['POST'])
def initialize():
    if not request.data:
        return jsonify({"error": "No data received"}), 400

    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 20, 30, 7, 21)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "initial_image.jpg"), img)

    threading.Thread(target=crop_mask).start()
    mask_processed = False
    status = "idle"

    return jsonify({"message": "Image received and processed successfully!"})


@app.route('/upload', methods=['POST'])
def upload_image():
    global chosen_number, waiting, status

    if not request.data:
        return jsonify({"error": "No data received"}), 400

    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 20, 30, 7, 21)

    cv2.imwrite("uploads/received_image.jpg", img)

    threading.Thread(target=process_image).start()

    chosen_number = None
    waiting = True
    status = "processing"

    return jsonify({"valid": True})


@app.route('/pick')
def pick_number():
    """Webpage for the user to pick a number based on the uploaded image."""


    return render_template("pick.html", image_url="/uploads/received_image_processed.jpg")


@app.route('/uploads/<filename>')
def serve_file(filename):
    return open(os.path.join(UPLOAD_FOLDER, filename), "rb").read()


@app.route('/answer', methods=['POST'])
def answer():
    """Receive the number picked by the user from the webpage."""
    print("Received answer")
    global chosen_number, waiting, status
    number = request.form.get("number")
    if number is None:
        return "No number provided.", 400

    chosen_number = int(number)
    status = "idle"
    waiting = False

    if chosen_number > 0 and chosen_number <= model_classes:

        dest_folder = os.path.join(model, str(chosen_number))
        os.makedirs(dest_folder, exist_ok=True)
        os.rename("uploads/model_input.jpg", os.path.join(dest_folder, f"{int(time.time())}.jpg"))

    return f"Thanks! You picked {chosen_number}"


@app.route('/log', methods=['POST'])
def receive_status():
    data = request.get_json()
    if not data or 'status' not in data:
        return jsonify({"error": "No status provided"}), 400
    status = data['status']
    print(f"\033[92mReceived status update: {status}\033[0m")
    return jsonify({"message": "Status received", "status": status})


@app.route('/result', methods=['GET'])
def get_result():
    if status == "processing" or status == "user_input":
        print("Image is still being processed...")
        return jsonify({"status": status})
    print(f"Returning chosen number: {chosen_number}")
    return jsonify({"result": chosen_number, "status": "ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
