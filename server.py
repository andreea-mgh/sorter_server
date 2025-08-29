from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import time

import threading

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Shared state (simple in-memory store for demo)
chosen_number = None
waiting = False
status = "starting"

model = "model_1"

mask_processed = False

def crop_mask():
    global mask_processed

    img = cv2.imread('uploads/initial_image.jpg')
    # hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 20, 45])
    upper_green = np.array([125, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((15, 15), np.uint8)

    # Perform morphological closing
    morphex_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # row_sums = np.sum(mask, axis=1) / np.shape(mask)[1]
    # col_sums = np.sum(mask, axis=0) / np.shape(mask)[0]

    # thresh = 80
    # top_edge = np.argmax(row_sums > thresh)
    # bottom_edge = len(row_sums) - np.argmax(row_sums[::-1] > thresh) - 1
    # left_edge = np.argmax(col_sums > thresh)
    # right_edge = len(col_sums) - np.argmax(col_sums[::-1] > thresh) - 1



    # mask[:top_edge, :] = 0
    # mask[bottom_edge:, :] = 0
    # mask[:, :left_edge] = 0
    # mask[:, right_edge:] = 0

    cv2.imwrite("uploads/surface_mask.jpg", morphex_mask)
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
    global chosen_number, waiting, status

    correction = True

    img_init = cv2.imread("uploads/initial_image.jpg")
    img = cv2.imread("uploads/received_image.jpg")

    mask = cv2.imread("uploads/surface_mask.jpg", cv2.IMREAD_GRAYSCALE)


    # Color correction

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

    # mai putin verde
    diff[:, :, 1] = (diff[:, :, 1] * 0.5).astype(np.uint8)

    # mean = cv2.mean(diff, mask=mask)
    # print(f"Mean difference: {mean}")

    # meanmax = max(mean)

    cv2.imwrite("uploads/difference_image.jpg", diff)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)


    mean, std = cv2.meanStdDev(diff, mask=mask)

    print(f"Mean: {mean}, Std: {std}")
    mean = mean[0][0]
    std = std[0][0]

    if std < 8:
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
    kernel = np.ones((7, 7), np.uint8)
    obj_thresh = cv2.morphologyEx(obj_thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("uploads/received_image_masked.jpg", obj_thresh)


    # draw obj_thresh over image
    # img_masked = cv2.bitwise_and(img, img, mask=obj_thresh)


    # make square that contains biggest area in obj_thresh
    contours, _ = cv2.findContours(obj_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        img_masked = img[y:y+h, x:x+w]

        cv2.imwrite("uploads/received_image_processed.jpg", img_masked)
    else:
        print("No object contour detected.")

    # edges = cv2.Canny(diff, 40, 80)
    # # expand
    # kernel = np.ones((5, 5), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)
    # cv2.imwrite("uploads/received_image_edges.jpg", edges)

    # # density map of edges
    # density = detail_measure(edges)
    # # draw map
    # print(f"Edge density: {density}")

    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     print("No object detected.")
    #     chosen_number = 0
    #     waiting = False
    #     status = "idle"
    #     return
    
    # largest_contour = max(contours, key=cv2.contourArea)
    # area = cv2.contourArea(largest_contour)

    # if area < 3000:
    #     print(f"Object too small. Area: {area}")
    #     chosen_number = 0
    #     waiting = False
    #     status = "idle"
    #     return
    
    # print(f"Object detected. Area: {area}")
    
    # cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
    # cv2.imwrite("uploads/received_image_contours.jpg", img)
    

    # if meanmax < 50:
    #     print("No object detected.")
    #     chosen_number = 0
    #     waiting = False
    #     status = "idle"
    #     return

    print("Object detected.")

    # find center of mass
    # M = cv2.moments(obj_thresh)
    # if M["m00"] == 0:
    #     cx, cy = 0, 0
    # else:
    #     cx = int(M["m10"] / M["m00"])
    #     cy = int(M["m01"] / M["m00"])
    # print(f"Center of mass: ({cx}, {cy})")
    # # save received_image with labeled center
    # cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    # cv2.imwrite("uploads/received_image_labeled.jpg", img)


    # contours
    # diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # contours, _ = cv2.findContours(diff_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if contours:
    #     # Find the largest contour
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     # Draw the largest contour on the image
    #     cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
    #     cv2.imwrite("uploads/received_image_contours.jpg", img)

    # blockwise detail
    # block_details = blockwise_detail(diff)
    # for (x, y), score in block_details:
    #     print(f"Block at ({x}, {y}) has detail score: {score}")

    # # make image with block details
    # detail_image = np.zeros_like(diff)
    # for (x, y), score in block_details:
    #     cv2.putText(detail_image, f"{score:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # cv2.imwrite("uploads/received_image_details.jpg", detail_image)

@app.route('/init', methods=['POST'])
def initialize():
    if not request.data:
        return jsonify({"error": "No data received"}), 400

    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 20, 30, 7, 21)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "initial_image.jpg"), img)

    threading.Thread(target=crop_mask).start()

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

    while not mask_processed:
        time.sleep(0.1)

    threading.Thread(target=process_image).start()

    chosen_number = None
    waiting = True
    status = "processing"

    return jsonify({"valid": True})


@app.route('/pick')
def pick_number():
    """Webpage for the user to pick a number based on the uploaded image."""


    return render_template("pick.html", waiting=waiting, image_url="/uploads/received_image_processed.jpg", mask="/uploads/received_image_masked.jpg")


@app.route('/uploads/<filename>')
def serve_file(filename):
    return open(os.path.join(UPLOAD_FOLDER, filename), "rb").read()


@app.route('/answer', methods=['POST'])
def answer():
    """Receive the number picked by the user from the webpage."""
    print("Received answer")
    global chosen_number, waiting
    number = request.form.get("number")
    if number is None:
        return "No number provided.", 400

    chosen_number = int(number)
    waiting = False

    # save image in data/
    dest_folder = os.path.join(model, str(chosen_number))
    os.makedirs(dest_folder, exist_ok=True)
    if chosen_number:
        os.rename("uploads/received_image.jpg", os.path.join(dest_folder, f"{int(time.time())}.jpg"))

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
    if waiting:
        print("Image is still being processed...")
        return jsonify({"status": "waiting"})
    print(f"Returning chosen number: {chosen_number}")
    return jsonify({"result": chosen_number, "status": "ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
