from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import traceback
import os

app = Flask(__name__)

# --- Common helper function ---
def url_to_image(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        image_array = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# --- Function for /classify ---
def classify_food_image_basic(image_url):
    try:
        image = url_to_image(image_url)
        if image is None:
            return {"status": "Invalid", "message": "Unable to fetch image"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=100, param2=30, minRadius=50, maxRadius=350)

        if circles is None:
            return {"status": "False", "message": "No Plate Detected"}

        circles = np.uint16(np.around(circles))
        (x, y, r) = circles[0][0]
        plate_mask = np.zeros_like(gray)
        cv2.circle(plate_mask, (x, y), r, 255, -1)
        plate_area = cv2.bitwise_and(image, image, mask=plate_mask)
        hsv = cv2.cvtColor(plate_area, cv2.COLOR_BGR2HSV)

        lower_food = np.array([5, 50, 50])
        upper_food = np.array([35, 255, 255])
        food_mask = cv2.inRange(hsv, lower_food, upper_food)

        plate_pixel_count = np.sum(plate_mask > 0)
        food_pixel_count = np.sum(food_mask > 0)
        food_coverage = food_pixel_count / plate_pixel_count

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_coverage = np.sum(yellow_mask > 0) / plate_pixel_count

        if yellow_coverage > 0.5:
            return {"status": False}
        if food_coverage > 0.2:
            return {"status": True}
        else:
            return {"status": False}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

# --- Function for /compare ---
def classify_food_image(image_url, is_second_image=False):
    image = url_to_image(image_url)
    if image is None:
        return False
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                                   param1=100, param2=30, minRadius=50, maxRadius=400)

        if circles is None:
            return False

        circles = np.uint16(np.around(circles))
        (x, y, r) = circles[0][0]
        plate_mask = np.zeros_like(gray)
        cv2.circle(plate_mask, (x, y), r, 255, -1)
        plate_area = cv2.bitwise_and(image, image, mask=plate_mask)
        hsv = cv2.cvtColor(plate_area, cv2.COLOR_BGR2HSV)

        lower_food = np.array([5, 50, 50])
        upper_food = np.array([35, 255, 255])
        food_mask = cv2.inRange(hsv, lower_food, upper_food)

        plate_pixel_count = np.sum(plate_mask > 0)
        food_pixel_count = np.sum(food_mask > 0)
        food_coverage = food_pixel_count / plate_pixel_count if plate_pixel_count > 0 else 0

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_coverage = np.sum(yellow_mask > 0) / plate_pixel_count if plate_pixel_count > 0 else 0

        if is_second_image:
            return food_coverage < 0.1
        else:
            if yellow_coverage > 0.6:
                return False
            return food_coverage > 0.2

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# --- Routes ---
@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"status": "Error", "message": "Image URL is required"}), 400

    result = classify_food_image_basic(image_url)
    return jsonify(result)

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        data = request.get_json()
        if not data or "image_url1" not in data or "image_url2" not in data:
            return jsonify({"error": "Both image URLs are required."}), 400

        image_url1 = data["image_url1"]
        image_url2 = data["image_url2"]

        first_result = classify_food_image(image_url1)
        second_result = classify_food_image(image_url2, is_second_image=True)

        return jsonify({
            "final_result": first_result and second_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Start ---
if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)
