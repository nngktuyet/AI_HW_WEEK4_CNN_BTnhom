import os
import cv2
import numpy as np
import tensorflow as tf


OUTPUT_DIR = "captured_lines"
MODEL_DIR = "/Users/nguyenngockimtuyet/AI_HW/palmistry_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CAM_WIDTH = 1280
CAM_HEIGHT = 720


IMG_SIZE = 200

MODEL_PATHS = {
    "life": os.path.join(MODEL_DIR, "/Users/nguyenngockimtuyet/AI_HW/palmistry_model/palmistry_life.h5"),
    "heart": os.path.join(MODEL_DIR, "/Users/nguyenngockimtuyet/AI_HW/palmistry_model/palmistry_heart.h5"),
    "head": os.path.join(MODEL_DIR, "/Users/nguyenngockimtuyet/AI_HW/palmistry_model/palmistry_head.h5"),
    "fate": os.path.join(MODEL_DIR, "/Users/nguyenngockimtuyet/AI_HW/palmistry_model/palmistry_fate.h5")
}

CLASS_NAMES = ["broken", "clear", "faint"]


def load_all_models():
    models = {}

    for line_type, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        models[line_type] = tf.keras.models.load_model(model_path)
        print(f"Đã load model: {line_type} - input shape: {models[line_type].input_shape}")

    return models

def resize_with_padding(img, size=200):
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return None

    scale = size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def draw_palm_guide(frame):
    h, w = frame.shape[:2]

    guide_w = int(w * 0.42)
    guide_h = int(h * 0.72)

    x1 = int((w - guide_w) / 2)
    y1 = int((h - guide_h) / 2)
    x2 = x1 + guide_w
    y2 = y1 + guide_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (guide_w // 3, guide_h // 3)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "Place palm inside green box - Press SPACE to predict",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q / ESC / X to quit",
        (40, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame, (x1, y1, x2, y2)



def crop_by_ratio(img, box):
    h, w = img.shape[:2]

    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return img[y1:y2, x1:x2]


def save_line_crops(palm_roi):
    """
    Crop 4 vùng đường chỉ tay từ vùng lòng bàn tay.
    Nếu crop lệch thì chỉnh lại thông số line_boxes.
    """

    line_boxes = {
        # x1, y1, x2, y2 theo tỷ lệ trong khung lòng bàn tay

        # Tâm đạo: phía trên lòng bàn tay
        "heart": (0.18, 0.20, 0.88, 0.42),

        # Trí đạo: giữa lòng bàn tay
        "head": (0.15, 0.36, 0.88, 0.60),

        # Sinh đạo: vùng quanh gò ngón cái
        "life": (0.05, 0.30, 0.55, 0.88),

        # Định mệnh đạo: dọc giữa lòng bàn tay
        "fate": (0.35, 0.28, 0.68, 0.92)
    }

    saved_paths = {}

    for line_name, box in line_boxes.items():
        crop = crop_by_ratio(palm_roi, box)

        if crop is None or crop.size == 0:
            print(f"Không crop được {line_name}")
            continue

        crop = resize_with_padding(crop, IMG_SIZE)

        if crop is None:
            print(f"Crop lỗi: {line_name}")
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{line_name}.jpg")
        cv2.imwrite(save_path, crop)

        saved_paths[line_name] = save_path
        print(f"Đã lưu: {save_path}")

    return saved_paths

def preprocess_for_cnn(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")

    img = resize_with_padding(img, IMG_SIZE)

    if img is None:
        raise ValueError(f"Ảnh lỗi: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_one_line(model, img_path):
    img = preprocess_for_cnn(img_path)

    pred = model.predict(img, verbose=0)[0]

    index = int(np.argmax(pred))
    feature = CLASS_NAMES[index]
    confidence = float(np.max(pred)) * 100

    return feature, round(confidence, 2)


def predict_all_lines(models, image_paths):
    results = {}

    for line_type in ["life", "heart", "head", "fate"]:
        if line_type not in image_paths:
            print(f"Thiếu ảnh crop: {line_type}")
            continue

        feature, confidence = predict_one_line(
            models[line_type],
            image_paths[line_type]
        )

        results[line_type] = {
            "feature": feature,
            "confidence": confidence
        }

    return results

def interpret_life(feature):
    if feature == "clear":
        return "Sinh đạo rõ: năng lượng ổn định, sức bền tốt, khả năng hồi phục khá tốt."
    elif feature == "faint":
        return "Sinh đạo mờ: năng lượng dễ dao động, cần chú ý nghỉ ngơi và cân bằng sức khỏe."
    elif feature == "broken":
        return "Sinh đạo đứt đoạn: cuộc sống có thể có nhiều giai đoạn thay đổi hoặc bước ngoặt lớn."
    return "Sinh đạo: chưa xác định rõ đặc điểm."


def interpret_heart(feature):
    if feature == "clear":
        return "Tâm đạo rõ: cảm xúc rõ ràng, chân thành, xu hướng tình cảm ổn định."
    elif feature == "faint":
        return "Tâm đạo mờ: cảm xúc kín đáo, khó bộc lộ hoặc chưa thật sự chắc chắn."
    elif feature == "broken":
        return "Tâm đạo đứt đoạn: tình cảm dễ biến động, nhạy cảm, cần học cách giao tiếp cảm xúc tốt hơn."
    return "Tâm đạo: chưa xác định rõ đặc điểm."


def interpret_head(feature):
    if feature == "clear":
        return "Trí đạo rõ: tư duy logic, khả năng tập trung và phân tích tốt."
    elif feature == "faint":
        return "Trí đạo mờ: dễ phân tán, cần rèn sự tập trung và tư duy hệ thống."
    elif feature == "broken":
        return "Trí đạo đứt đoạn: tư duy có nhiều thay đổi, linh hoạt nhưng dễ thiếu nhất quán."
    return "Trí đạo: chưa xác định rõ đặc điểm."


def interpret_fate(feature):
    if feature == "clear":
        return "Định mệnh đạo rõ: định hướng sự nghiệp tương đối rõ, có xu hướng kiên trì với mục tiêu."
    elif feature == "faint":
        return "Định mệnh đạo mờ: con đường sự nghiệp có thể chưa ổn định hoặc còn phụ thuộc môi trường."
    elif feature == "broken":
        return "Định mệnh đạo đứt đoạn: sự nghiệp có thể có nhiều bước ngoặt hoặc thay đổi hướng đi."
    return "Định mệnh đạo: chưa xác định rõ đặc điểm."


def generate_reading(results):
    if len(results) < 4:
        return "Chưa nhận đủ 4 đường chỉ tay. Hãy canh tay rõ hơn trong khung.", []

    life = results["life"]["feature"]
    heart = results["heart"]["feature"]
    head = results["head"]["feature"]
    fate = results["fate"]["feature"]

    features = [life, heart, head, fate]

    clear_count = features.count("clear")
    faint_count = features.count("faint")
    broken_count = features.count("broken")

    if clear_count >= 3:
        overall = "Tổng quan: các đường chính khá rõ, cho thấy xu hướng ổn định, có định hướng và khả năng duy trì mục tiêu tốt."
    elif broken_count >= 2:
        overall = "Tổng quan: nhiều đường có dấu hiệu đứt đoạn, cho thấy cuộc sống có thể có nhiều thay đổi hoặc các giai đoạn tái định hướng."
    elif faint_count >= 2:
        overall = "Tổng quan: nhiều đường còn mờ, cho thấy năng lượng hoặc định hướng chưa thật sự mạnh, cần rèn sự ổn định và tập trung."
    else:
        overall = "Tổng quan: đặc điểm khá cân bằng, vừa có điểm ổn định vừa có những vùng cần cải thiện."

    details = [
        interpret_life(life),
        interpret_heart(heart),
        interpret_head(head),
        interpret_fate(fate)
    ]

    return overall, details

def main():
    print("Đang load 4 model CNN...")
    models = load_all_models()
    print("Load model xong.")

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    window_name = "Palmistry Webcam System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Webcam đang chạy.")
    print("Đưa lòng bàn tay vào khung xanh.")
    print("Nhấn SPACE để chụp và dự đoán.")
    print("Nhấn Q hoặc ESC để thoát.")
    print("Hoặc bấm nút X trên cửa sổ webcam.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Không đọc được frame.")
            break

        frame = cv2.flip(frame, 1)

        display_frame, guide_box = draw_palm_guide(frame.copy())

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("Đang thoát webcam...")
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Cửa sổ webcam đã đóng.")
            break

        if key == 32:
            x1, y1, x2, y2 = guide_box

            palm_roi = frame[y1:y2, x1:x2]

            if palm_roi.size == 0:
                print("Không lấy được vùng lòng bàn tay.")
                continue

            palm_path = os.path.join(OUTPUT_DIR, "palm_roi.jpg")
            cv2.imwrite(palm_path, palm_roi)

            print("\nĐã chụp lòng bàn tay.")
            print(f"Đã lưu palm ROI: {palm_path}")

            image_paths = save_line_crops(palm_roi)

            print("\nĐang chạy dự đoán 4 CNN model...")
            results = predict_all_lines(models, image_paths)

            print("\n===== KẾT QUẢ DỰ ĐOÁN =====")
            for line, info in results.items():
                print(f"{line.upper()}: {info['feature']} - {info['confidence']}%")

            overall, details = generate_reading(results)

            print("\n===== DIỄN GIẢI PALMISTRY =====")
            print(overall)

            for detail in details:
                print("-", detail)

            print("\nBạn có thể canh tay lại và nhấn SPACE để dự đoán lần nữa.")
            print("Nhấn Q hoặc ESC để thoát.")

    cap.release()
    cv2.destroyAllWindows()

    for _ in range(5):
        cv2.waitKey(1)

    print("Đã thoát chương trình.")


if __name__ == "__main__":
    main()