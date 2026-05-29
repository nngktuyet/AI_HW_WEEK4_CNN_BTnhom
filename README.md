# Palmistry Line Analysis

## 1. Giới thiệu

Đây là project nhận diện và phân tích đường chỉ tay bằng webcam, sử dụng mô hình **Convolutional Neural Network (CNN)**.

Hệ thống sẽ chụp lòng bàn tay qua webcam, crop 4 vùng đường chỉ tay chính, đưa từng vùng vào model CNN riêng và trả về kết quả phân loại.

---

## 2. File chính

```text
webcam_palmistry_full.py
```

Đây là file dùng để chạy hệ thống Palmistry bằng webcam.

---

## 3. Các file train model

Bốn notebook sau chỉ dùng để **train model**, không phải file chạy chính:

```text
life_train.ipynb
heart_train.ipynb
head_train.ipynb
fate_train.ipynb
```

Ý nghĩa từng file:

* `life_train.ipynb`: train model cho đường sinh đạo.
* `heart_train.ipynb`: train model cho đường tâm đạo.
* `head_train.ipynb`: train model cho đường trí đạo.
* `fate_train.ipynb`: train model cho đường định mệnh.

Sau khi train xong, mỗi notebook tạo ra một model `.h5`:

```text
palmistry_life.h5
palmistry_heart.h5
palmistry_head.h5
palmistry_fate.h5
```

Các model này cần được đặt trong folder:

```text
palmistry_model/
```

---

## 4. Cài đặt môi trường

Mở Terminal trong thư mục project:

```bash
cd /Users/nguyenngockimtuyet/AI_HW
```

Cài các thư viện cần thiết:

```bash
python3 -m pip install tensorflow keras opencv-python numpy matplotlib
```

Nếu dùng Windows:

```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

---

## 5. Các class dự đoán

Mỗi đường chỉ tay được phân loại thành 3 nhóm:

```text
broken
clear
faint
```

Ý nghĩa cơ bản:

* `broken`: đường bị đứt đoạn.
* `clear`: đường rõ nét.
* `faint`: đường mờ.

---

## 6. Cách chạy

Nếu đã có sẵn 4 model `.h5`, chỉ cần chạy file:

```bash
python3 webcam_palmistry_full.py
```

Không cần chạy lại 4 notebook train model.

Lưu ý: không nên chạy file webcam trong Jupyter Notebook vì OpenCV có thể làm crash kernel.

---

## 7. Hướng dẫn sử dụng

Sau khi chạy file:

```text
1. Webcam sẽ mở lên.
2. Đưa lòng bàn tay vào khung xanh.
3. Canh lòng bàn tay nằm trong hình oval.
4. Nhấn SPACE để chụp và dự đoán.
5. Xem kết quả trong Terminal.
6. Nhấn Q, ESC hoặc nút X để thoát.
```

---

## 8. Quy trình hoạt động

```text
Webcam
→ Hiển thị khung canh lòng bàn tay
→ Người dùng nhấn SPACE
→ Crop vùng lòng bàn tay
→ Crop 4 vùng đường chỉ tay
→ Resize ảnh về 200x200
→ Đưa từng ảnh vào CNN tương ứng
→ Dự đoán broken / clear / faint
→ Tổng hợp và diễn giải kết quả
```

---

## 9. Output mẫu

```text
===== KẾT QUẢ DỰ ĐOÁN =====
LIFE: broken - 91.03%
HEART: clear - 99.95%
HEAD: clear - 99.99%
FATE: faint - 99.62%

===== DIỄN GIẢI PALMISTRY =====
Tổng quan: đặc điểm khá cân bằng, vừa có điểm ổn định vừa có những vùng cần cải thiện.

- Sinh đạo đứt đoạn: cuộc sống có thể có nhiều giai đoạn thay đổi hoặc bước ngoặt lớn.
- Tâm đạo rõ: cảm xúc rõ ràng, chân thành, xu hướng tình cảm ổn định.
- Trí đạo rõ: tư duy logic, khả năng tập trung và phân tích tốt.
- Định mệnh đạo mờ: con đường sự nghiệp có thể chưa ổn định hoặc còn phụ thuộc môi trường.
```

---

## 10. Kết luận

Project Palmistry Line Analysis cho thấy cách tích hợp CNN vào một ứng dụng webcam thực tế. Bài này không chỉ train model mà còn kết hợp OpenCV để lấy ảnh từ webcam, crop vùng ảnh cần phân tích và trả kết quả dự đoán trực tiếp cho người dùng.
