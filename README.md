# CNN Image Recognition Projects

## 1. Giới thiệu

Đây là project tổng hợp các bài toán nhận dạng hình ảnh sử dụng mô hình CNN. Project gồm 4 bài chính:

1. **Banknote Recognition** — nhận diện mệnh giá tiền.
2. **Food Recognition** — nhận diện món ăn.
3. **Flower Recognition** — nhận diện loài hoa.
4. **Palmistry Line Analysis** — nhận diện và phân tích đường chỉ tay bằng webcam.

Mục tiêu của project là áp dụng Convolutional Neural Network vào các bài toán phân loại ảnh khác nhau, từ nhận dạng vật thể thông thường đến nhận diện đặc trưng hình ảnh phức tạp hơn như đường chỉ tay.

---

## 2. Danh sách file nộp

Các file chính trong project gồm:

```text
webcam_palmistry_full.py
life_train.ipynb
heart_train.ipynb
head_train.ipynb
fate_train.ipynb
Food recognition.ipynb
flowers_train.ipynb
AI_HW_vnbanknotes.ipynb
```

Trong đó:

* `webcam_palmistry_full.py`: file chính để chạy hệ thống Palmistry bằng webcam.
* `life_train.ipynb`: notebook train model cho đường sinh đạo.
* `heart_train.ipynb`: notebook train model cho đường tâm đạo.
* `head_train.ipynb`: notebook train model cho đường trí đạo.
* `fate_train.ipynb`: notebook train model cho đường định mệnh.
* `Food recognition.ipynb`: notebook train model nhận diện thức ăn.
* `flowers_train.ipynb`: notebook train model nhận diện hoa.
* `AI_HW_vnbanknotes.ipynb`: notebook train model nhận diện tiền giấy.

---

## 3. Cài đặt môi trường

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

# PHẦN 1: BANKNOTE RECOGNITION

## 4. Mô tả bài nhận diện tiền

Bài **Banknote Recognition** sử dụng CNN để nhận diện mệnh giá tiền từ ảnh đầu vào.

Ảnh tiền được chia thành các class theo từng mệnh giá, ví dụ:

```text
10000
20000
50000
100000
200000
500000
```

Mục tiêu là khi đưa vào một ảnh tiền, model có thể dự đoán ảnh đó thuộc mệnh giá nào.

---

## 5. Quy trình xử lý bài nhận diện tiền

Pipeline cơ bản:

```text
Ảnh tiền
→ Resize ảnh
→ Chuẩn hóa pixel
→ Đưa vào CNN
→ Dự đoán mệnh giá
```

Nếu dataset có cả mặt trước và mặt sau của tờ tiền, nên đưa cả hai mặt vào train để model nhận diện tốt hơn trong thực tế.

---

## 6. Output bài nhận diện tiền

Ví dụ output:

```text
Predicted class: 500000
Confidence: 94.52%
```

---

# PHẦN 2: FOOD RECOGNITION

## 7. Mô tả bài nhận diện thức ăn

File:

```text
Food recognition.ipynb
```

dùng để train mô hình CNN nhận diện các loại thức ăn từ ảnh.

Dataset được chia thành các folder class, mỗi folder tương ứng với một loại món ăn.
```text
Train/
├── Banh khot/
├── Banh mi/
├── Bun bo Hue/
├── Bun dau mam tom/
├── Bun rieu/
├── Bun thit nuong/
├── Com tam/
├── Goi cuon/
├── Hu tieu/
└── Pho/
```

---

## 8. Quy trình xử lý bài nhận diện thức ăn

```text
Ảnh món ăn
→ Resize ảnh
→ Data augmentation
→ Train CNN
→ Đánh giá accuracy/loss
→ Dự đoán món ăn mới
```

Model học các đặc trưng hình ảnh như màu sắc, hình dạng, texture và bố cục của món ăn để phân loại.

---

## 9. Output bài nhận diện thức ăn

Ví dụ:

```text
Predicted food: pizza
Confidence: 88.40%
```

---

# PHẦN 3: FLOWER RECOGNITION

## 10. Mô tả bài nhận diện hoa

File:

```text
flowers_train.ipynb
```

dùng để train mô hình CNN nhận diện các loài hoa.

Dataset được chia thành các class theo tên loài hoa.

Ví dụ:

```text
flowers_data/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

---

## 11. Quy trình xử lý bài nhận diện hoa

```text
Ảnh hoa
→ Resize ảnh
→ Chuẩn hóa dữ liệu
→ Train CNN
→ Đánh giá model
→ Dự đoán loài hoa
```

Model học các đặc trưng như màu cánh hoa, hình dạng cánh hoa, bố cục bông hoa và nền ảnh.

---

## 12. Output bài nhận diện hoa

Ví dụ:

```text
Predicted flower: sunflower
Confidence: 91.25%
```

---

# PHẦN 4: PALMISTRY LINE ANALYSIS

## 13. Mô tả bài Palmistry

Bài **Palmistry Line Analysis** sử dụng webcam để chụp lòng bàn tay, sau đó crop 4 vùng đường chỉ tay chính:

* `life` — sinh đạo
* `heart` — tâm đạo
* `head` — trí đạo
* `fate` — định mệnh đạo

Mỗi đường chỉ tay được đưa vào một model CNN riêng để phân loại đặc tính:

```text
broken
clear
faint
```

Sau đó hệ thống tổng hợp kết quả từ 4 model và tạo phần diễn giải palmistry.

---

## 14. Các file train model Palmistry

Bốn notebook sau chỉ dùng để **train model**, không phải file chạy chính:

```text
life_train.ipynb
heart_train.ipynb
head_train.ipynb
fate_train.ipynb
```

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

## 15. File chạy chính của Palmistry

Để chạy bài Palmistry, chỉ cần chạy file:

```text
webcam_palmistry_full.py
```

Không cần chạy lại 4 notebook train nếu đã có sẵn 4 model `.h5`.

---

## 16. Cách chạy bài Palmistry

Mở Terminal:

```bash
cd /Users/nguyenngockimtuyet/AI_HW
python3 webcam_palmistry_full.py
```

Sau khi chạy:

```text
1. Webcam sẽ mở lên.
2. Đưa lòng bàn tay vào khung xanh.
3. Canh lòng bàn tay nằm trong hình oval.
4. Nhấn SPACE để chụp và dự đoán.
5. Xem kết quả trong Terminal.
6. Nhấn Q, ESC hoặc nút X để thoát.
```

Không nên chạy file webcam trong Jupyter Notebook vì OpenCV có thể làm crash kernel.

---

## 17. Quy trình hoạt động của Palmistry

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

## 18. Output bài Palmistry

Ví dụ:

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

## 19. Kết luận

Project đã triển khai nhiều ứng dụng CNN cho nhận dạng hình ảnh:

```text
Banknote Recognition
Food Recognition
Flower Recognition
Palmistry Line Analysis
```

Các bài toán đều sử dụng quy trình chung:

```text
Image Input
→ Preprocessing
→ CNN Model
→ Prediction
→ Result Output
```

Thông qua project này, người học có thể hiểu cách xây dựng mô hình CNN, cách xử lý dữ liệu ảnh, cách train model và cách tích hợp model vào một ứng dụng webcam thực tế.
