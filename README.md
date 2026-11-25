# 🇻🇳 Vietnamese Word Segmentation: From Baseline to LSTM

**Seminar Project – Natural Language Processing**

Ứng dụng và so sánh các phương pháp Baseline, Hybrid (Maximum Matching + Bigram Language Model) và LSTM trong bài toán tách từ tiếng Việt (Vietnamese Word Segmentation).

---

## 1. Project Overview

### 1.1. Introduction

Word Segmentation (tách từ) là một bước tiền xử lý quan trọng trong các hệ thống Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing – NLP).

Khác với nhiều ngôn ngữ sử dụng khoảng trắng để phân tách từ, tiếng Việt sử dụng khoảng trắng chủ yếu để phân tách âm tiết (syllable). Một từ tiếng Việt có thể bao gồm một hoặc nhiều âm tiết.

Ví dụ:

- `sinh viên` là một từ gồm hai âm tiết.
- `xử lý` là một từ gồm hai âm tiết.
- `ngôn ngữ tự nhiên` có thể được phân tách thành nhiều đơn vị từ vựng theo ngữ cảnh và quy ước gán nhãn.

Do đó, việc xác định chính xác ranh giới giữa các từ là một bài toán quan trọng và không thể giải quyết đầy đủ chỉ bằng cách tách văn bản theo khoảng trắng.

### 1.2. Research Question

Dự án tập trung trả lời câu hỏi:

**Liệu mô hình LSTM có thể cải thiện hiệu quả tách từ tiếng Việt so với các phương pháp dựa trên luật và từ điển hay không?**

Nhóm sẽ đánh giá hiệu quả của ba hướng tiếp cận:

1. Baseline: Whitespace, Regex và Rule-based Processing.
2. Hybrid: Maximum Matching kết hợp Bigram Language Model.
3. LSTM: Sequence Labeling sử dụng mạng Long Short-Term Memory.

### 1.3. Project Objectives

- Tìm hiểu cơ sở lý thuyết của Vietnamese Word Segmentation.
- Nghiên cứu các phương pháp tách từ truyền thống và phương pháp học sâu.
- Xây dựng pipeline tiền xử lý và biểu diễn dữ liệu.
- Triển khai ba phương pháp trên cùng một bộ dữ liệu.
- Đánh giá và so sánh hiệu quả bằng các chỉ số định lượng.
- Phân tích các trường hợp dự đoán đúng và sai.
- Xây dựng ứng dụng demo trực quan cho phép so sánh kết quả của ba phương pháp.

---

## 2. Methodology

### 2.1. Baseline – Whitespace & Regex

Phương pháp đối chứng đơn giản sử dụng khoảng trắng để tách âm tiết, kết hợp Regular Expressions và các quy tắc xử lý văn bản.

**Nguyên lý:**
- Tách văn bản dựa trên khoảng trắng.
- Nhận diện dấu câu, số, URL, ngày tháng và các mẫu đặc biệt.
- Chuẩn hóa đầu vào trước khi xử lý.

**Ưu điểm:**
- Đơn giản, tốc độ xử lý nhanh.
- Không cần dữ liệu huấn luyện.
- Dễ triển khai và kiểm thử.

**Hạn chế:**
- Không nhận diện đầy đủ từ ghép nhiều âm tiết.
- Không giải quyết được nhập nhằng ranh giới từ.
- Không khai thác được ngữ cảnh ngôn ngữ.

### 2.2. Hybrid – Maximum Matching + Bigram Language Model

Phương pháp kết hợp tri thức từ vựng với mô hình ngôn ngữ thống kê.

**Các thành phần chính:**
1. Dictionary: Từ điển chứa các đơn vị từ vựng tiếng Việt.
2. Maximum Matching: Tạo các phương án phân tách từ ứng viên.
3. Bigram Language Model: Ước lượng xác suất các chuỗi từ.
4. Ambiguity Resolution: Lựa chọn phương án phân tách phù hợp nhất.

**Nguyên lý:**

- Tạo từ điển từ dữ liệu huấn luyện.
- Sử dụng Maximum Matching để tìm các phương án ứng viên.
- Tính xác suất các phương án bằng Bigram.
- Áp dụng kỹ thuật làm trơn để xử lý các Bigram chưa xuất hiện.
- Lựa chọn phương án có điểm xác suất tốt nhất.

**Tài liệu nghiên cứu chính:**

Phuong, L. H., Huyen, N. T. M., Roussanaly, A., & Vinh, H. T. (2008). *A Hybrid Approach to Word Segmentation of Vietnamese Texts.*

DOI: https://doi.org/10.1007/978-3-540-88282-4_23

Lưu ý: Phiên bản triển khai của dự án sẽ ghi rõ những thành phần được tái hiện và những thành phần được giản lược so với kiến trúc trong nghiên cứu gốc.

### 2.3. LSTM – Sequence Labeling

Sử dụng mạng Long Short-Term Memory để giải quyết bài toán Word Segmentation dưới dạng gán nhãn chuỗi.

**Kiến trúc dự kiến:**

Input Syllables → Embedding → LSTM → Linear → Softmax → BIO Labels → Word Reconstruction.

**Quy trình:**

1. Tách văn bản thành chuỗi âm tiết.
2. Chuyển đổi âm tiết thành các vector biểu diễn.
3. Đưa chuỗi vector qua LSTM để học thông tin ngữ cảnh.
4. Dự đoán nhãn tại từng vị trí.
5. Khôi phục ranh giới từ dựa trên chuỗi nhãn BIO.

**BIO Scheme:**
- B (Beginning): Bắt đầu một từ mới.
- I (Inside): Âm tiết tiếp tục thuộc từ hiện tại.

Dự án ưu tiên xây dựng LSTM cơ bản. BiLSTM hoặc các kiến trúc mở rộng có thể được xem xét nếu thời gian và tài nguyên cho phép.

---

## 3. Research Workflow

Dự án được triển khai theo quy trình:

1. **Problem Understanding:** Tìm hiểu bài toán và đặc điểm ngôn ngữ tiếng Việt.
2. **Literature Review:** Nghiên cứu Baseline, Hybrid và LSTM.
3. **Data Collection:** Thu thập corpus và xác định quy ước tách từ.
4. **Data Preprocessing:** Làm sạch, chuẩn hóa và chuyển đổi dữ liệu.
5. **Dataset Annotation:** Chuẩn bị nhãn tham chiếu và BIO.
6. **Model Development:** Triển khai ba phương pháp.
7. **Model Training:** Huấn luyện LSTM và chuẩn bị tài nguyên cho Hybrid.
8. **Evaluation:** Đánh giá các mô hình trên cùng tập Test.
9. **Error Analysis:** Phân tích lỗi và so sánh kết quả.
10. **Demo & Presentation:** Xây dựng ứng dụng demo và hoàn thiện seminar.

---

## 4. Dataset & Preprocessing

### 4.1. Dataset

Bộ dữ liệu chính thức được nhóm lựa chọn: **Universal Dependencies Vietnamese Treebank (UD Vietnamese-VTB)**.
Kho lưu trữ: [UD_Vietnamese-VTB (GitHub)](https://github.com/UniversalDependencies/UD_Vietnamese-VTB)

Dữ liệu có nguồn gốc từ Vietnamese Treebank (VTB) của dự án VLSP, bao gồm các bài báo chí và đã được chuẩn hóa theo chuẩn quốc tế CoNLL-U. Dữ liệu được chia sẵn thành 3 tập train, dev và test.

**Thống kê quy mô ngữ liệu:**
- **Train (`vi_vtb-ud-train.conllu`)**: 1.400 câu – 20.215 token từ – 24.973 đơn vị đầu vào – 21.345 âm tiết ước tính.
- **Dev (`vi_vtb-ud-dev.conllu`)**: 1.123 câu – 26.162 token từ – 31.927 đơn vị đầu vào – 27.390 âm tiết ước tính.
- **Test (`vi_vtb-ud-test.conllu`)**: 800 câu – 11.692 token từ – 13.857 đơn vị đầu vào – 11.839 âm tiết ước tính.
- **Tổng cộng**: 3.323 câu, 58.069 token từ, 70.757 đơn vị đầu vào, 60.574 âm tiết ước tính.

Toàn bộ dữ liệu gốc được lưu giữ nguyên trạng tại `data/raw/ud_vtb/`. Chi tiết cấu trúc dữ liệu và hướng dẫn sử dụng xem tại [docs/methodology/dataset.md](docs/methodology/dataset.md) và [data/raw/README.md](data/raw/README.md).

### 4.2. Data Preprocessing

- Chuẩn hóa Unicode và văn bản.
- Xử lý dấu câu và các ký tự đặc biệt.
- Kiểm tra dữ liệu thiếu hoặc lỗi định dạng.
- Thống nhất quy tắc phân tách từ.
- Chuyển đổi dữ liệu sang định dạng phục vụ từng mô hình.

### 4.3. Dataset Split

Dữ liệu được chia thành:
- Training Set: Huấn luyện mô hình.
- Validation Set: Điều chỉnh cấu hình và theo dõi hiệu quả.
- Test Set: Đánh giá cuối cùng.

Ba phương pháp sử dụng cùng một tập Test và cùng một bộ dữ liệu tham chiếu.

Không sử dụng thông tin từ Test Set để xây dựng từ điển, Bigram hoặc điều chỉnh mô hình.

---

## 5. Experimental Evaluation

Các mô hình được đánh giá bằng những chỉ số:

| Metric | Description |
|---|---|
| Precision | Tỷ lệ từ dự đoán đúng trên tổng số từ dự đoán |
| Recall | Tỷ lệ từ dự đoán đúng trên tổng số từ thực tế |
| F1-score | Trung bình điều hòa Precision và Recall |
| OOV Recall | Khả năng nhận diện đúng các từ OOV |
| Inference Time | Thời gian xử lý dữ liệu |

Việc xác định một từ dự đoán đúng dựa trên sự trùng khớp chính xác về ranh giới từ với dữ liệu tham chiếu.

### Error Analysis

Nhóm tập trung phân tích các trường hợp:
- Nhập nhằng ranh giới từ.
- Từ ghép nhiều âm tiết.
- Tên riêng và thuật ngữ.
- Từ mới, từ lóng và OOV.
- Dấu câu và các mẫu văn bản đặc biệt.

Kết quả sẽ được trình bày bằng bảng thống kê, biểu đồ, và ví dụ thực tế.

---

## 6. Demo Application

Dự án hướng tới xây dựng ứng dụng demo bằng Streamlit.

**Các chức năng dự kiến:**
- Nhập văn bản tiếng Việt tùy ý.
- Thực hiện tách từ bằng ba phương pháp.
- Hiển thị kết quả phân tách song song.
- So sánh kết quả giữa các phương pháp.
- Hiển thị thời gian xử lý.
- Minh họa các trường hợp mô hình dự đoán khác nhau.

---

## 7. Repository Structure

```text
vietnamese-word-segmentation/
├── configs/             # Model and data configurations
├── data/                # Raw, processed and custom test data
├── notebooks/           # EDA and experimental notebooks
├── src/
│   ├── baseline/        # Regex and whitespace baseline
│   ├── hybrid/          # Dictionary + Maximum Matching + Bigram
│   ├── lstm/            # LSTM model and training pipeline
│   ├── data/             # Preprocessing and annotation
│   ├── evaluation/       # Metrics and error analysis
│   └── utils/            # Shared utilities
├── scripts/              # Execution scripts
├── tests/                # Unit tests
├── results/              # Metrics, logs and figures
├── app/                  # Streamlit demo
├── docs/
│   ├── theory/           # Theoretical foundations
│   ├── methodology/      # Dataset and experimental design
│   ├── report/           # Seminar report
│   ├── slides/           # Presentation slides
│   └── presentation/     # Speaker notes and demo guide
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 8. Team Responsibilities

Dự án được thực hiện bởi nhóm 10 thành viên.

| Member | Responsibility |
|---|---|
| TV01 (Tâm) | Problem Overview & Research Objectives |
| TV02 (Dương) | Hybrid Approach & Literature Review |
| TV03 (Hải) | Dataset Collection & Documentation |
| TV04 (Phát) | EDA, Preprocessing & BIO Annotation |
| TV05 (Thành) | Hybrid Model Implementation |
| TV06 (Khang) | LSTM Model Implementation |
| TV07 (Phương) | Evaluation & Error Analysis |
| TV08 (Hoàng) | RNN/LSTM Theoretical Foundation |
| TV09 (My) | Slides & Content Integration |
| TV10 (Phúc) | Presentation Script & Demo |

Chi tiết nhiệm vụ, sản phẩm bàn giao và deadline được trình bày trong `docs/project_plan.md`.

---

## 9. Project Timeline

| Period | Milestone |
|---|---|
| 13–14/11/2026 | Project Planning & Repository Setup |
| 15–17/11/2026 | Literature Review & Data Collection |
| 18–21/11/2026 | Preprocessing & Model Implementation |
| 22–23/11/2026 | Evaluation & Error Analysis |
| 24–25/11/2026 | Slides, Demo & Rehearsal |
| 26/11/2026 | Seminar Presentation |

---

## 10. Technologies

Các công nghệ dự kiến:
- Python
- PyTorch hoặc TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Git & GitHub

Phiên bản thư viện và framework sử dụng thực tế sẽ được ghi lại trong `requirements.txt`.

---

## 11. Expected Deliverables

1. Báo cáo lý thuyết về Vietnamese Word Segmentation, Hybrid và LSTM.
2. Bộ mã nguồn triển khai Baseline, Hybrid và LSTM.
3. Pipeline tiền xử lý và chuẩn bị dữ liệu.
4. Kết quả huấn luyện và đánh giá.
5. Báo cáo so sánh và phân tích lỗi.
6. Ứng dụng demo Streamlit.
7. Slide seminar và tài liệu thuyết trình.

---

## 12. References

1. Phuong, L. H., Huyen, N. T. M., Roussanaly, A., & Vinh, H. T. (2008). *A Hybrid Approach to Word Segmentation of Vietnamese Texts*. https://doi.org/10.1007/978-3-540-88282-4_23
2. VLSP 2013 – Word Segmentation & POS Tagging Task. https://vlsp.org.vn/vlsp2013/eval/ws-pos
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

---

**Project Status:** In Progress

**Seminar Date:** 26 November 2026

**Project Focus:** Vietnamese Word Segmentation – From Rule-based and Statistical Methods to Deep Learning.
