# Penerapan Algoritma Decision Tree Classifier dalam Memprediksi Status Kelulusan dan Risiko Dropout Mahasiswa

## Deskripsi Proyek

Proyek ini merupakan implementasi Machine Learning untuk melakukan klasifikasi status akademik mahasiswa menggunakan algoritma **Decision Tree Classifier**. Proyek dikembangkan menggunakan Python melalui Jupyter Notebook dengan memanfaatkan dataset mahasiswa yang mencakup informasi akademik, demografi, dan faktor sosial-ekonomi.

Tujuan proyek ini adalah menerapkan tahapan Machine Learning, mulai dari eksplorasi data, preprocessing, visualisasi, pelatihan model, evaluasi performa, hingga penyimpanan model dalam format `.pkl`.

Model menggunakan kolom `Target` sebagai variabel yang diprediksi. Status akademik dalam dataset terdiri dari tiga kategori, yaitu *Dropout*, *Enrolled*, dan *Graduate*.

## Tujuan Proyek

- Memahami karakteristik dataset mahasiswa melalui eksplorasi data.
- Melakukan preprocessing untuk mempersiapkan data sebelum pemodelan.
- Membuat visualisasi untuk memahami distribusi dan korelasi data.
- Membangun model klasifikasi menggunakan Decision Tree.
- Mengevaluasi performa model menggunakan metrik klasifikasi.
- Menyimpan model yang telah dilatih agar dapat digunakan kembali.

## Informasi Sumber / Dataset

**Judul Dataset:** Predict Students' Dropout and Academic Success (Higher Education Predictors of Student Retention)

**Pengunggah / Publisher di Kaggle:** The Devastator

**Sumber Dataset:** Kaggle

**Link Dataset:**  
https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

### Deskripsi Dataset

Dataset ini memuat informasi mahasiswa di perguruan tinggi, termasuk karakteristik akademik, demografi, serta faktor sosial-ekonomi. Beberapa variabel yang tersedia berkaitan dengan status perkawinan, kualifikasi pendidikan orang tua, kondisi biaya kuliah, dan performa akademik mahasiswa.

Dataset digunakan untuk mempelajari klasifikasi status mahasiswa ke dalam tiga kategori, yaitu:

- **Dropout:** mahasiswa yang berhenti atau tidak melanjutkan studi.
- **Enrolled:** mahasiswa yang masih terdaftar atau menjalani studi.
- **Graduate:** mahasiswa yang telah menyelesaikan studi.

Dataset ini mendukung analisis faktor-faktor yang berkaitan dengan keberhasilan akademik dan keberlanjutan pendidikan mahasiswa.

**Catatan:** Dataset yang digunakan dalam proyek ini diperoleh melalui halaman Kaggle yang dicantumkan di atas. Jumlah data yang digunakan pada tahap pemodelan dapat berbeda dari jumlah data asli karena tahapan preprocessing di dalam notebook.

## Teknologi dan Library

Proyek ini menggunakan Python dengan beberapa library berikut:

- **Pandas:** membaca, mengolah, dan menganalisis data.
- **Seaborn:** membuat visualisasi distribusi data, boxplot, dan heatmap.
- **Matplotlib:** menampilkan grafik dan visualisasi hasil analisis.
- **Scikit-learn:** melakukan encoding, standardisasi, pembagian dataset, pelatihan model, dan evaluasi klasifikasi.
- **Joblib:** menyimpan model Machine Learning ke dalam file `.pkl`.

## Tahapan Pengerjaan

### 1. Memuat Dataset

Dataset dibaca dari file CSV menggunakan fungsi `pandas.read_csv()`.

File yang digunakan dalam notebook adalah:

`Predict Students' Dropout and Academic Success.csv`

Setelah dataset dimuat, dilakukan pemeriksaan awal untuk memahami struktur data, tipe kolom, jumlah observasi, dan statistik deskriptif.

### 2. Eksplorasi Data

Eksplorasi data dilakukan menggunakan fungsi seperti `info()` dan `describe(include="all")`. Tahap ini bertujuan memahami karakteristik dataset sebelum melanjutkan ke proses preprocessing dan pemodelan.

### 3. Pemeriksaan Missing Values

Pemeriksaan nilai kosong dilakukan pada setiap kolom. Pada bagian preprocessing awal, nilai kosong pada kolom numerik ditangani menggunakan median, sedangkan kolom kategorikal menggunakan modus. Notebook juga menentukan kolom dengan jumlah missing values minimal 1.000 untuk dihapus.

Dataset kemudian dimuat kembali pada tahapan analisis berikutnya. Pada data yang dimuat ulang tersebut, pemeriksaan menunjukkan tidak adanya missing values. Dengan demikian, hasil preprocessing awal tidak seluruhnya menjadi input bagi tahapan berikutnya.

### 4. Deteksi dan Penanganan Outlier

Outlier dianalisis menggunakan boxplot dan metode Interquartile Range (IQR). Batas bawah dan batas atas ditentukan berdasarkan kuartil pertama, kuartil ketiga, dan rentang antarkuartil.

Baris yang memiliki nilai di luar batas IQR pada setidaknya satu fitur numerik dikeluarkan dari data pada tahapan tersebut. Visualisasi boxplot kemudian digunakan untuk melihat distribusi data setelah penyaringan.

### 5. Standardisasi Data Numerik

Standardisasi dilakukan menggunakan `StandardScaler` dari Scikit-learn. Metode ini mengubah skala fitur numerik agar memiliki rata-rata mendekati nol dan standar deviasi mendekati satu.

Notebook juga menampilkan histogram sebelum dan sesudah standardisasi untuk membandingkan distribusi fitur numerik.

### 6. Pemeriksaan Data Duplikat

Data duplikat diperiksa menggunakan fungsi `duplicated()` dan dihapus menggunakan `drop_duplicates()`. Tahap ini bertujuan mengurangi baris data yang teridentifikasi sebagai duplikat.

### 7. Encoding Variabel Kategorikal

Notebook menggunakan dua metode encoding, yaitu:

- **One-Hot Encoding** melalui `pandas.get_dummies()`.
- **Label Encoding** melalui `LabelEncoder` dari Scikit-learn.

Hasil Label Encoding disimpan dalam DataFrame `df_lencoder` dan digunakan pada tahapan analisis serta pemodelan berikutnya.

### 8. Visualisasi dan Analisis Korelasi

Visualisasi digunakan untuk memahami distribusi dan hubungan antarvariabel dalam dataset. Visualisasi yang dibuat meliputi histogram, boxplot, heatmap korelasi, serta bar plot korelasi absolut terhadap kolom `Target`.

Analisis korelasi digunakan untuk melihat hubungan statistik antarvariabel. Korelasi tidak secara langsung menunjukkan hubungan sebab-akibat.

### 9. Pemisahan Fitur dan Target

Data dipisahkan menjadi dua bagian:

- `X`: fitur atau variabel input selain kolom `Target`.
- `y`: kolom `Target` sebagai label yang akan diprediksi.

Pembagian data dilakukan menggunakan `train_test_split()` dengan proporsi 80% data latih dan 20% data uji serta `random_state=1`.

### 10. Pelatihan Model Decision Tree

Algoritma yang digunakan adalah `DecisionTreeClassifier` dari Scikit-learn.

Model diinisialisasi dengan `random_state=42`, kemudian dilatih menggunakan data latih. Setelah proses pelatihan selesai, model digunakan untuk menghasilkan prediksi terhadap data uji.

Decision Tree membentuk aturan keputusan berdasarkan fitur yang tersedia untuk melakukan klasifikasi terhadap data.

### 11. Evaluasi Model

Evaluasi dilakukan menggunakan beberapa metrik:

- **Accuracy Score:** mengukur proporsi prediksi yang benar dari seluruh data uji.
- **Precision:** mengukur ketepatan prediksi untuk setiap kelas.
- **Recall:** mengukur kemampuan model mengenali data aktual pada setiap kelas.
- **F1-score:** menggabungkan precision dan recall.
- **Confusion Matrix:** membandingkan label aktual dengan hasil prediksi model.

Berdasarkan output evaluasi yang tersimpan dalam notebook, hasil model adalah sebagai berikut.

| Metrik | Hasil |
|---|---:|
| Akurasi | 61,58% |
| Jumlah data setelah preprocessing | 948 |
| Data latih | 758 |
| Data uji | 190 |

### Classification Report

| Label hasil encoding | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0,36 | 0,57 | 0,44 | 23 |
| 1 | 0,47 | 0,44 | 0,45 | 48 |
| 2 | 0,76 | 0,70 | 0,73 | 119 |

Label 0, 1, dan 2 merupakan representasi numerik dari kelas target setelah encoding. Nama kelas untuk setiap angka tidak dicantumkan secara terpisah karena pemetaan tersebut tidak ditampilkan dalam laporan evaluasi.

Hasil evaluasi memperlihatkan bahwa performa model berbeda pada setiap kelas. Oleh karena itu, akurasi perlu dianalisis bersama precision, recall, F1-score, dan confusion matrix.

### 12. Penyimpanan Model

Model Decision Tree yang telah dilatih disimpan menggunakan `joblib.dump()` dengan nama file:

`model_decision_tree.pkl`

File ini menyimpan objek model yang telah dilatih agar dapat dimuat kembali. Notebook ini berfokus pada proses analisis dan pemodelan; aplikasi web atau API untuk melakukan prediksi secara langsung belum termasuk dalam implementasinya.

## Alur Pengerjaan

Alur utama proyek ini meliputi:

1. Memuat dataset dari file CSV.
2. Melakukan eksplorasi data.
3. Memeriksa missing values.
4. Mendeteksi dan menangani outlier.
5. Melakukan standardisasi data numerik.
6. Memeriksa dan menghapus data duplikat.
7. Melakukan encoding variabel kategorikal.
8. Membuat visualisasi dan analisis korelasi.
9. Memisahkan fitur dan target.
10. Membagi data menjadi data latih dan data uji.
11. Melatih model Decision Tree Classifier.
12. Mengevaluasi hasil klasifikasi.
13. Menyimpan model dalam format `.pkl`.

## Kesimpulan

Proyek ini menerapkan algoritma Decision Tree Classifier untuk melakukan klasifikasi status akademik mahasiswa berdasarkan dataset yang diperoleh melalui Kaggle. Tahapan pengerjaan meliputi eksplorasi data, preprocessing, visualisasi, encoding, pelatihan model, evaluasi, dan penyimpanan model.

Berdasarkan hasil evaluasi yang tersimpan dalam notebook, model memperoleh akurasi sebesar **61,58%** pada data uji. Hasil tersebut menjadi gambaran performa model pada eksperimen ini dan dapat dijadikan dasar untuk pengembangan lebih lanjut, terutama dalam perbaikan preprocessing, pengendalian data leakage, serta evaluasi model yang lebih menyeluruh.

## Referensi

The Devastator. *Predict Students' Dropout and Academic Success (Higher Education Predictors of Student Retention).* Kaggle.

Link dataset: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
