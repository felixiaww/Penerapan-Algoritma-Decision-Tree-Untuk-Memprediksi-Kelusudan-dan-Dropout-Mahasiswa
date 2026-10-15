# Penerapan Algoritma Decision Tree Classifier dalam Memprediksi Status Kelulusan dan Risiko Dropout Mahasiswa

## Deskripsi Proyek

Proyek ini merupakan implementasi Machine Learning untuk melakukan klasifikasi status akademik mahasiswa menggunakan algoritma **Decision Tree Classifier**. Model dibangun dengan memanfaatkan dataset *Predict Students' Dropout and Academic Success* yang berisi informasi terkait latar belakang mahasiswa, proses penerimaan, serta performa akademik.

Tujuan proyek ini adalah mempelajari penerapan tahapan Machine Learning, mulai dari eksplorasi data, preprocessing, visualisasi, pembentukan model klasifikasi, evaluasi performa, hingga penyimpanan model dalam format `.pkl`.

Model menggunakan kolom `Target` sebagai variabel yang diprediksi. Dataset aslinya memiliki tiga kategori status akademik, yaitu *Dropout*, *Enrolled*, dan *Graduate*.

## Tujuan

- Memahami karakteristik dataset mahasiswa melalui eksplorasi data.
- Melakukan preprocessing untuk mempersiapkan data sebelum pemodelan.
- Memvisualisasikan distribusi data dan korelasi antarvariabel.
- Membangun model klasifikasi menggunakan Decision Tree.
- Mengevaluasi hasil prediksi menggunakan metrik klasifikasi.
- Menyimpan model yang telah dilatih agar dapat digunakan kembali.

## Sumber Dataset

Dataset yang digunakan adalah **Predict Students' Dropout and Academic Success**, yang tersedia melalui UCI Machine Learning Repository.

**Sumber resmi:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Dataset ini dikembangkan berdasarkan data dari sebuah institusi pendidikan tinggi dan mencakup informasi mahasiswa dari berbagai program studi. Variabelnya meliputi informasi penerimaan, karakteristik demografis, kondisi sosial ekonomi, serta performa akademik pada semester pertama dan kedua.

Berdasarkan dokumentasi UCI, dataset asli memiliki 4.424 baris data dan 36 fitur, dengan `Target` sebagai variabel klasifikasi. Dataset tersebut telah melalui proses preprocessing oleh penyedia aslinya.

**Referensi dataset:**

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Lisensi dataset: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Teknologi dan Library

Proyek ini menggunakan Python dengan beberapa library berikut:

- **Pandas** — membaca, mengolah, dan menganalisis data dalam bentuk DataFrame.
- **Seaborn** — membuat visualisasi distribusi data, boxplot, dan heatmap.
- **Matplotlib** — menampilkan grafik dan visualisasi hasil analisis.
- **Scikit-learn** — melakukan encoding, standardisasi, pembagian data, pembuatan model, dan evaluasi klasifikasi.
- **Joblib** — menyimpan model Machine Learning ke dalam file `.pkl`.

## Tahapan Pengerjaan

### 1. Memuat Dataset

Dataset dibaca dari file CSV menggunakan `pandas.read_csv()`. Notebook menggunakan nama file:

`Predict Students' Dropout and Academic Success.csv`

Data kemudian diperiksa untuk melihat contoh baris, tipe data, jumlah data, serta ringkasan statistik deskriptif.

### 2. Eksplorasi Data

Eksplorasi dilakukan menggunakan `DataFrame.info()` dan `describe(include="all")`. Tahapan ini digunakan untuk memahami struktur dataset, tipe variabel, jumlah observasi, dan statistik deskriptif sebelum analisis lanjutan.

### 3. Pemeriksaan Missing Values

Notebook memeriksa nilai yang hilang pada setiap kolom. Pada bagian preprocessing awal, nilai yang hilang pada kolom numerik diisi menggunakan median, sedangkan kolom kategorikal diisi menggunakan modus. Kolom dengan jumlah missing values minimal 1.000 juga ditentukan untuk dihapus.

Namun, dataset yang dimuat ulang pada tahapan analisis berikutnya tercatat tidak memiliki missing values. Selain itu, proses pemuatan ulang tersebut membuat hasil preprocessing awal tidak seluruhnya digunakan dalam tahapan berikutnya.

### 4. Deteksi dan Penanganan Outlier

Outlier diperiksa menggunakan boxplot dan metode Interquartile Range (IQR). Batas deteksi dihitung berdasarkan kuartil pertama (Q1), kuartil ketiga (Q3), dan rentang antarkuartil (IQR).

Baris yang memiliki nilai di luar batas IQR pada setidaknya satu fitur numerik dikeluarkan dari data yang digunakan pada tahap tersebut. Setelah itu, boxplot kembali dibuat untuk melihat distribusi data setelah penyaringan.

### 5. Standardisasi Data Numerik

Standardisasi dilakukan menggunakan `StandardScaler` dari Scikit-learn. Proses ini mengubah skala fitur numerik agar memiliki rata-rata mendekati 0 dan standar deviasi mendekati 1.

Notebook juga menampilkan histogram sebelum dan sesudah standardisasi untuk membandingkan distribusi salah satu fitur numerik.

### 6. Pemeriksaan Data Duplikat

Baris duplikat diperiksa menggunakan `DataFrame.duplicated()`. Baris yang teridentifikasi sebagai duplikat kemudian dihapus menggunakan `drop_duplicates()`.

### 7. Encoding Variabel Kategorikal

Notebook menerapkan dua pendekatan encoding:

- **One-Hot Encoding** menggunakan `pandas.get_dummies()`.
- **Label Encoding** menggunakan `LabelEncoder` dari Scikit-learn.

Hasil Label Encoding disimpan dalam DataFrame `df_lencoder` dan digunakan sebagai dasar untuk tahapan analisis korelasi serta pemodelan berikutnya.

### 8. Visualisasi dan Analisis Korelasi

Beberapa visualisasi dibuat untuk memahami karakteristik data, yaitu:

- Histogram untuk melihat distribusi setiap variabel.
- Histogram beberapa fitur numerik.
- Heatmap untuk melihat korelasi antarvariabel.
- Bar plot untuk menampilkan korelasi absolut setiap variabel terhadap `Target`.

Visualisasi tersebut membantu mengeksplorasi pola data. Nilai korelasi sendiri tidak membuktikan hubungan sebab-akibat.

### 9. Pemisahan Fitur dan Target

Variabel dipisahkan menjadi dua bagian:

- `X`: seluruh fitur selain kolom `Target`.
- `y`: kolom `Target` sebagai label yang akan diprediksi.

Dataset kemudian dibagi menggunakan `train_test_split()` dengan proporsi 80% untuk data latih dan 20% untuk data uji. Parameter `random_state=1` digunakan agar pembagian data dapat direproduksi.

### 10. Pembuatan dan Pelatihan Model

Algoritma yang digunakan adalah `DecisionTreeClassifier` dari Scikit-learn.

Model diinisialisasi dengan `random_state=42`, kemudian dilatih menggunakan `x_train` dan `y_train`. Setelah pelatihan, model menghasilkan prediksi terhadap `x_test`.

Decision Tree melakukan klasifikasi berdasarkan aturan keputusan yang dibentuk dari fitur-fitur data selama proses pelatihan.

### 11. Evaluasi Model

Evaluasi dilakukan menggunakan:

- **Accuracy Score** untuk mengukur proporsi prediksi yang benar.
- **Classification Report** untuk menampilkan precision, recall, dan F1-score setiap kelas.
- **Confusion Matrix** untuk membandingkan label aktual dengan hasil prediksi model.

Berdasarkan output yang tersimpan di notebook, hasil evaluasi model adalah:

| Metrik | Hasil |
|---|---:|
| Akurasi | 61,58% |
| Jumlah data setelah preprocessing yang digunakan untuk pemodelan | 948 |
| Data latih | 758 |
| Data uji | 190 |

Akurasi tersebut merupakan hasil pada pembagian data uji yang digunakan dalam notebook, bukan jaminan performa pada seluruh populasi mahasiswa.

**Classification Report:**

| Label hasil encoding | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0,36 | 0,57 | 0,44 | 23 |
| 1 | 0,47 | 0,44 | 0,45 | 48 |
| 2 | 0,76 | 0,70 | 0,73 | 119 |

Label 0, 1, dan 2 merupakan hasil encoding dari kelas pada kolom `Target`. Nilai tersebut tidak dituliskan sebagai nama kelas tertentu karena pemetaan label tidak ditampilkan secara eksplisit dalam laporan evaluasi notebook.

Hasil evaluasi menunjukkan bahwa performa model berbeda pada setiap kelas. Oleh karena itu, akurasi perlu dibaca bersama precision, recall, F1-score, dan confusion matrix.

### 12. Penyimpanan Model

Model Decision Tree yang telah dilatih disimpan menggunakan `joblib.dump()` dengan nama file:

`model_decision_tree.pkl`

File tersebut berisi objek model yang sudah dilatih sehingga dapat dimuat kembali untuk penggunaan berikutnya. Namun, notebook ini belum menyediakan aplikasi web, API prediksi, atau pipeline lengkap untuk memproses data baru secara otomatis.

## Struktur Proses

Secara ringkas, alur pengerjaan pada notebook adalah:

1. Memuat dan mengeksplorasi dataset.
2. Memeriksa missing values.
3. Memvisualisasikan dan menyaring outlier.
4. Melakukan standardisasi serta menghapus duplikat.
5. Melakukan encoding variabel kategorikal.
6. Menganalisis distribusi dan korelasi.
7. Memisahkan fitur dan target.
8. Membagi data menjadi training dan testing.
9. Melatih Decision Tree Classifier.
10. Mengevaluasi hasil klasifikasi.
11. Menyimpan model ke file `.pkl`.


## Kesimpulan

Proyek ini menunjukkan penerapan dasar Machine Learning untuk klasifikasi status akademik mahasiswa menggunakan Decision Tree Classifier. Tahapannya mencakup eksplorasi data, preprocessing, visualisasi, encoding, pelatihan model, evaluasi, dan penyimpanan model.

Model memperoleh akurasi sebesar **61,58%** pada data uji yang digunakan di notebook. Hasil ini dapat dijadikan dasar untuk pembelajaran dan pengembangan lanjutan, terutama dalam perbaikan preprocessing, pengendalian data leakage, dan evaluasi model yang lebih menyeluruh.

## Sumber

Judul Dataset: Predict Students' Dropout and Academic Success (Higher Education Predictors of Student Retention)

Pengunggah / Publisher di Kaggle: The Devastator

Link Dataset: Kaggle - Higher Education Predictors of Student Retention

Deskripsi Singkat: Dataset ini mencakup data akademik, demografi, serta faktor sosial-ekonomi mahasiswa (seperti status perkawinan, kualifikasi orang tua, tingkat pengeluaran/biaya kuliah, dan nilai semester) untuk memprediksi retensi serta tingkat kelulusan/dropout mahasiswa (Dropout, Enrolled, Graduate).
