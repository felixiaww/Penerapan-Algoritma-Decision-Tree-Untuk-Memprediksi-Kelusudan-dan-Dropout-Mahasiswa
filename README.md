# Penerapan Algoritma *Decision Tree Classifier* dalam Memprediksi Status Kelulusan dan Risiko *Dropout* Mahasiswa

## Deskripsi Proyek

Proyek ini merupakan implementasi *machine learning* menggunakan algoritma **Decision Tree Classifier** untuk melakukan klasifikasi status akademik mahasiswa berdasarkan dataset *Predict Students' Dropout and Academic Success*.

Tujuan proyek ini adalah mempelajari penerapan metode klasifikasi dalam menganalisis data mahasiswa dan memprediksi kategori pada variabel target (`Target`). Pendekatan ini dapat digunakan untuk mengeksplorasi hubungan antara karakteristik mahasiswa dan status akademiknya, serta memahami bagaimana algoritma *Decision Tree* menghasilkan prediksi berdasarkan pola yang ditemukan dalam data.

Proyek dikembangkan menggunakan Python melalui Jupyter Notebook, mulai dari eksplorasi data, preprocessing, visualisasi, pembagian data latih dan data uji, pelatihan model, evaluasi performa, hingga penyimpanan model.

## Tujuan Proyek

1. Memahami karakteristik dan struktur dataset mahasiswa.
2. Melakukan preprocessing untuk mempersiapkan data sebelum pemodelan.
3. Menganalisis distribusi data dan hubungan antarvariabel melalui visualisasi.
4. Menerapkan algoritma *Decision Tree Classifier* untuk klasifikasi status akademik mahasiswa.
5. Mengevaluasi hasil klasifikasi menggunakan metrik evaluasi yang tersedia pada Scikit-learn.
6. Menyimpan model yang telah dilatih agar dapat digunakan kembali.

## Dataset

Dataset yang digunakan bernama **Predict Students' Dropout and Academic Success.csv**.

Dataset dibaca menggunakan Pandas dan digunakan sebagai sumber data untuk proses analisis serta pemodelan. Variabel `Target` ditetapkan sebagai variabel yang akan diprediksi, sedangkan kolom lainnya digunakan sebagai fitur masukan.

File dataset harus tersedia di lokasi yang sesuai dengan path pada source code agar notebook dapat dijalankan. Dataset tidak otomatis tersedia hanya dengan mengunduh file notebook ini.

## Teknologi dan Library

- **Python** — bahasa pemrograman untuk pengolahan data dan implementasi machine learning.
- **Jupyter Notebook** — lingkungan untuk menjalankan kode secara bertahap.
- **Pandas** — membaca, mengolah, dan menganalisis data dalam bentuk DataFrame.
- **Matplotlib** — membuat grafik dan visualisasi data.
- **Seaborn** — membuat visualisasi statistik, seperti box plot dan heatmap.
- **Scikit-learn** — melakukan preprocessing, membagi dataset, membangun model klasifikasi, dan mengevaluasi prediksi.
- **Joblib** — menyimpan model yang telah dilatih ke dalam file `.pkl`.

## Tahapan Pengerjaan

### 1. Memuat dan Mengeksplorasi Dataset

Dataset dimuat menggunakan `pandas.read_csv()`. Selanjutnya, dilakukan pemeriksaan beberapa bagian data, informasi tipe kolom, statistik deskriptif, dan jumlah nilai yang hilang.

Tahap ini bertujuan untuk memahami struktur dataset sebelum dilakukan pengolahan lebih lanjut.

### 2. Preprocessing Data

Preprocessing dilakukan melalui beberapa proses berikut:

- **Penanganan missing values:** kode awal mengisi nilai yang hilang pada kolom numerik menggunakan median dan pada kolom kategorikal menggunakan modus. Kolom dengan jumlah nilai hilang minimal 1.000 juga dipisahkan untuk dihapus dari DataFrame pada tahap tersebut.
- **Deteksi outlier:** menggunakan metode *Interquartile Range* (IQR) dengan batas 1,5 kali IQR untuk mengidentifikasi dan menyaring baris yang memiliki outlier pada fitur numerik.
- **Standardisasi:** menggunakan `StandardScaler` untuk melakukan standardisasi terhadap fitur numerik.
- **Pemeriksaan duplikasi:** mengidentifikasi dan menghapus baris duplikat.
- **Encoding data kategorikal:** menerapkan `LabelEncoder` untuk mengubah nilai kategorikal menjadi representasi numerik.

Kode juga menerapkan `pd.get_dummies()` sebagai proses *One-Hot Encoding* dan melakukan pemeriksaan ulang terhadap nilai yang hilang.

### 3. Analisis dan Visualisasi Data

Tahap eksplorasi data dilakukan menggunakan Matplotlib dan Seaborn. Visualisasi yang terdapat dalam notebook meliputi:

- Box plot untuk mengamati distribusi dan outlier.
- Histogram untuk melihat distribusi variabel.
- Heatmap untuk menampilkan korelasi antarvariabel.
- Bar plot untuk menampilkan korelasi absolut setiap variabel terhadap `Target`.

Visualisasi tersebut membantu memahami karakteristik data dan hubungan antarvariabel sebelum proses klasifikasi.

### 4. Menentukan Fitur dan Target

Data dipisahkan menjadi dua bagian:

- **X (fitur):** seluruh kolom selain `Target`.
- **y (target):** kolom `Target` yang menjadi label klasifikasi.

Pemisahan ini diperlukan agar model mempelajari hubungan antara fitur masukan dan label yang akan diprediksi.

### 5. Membagi Data Latih dan Data Uji

Notebook menggunakan fungsi `train_test_split()` untuk membagi data dengan konfigurasi:

- Data latih: 80%.
- Data uji: 20%.
- `random_state=1` untuk menjaga konsistensi pembagian data ketika dijalankan kembali pada kondisi yang sama.

Data latih digunakan untuk membangun model, sedangkan data uji digunakan untuk mengevaluasi hasil prediksi pada data yang tidak digunakan dalam pelatihan.

### 6. Membangun Model Decision Tree Classifier

Model dibuat menggunakan `DecisionTreeClassifier` dari Scikit-learn dengan konfigurasi `random_state=42`.

Model dilatih menggunakan `x_train` dan `y_train` melalui metode `fit()`. Setelah pelatihan, model menghasilkan prediksi terhadap `x_test` menggunakan metode `predict()`.

Algoritma *Decision Tree* membangun struktur keputusan berdasarkan fitur dalam dataset untuk menentukan kelas prediksi. Hasil klasifikasi bergantung pada pola yang dipelajari dari data pelatihan.

### 7. Evaluasi Model

Performa model dievaluasi menggunakan beberapa komponen:

- **Accuracy Score:** mengukur proporsi prediksi yang benar terhadap seluruh data uji.
- **Classification Report:** menampilkan metrik klasifikasi, termasuk precision, recall, F1-score, dan support untuk setiap kelas.
- **Confusion Matrix:** membandingkan label aktual dengan label hasil prediksi untuk melihat pola klasifikasi yang benar maupun salah.

Nilai evaluasi yang sebenarnya diperoleh ketika notebook dijalankan. Dokumentasi ini tidak mencantumkan angka akurasi tertentu karena nilai hasil eksekusi tidak dijadikan dasar dalam penulisan ini.

### 8. Menyimpan Model

Model disimpan menggunakan library Joblib ke dalam file:

`model_decision_tree.pkl`

Penyimpanan dilakukan dengan `joblib.dump()` setelah model dilatih. File tersebut berfungsi sebagai artefak model yang dapat dimuat kembali untuk penggunaan selanjutnya, dengan memperhatikan kesesuaian preprocessing dan struktur fitur yang digunakan saat pelatihan.

## Struktur Proyek

Struktur file yang dapat digunakan untuk menyimpan proyek di GitHub adalah sebagai berikut:

```text
decision-tree-classifier-mahasiswa/
├── Machine Learning Mahasiswa.ipynb
├── Predict Students' Dropout and Academic Success.csv
├── model_decision_tree.pkl
└── README.md
```

Keterangan:

- `Machine Learning Mahasiswa.ipynb`: notebook yang berisi source code, preprocessing, visualisasi, pelatihan, dan evaluasi model.
- `Predict Students' Dropout and Academic Success.csv`: dataset yang dibutuhkan untuk menjalankan notebook.
- `model_decision_tree.pkl`: file model hasil pelatihan, jika notebook telah dijalankan hingga tahap penyimpanan.
- `README.md`: dokumentasi proyek.

File dataset dan model `.pkl` tidak otomatis dibuat hanya dengan menyalin source code notebook. Keduanya perlu disediakan atau dihasilkan melalui proses yang sesuai.

## Cara Menjalankan Proyek

### 1. Persiapkan Lingkungan

Pastikan Python dan Jupyter Notebook telah tersedia pada perangkat.

### 2. Instal Library

Jalankan perintah berikut pada terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib notebook
```

### 3. Siapkan Dataset

Tempatkan file `Predict Students' Dropout and Academic Success.csv` pada lokasi yang sesuai dengan path yang digunakan di dalam notebook.

### 4. Jalankan Notebook

Buka `Machine Learning Mahasiswa.ipynb` menggunakan Jupyter Notebook atau lingkungan lain yang mendukung file `.ipynb`.

Jalankan setiap sel kode sesuai urutannya. Jika proses berhasil dan sel penyimpanan model dijalankan, model akan disimpan sebagai `model_decision_tree.pkl`.

## Hasil yang Diharapkan

Berdasarkan fungsi yang ditulis dalam notebook, proses ini menghasilkan:

1. Informasi struktur dan statistik dataset.
2. Visualisasi distribusi, outlier, dan korelasi data.
3. Data latih dan data uji untuk klasifikasi.
4. Model klasifikasi menggunakan *Decision Tree Classifier*.
5. Hasil prediksi pada data uji.
6. Laporan evaluasi berupa accuracy score, classification report, dan confusion matrix.
7. File model hasil pelatihan dalam format `.pkl`.

Hasil aktual, termasuk nilai akurasi dan kemampuan model dalam membedakan setiap kelas, perlu dilihat dari hasil eksekusi notebook.

## Catatan Implementasi

Beberapa tahapan preprocessing dalam notebook perlu diperhatikan ketika proyek dikembangkan lebih lanjut. Pada salah satu sel, dataset dimuat kembali ke variabel `df` dari file CSV asli. Akibatnya, hasil penanganan missing values pada sel sebelumnya tidak otomatis diteruskan ke tahap pemodelan.

Notebook juga membuat hasil One-Hot Encoding, tetapi variabel yang digunakan dalam pemodelan berikutnya adalah `df_lencoder`, yang dibentuk melalui Label Encoding. Oleh karena itu, kedua metode encoding tersebut tidak digunakan secara bersamaan sebagai fitur model pada tahap akhir.

Selain itu, pemfilteran outlier dan standardisasi perlu diperiksa kembali agar daftar fitur numerik tetap sesuai dengan kolom yang tersedia. Untuk penggunaan prediksi pada data baru, seluruh langkah preprocessing sebaiknya disusun secara konsisten dan dipelajari hanya dari data latih guna menghindari kebocoran informasi (*data leakage*).

Catatan ini penting agar hasil evaluasi dapat ditafsirkan secara tepat dan model dapat digunakan kembali dengan proses yang konsisten.

## Kesimpulan

Proyek ini menerapkan algoritma *Decision Tree Classifier* sebagai metode klasifikasi untuk menganalisis dan memprediksi variabel `Target` pada dataset mahasiswa. Tahapannya mencakup eksplorasi data, preprocessing, visualisasi, pembagian data, pelatihan model, evaluasi, dan penyimpanan model.

Melalui proyek ini, proses penerapan machine learning pada data pendidikan dapat dipelajari secara terstruktur. Namun, penggunaan model sebagai sistem peringatan dini bagi mahasiswa yang berisiko *dropout* masih memerlukan evaluasi lebih lanjut, validasi hasil, dan pengujian pada data yang sesuai sebelum digunakan untuk mendukung pengambilan keputusan.

## Author

Dikembangkan sebagai proyek pembelajaran dan penerapan *Machine Learning* menggunakan Python dan Scikit-learn.
