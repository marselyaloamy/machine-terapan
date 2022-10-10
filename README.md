# Laporan Proyek Machine Learning - Marselya Loamy

##Domain Proyek
###Tumbuhan menjadi salah satu organisme yang mampu memproduksi senyawa toksin, bahkan dapat dimanfaatkan sebagai bahan racun oleh manusia. Tumbuhan memiliki ribuan bahkan jutaan jenis, di mana antar tumbuhan mempunyai kesamaan maupun berbedaan yang samar. Sehingga diperlukan ketelitian untuk mengetahui tumbuhan yang memiliki ciri-ciri sama. Salah satu perbedaan yang dimiliki tumbuhan terletak pada pola daun, helai daun, tulang daun dan warna daun. Dengan demikian dapat difokuskan memanfaatkan pola daun, helai daun, tulang daun dan juga warna untuk pengenalan parameter. Dari parameter tersebut sistem bisa mengenali ciri dari tumbuhan agar sistem bisa melakukan klasifikasi untuk menentukan tumbuhan beracun.Hal ini disebabkan senyawa toksin dalam tumbuhan mengandung zat kimia yang mampu menyebabkan rasa sakit bahkan kematian jika terjadi kontak langsung dengan manusia atau hewan baik dihirup atau dimakan dengan kadar yang berlebihan. Kadar racun alami dalam tumbuhan memang terbilang cukup rendah, akan tetapi terdapat beberapa jenis tumbuhan yang memiliki kadar racun yang tinggi sehingga membutuhkan keterampilan khusus ketika akan mengolahnya.

##Business Understanding
###Problem Statements
Menjelaskan pernyataan masalah latar belakang:
* Bagaimana membuat sistem klasifikasi tumbuhan beracun ?
* Berapa tingkat akurasi sistem klasifikasi tumbuhan beracun ?
Goals
Menjelaskan tujuan dari pernyataan masalah:
* Membuat sistem klasifikasi tumbuhan beracun.
* mengetahui seberapa akurat sebuah citra. Klasfikasi Interpretasi citra bertujuan untuk pengelompokkan atau membuat segmentasi.

#Data Understanding
`Data yang saya gunakan dalam proyek ini adalah tumbuhan beracun,tidak beracun dan tumbuhan keduanya yang dianalisa berdasarkan gambar dihutan dan ditempat yang jarang ditemukan.Jumlah data dari 3 file yaitu 2.954.
    File 1 non and toxic_image perpaduan gambar tumbuhan yang beracun dan bisa tidak beracun tergantung dengan situasi atau faktor yang dilakukan manusia.Gambar tumbuhan ini mempunyai karakteris yang beragam jadi sedikit susah untuk mengenalinya.Diambil dengan format jpg, nama gambar dikelompokkan dengan angka bukan huruf, Ukuran dan gambar berwarna pada file ini ada yang 375x500 px,500x318 px dan 500x274 px.
    File 2 nontoxic dengan gambar tumbuhan tidak beracun tentu saja tumbuhan yang aman untuk dikonsumsi maupun dikembangbiakan,karakteristik gambar ini banyak yang berbentuk daun yang segar dan lebar tapi ada juga yang memiliki daun yang tipis dengan batang yang kurus.Diambil dengan format jpg,nama gambar dikelompokkan dengan angka bukan huruf, ukuran gambar ada yang 333x500px,375x500px dan 500x500px.
    File 3 yaitu gambar tumbuhan beracun yang berbahaya dan banyak tumbuhan asing yang banyak belum diketahui ciri-ciri Bergetah, warna mencolok, daun tunggal dan tampak mengkilap atau kusam menimbulkan kematian yang mematikan.Diambil dengan format jpg. ukuran dan gambar berwarna 500x281px,375x500px dll. 
Sumber atau tautan untuk mengunduh dataset(https://www.kaggle.com/datasets/hanselliott/toxic-plant-classification)

Variabel-variabel pada Tumbuhan Beracun dataset adalah sebagai berikut:
* characteristic : Bergetah, warna mencolok, daun tunggal dan tampak mengkilap atau kusam
* impact : menimbulkan dampak buruk bagi tubuh, baik dengan menelan atau menyentuhnya. Ini berbahaya terutama bagi orang yang memiliki alergi dan kepekaan terhadap zat tertentu, dengan gejala yang biasa terjadi seperti peradangan kulit, mual, muntah, dan gatal-gatal.

#Data Preparation
Rasio pembagian dataset dengan membagi data latih menjadi latih dan validasi dengan 80/20 bertujuan karena jumlah data lebih dari cukup dan menggunakan lebih sedikit data latih.
Langkah pertama masukan file dan diunzip dengan os.listdir, dengan mengambil path ke direktori yang berisi gambar yang diurutkan dalam sub direktori dan parameter augmentasi gambar. Pada _ImageDataGenerator_ saya menargetkan nilai 0 dan 1 dengan penskala 1/255, gambar input berputar dari 0 hingga 20 derajat secara acak, _horizontal_ _flip_ menunjukkan gambar secara horizontal, gambar digeser kisaran 20 derajat dan fill_mode = mengisi area dengan pantulan gambar. 
Saya membagi data train set 80% dan 20% test set menggunakan validation split, di temukan 2364 gambar training dalam 3 kelas dan 590 gambar test dalam 3 kelas. Saya menggunakan library tensorflow dan model sequential. Di _compile_ dengan loss _categorical_ _crossentorypy_ dan optimizer Adam. Langkah terakhir memanggil model.fit untuk melatihnya.

#Modeling
Tahapannya saya memakai ImageDataGenerator yang menghasilkan gambar menggunakan teknik _Image Augmentation_ dan membagi data test dan tasting dengan parameter validation split, Saya menggunakan library tensorflow dan model sequential untuk tumpukan lapisan yang terdiri dari satu tensor input dan satu tensor output menggunakan 3 lapis convolution.

Cara kerja Conv2D untuk mendapatkan feature dari gambar. Dalam praktiknya, feature yang dicari oleh CNN filter dapat berupa tepi (edge) atau pola (pattern) lainnya dari obyek dalam gambar.diimplementasikan ke dalam pixel gambar RGB. Tiap pixel berada di posisi tertentu dalam koordinat X dan Y. Tiap pixel terdiri dari 3 kanal (channel) atau vektor (vector) warna, yaitu merah (Red), hijau (Green), dan biru (Blue). Dalam praktiknya, vektor kata disimpan array 1 dimensi.Conv2D ini saya memakai 1 16 feature map berukuran(3,3),lalu menurunkan skala gambar dengan meneruskannya melalu layer convulation dengan 32 peta fitur, ke3 64 peta fitur yang disebut Ekstraksi Fitur.

Cara kerja MaxPooling2D dengan menghitung nilai terbesar disetiap patch dan feature map. Membuat jumlah fitur pooled yang sama, layer pooling beroperasi pada peta fitur. Pada teknik _MaxPolling_ ini saya memakai 2 hidden layer.Setelah ekstrasi fitur selesai data akan merata menjadi satu vektor ke lapisan padat dengan 512 unit _perseptron_.

Cara kerja flatten memberikan input ke model. Lapisan pertama dari model jaringan saraf harus memiliki bentuk dan input data yang sama.juga mengubah tensor multidimensi menjadi tensor 1 dimensi yang dapat menggunakan pipih.

Cara kerja fungsi aktvasi yaitu mengubah nilai-nilai menjadi sesuatu yang setara antara 0,1 atau -1,1 untuk membuat keseluruhan proses seimbang secara statistik.
Semua lapisan dataset ini menggunakan aktifasi _Relu_ untuk downsample data dan aktivasi _softmax_ digunakan pada lapisan output.

Bagian model fit, disini ada untuk memanggil data training dengan cara kerja _steps_ _per_ _epoch_ yaitu argumen untuk fungsi fit model berdasarkan ukuran batch yang dipilih dan terdapat 20 _steps_ _per_ _epoch_ nya. Dengan cara kerja Epoch untuk menyediakan fungsi dalam bekerja dengan datetime, serta untuk mengubah antar representasi dan terdapat 10 epochs.

#Evaluation
Saya memakai metrik evaluasi akurasi klasifikasi yang digunakan. Metrik Akurasi Ini adalah rasio jumlah prediksi yang benar dengan jumlah total prediksi yang dibuat untuk kumpulan gambar yang digunakan. Grafik plot yang tertampil merupakan overfit karena mungkin training dataset ini tidak bisa diprediksi dengan tepat.
![image](https://user-images.githubusercontent.com/55178060/194866973-c2cb83b3-6fe5-47db-a83b-0e0f5320ded3.png)

Hasil Akurasi dari proyek saya dengan persentase acuracy 0.57% dan val_accuracy 0.47%, gambar yang digunakan mungkin tidak seimbang sehingga tidak cukup baik dalam mencapai akurasi goodfit. 
Gambar output
![image](https://user-images.githubusercontent.com/55178060/194867216-ed1dd8a4-d258-4bf0-9342-ff9385098196.png) menunjukkan sudah sesuai dengan gambar yang di maksud, gambar dengan nomor 107 yaitu daun rasberry berada pada bagian nontoxic berarti tumbuhan ini tidak beracun dan buahnya bisa dikonsumsikan.
