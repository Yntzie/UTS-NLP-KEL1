# Kelp1 Project - Dataset Explorer ABSA

Folder ini berisi output proyek Kelompok 1.

## Cara menjalankan Streamlit
```bash
pip install -r requirements.txt
streamlit run streamlit/Kelp1_app.py
```

## Catatan data
- File anotasi asli berisi 2.745 baris anotasi dari 3 annotator.
- Dataset unik berisi 915 review.
- Kolom `business_name` dan `rating` dibuat kosong karena tidak tersedia di file anotasi yang diunggah.
- Kategori diisi `Kuliner` berdasarkan isi review yang dominan membahas makanan/tempat makan.
- IRR dari dosen disalin ke `dataset/Kelp1_irr_ner.json` dan `dataset/Kelp1_irr_textcat.json` untuk ditampilkan di aplikasi.