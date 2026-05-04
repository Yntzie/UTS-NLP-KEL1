# Kelp1 - Dataset Explorer ABSA

Proyek ini disesuaikan dengan dataset terbaru dari kelompok 1.

## Sumber data utama

- `dataset/Kelp1_dataset_1.csv`: dataset mentah/kotor hasil scraping.
- `dataset/Kelp1_dataset_2.csv`: dataset hasil preprocessing.
- `dataset/db_nlp1_genap2526.jsonl`: data anotasi lengkap dari dosen/Prodigy.
- `dataset/Kelp1_dataset_anotasi.jsonl`: salinan data anotasi lengkap dengan nama file wajib.
- `dataset/Kelp1_dataset_anotasi_aggregated.jsonl`: data final hasil agregasi voting mayoritas untuk EDA dan Streamlit.
- `dataset/nlp1_iaaa_textcat.jsonl`: IRR klasifikasi teks/multi-label.
- `dataset/nlp1_iaaa.jsonl`: IRR NER/span.

## Menjalankan Streamlit

```bash
pip install -r requirements.txt
streamlit run streamlit/Kelp1_app.py
```
