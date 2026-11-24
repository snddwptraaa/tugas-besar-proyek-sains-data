```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Konfigurasi tampilan tabel & grafik
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
```


```python
# 1. Load ketiga dataset hasil crawling
df_master = pd.read_csv('bandung_crime_master_events.csv')
df_jun_sep = pd.read_csv('kriminalitas_bandung_juni_september.csv')
df_all = pd.read_csv('kriminalitas_jalanan_kota_bandung_all.csv')

total_raw = len(df_master) + len(df_jun_sep) + len(df_all)
print(f"[INFO] Total baris gabungan mentah (3 file): {total_raw} baris")
print(f"  - bandung_crime_master_events.csv          : {len(df_master)} baris")
print(f"  - kriminalitas_bandung_juni_september.csv  : {len(df_jun_sep)} baris (100% subset identik di dalam file all)")
print(f"  - kriminalitas_jalanan_kota_bandung_all.csv: {len(df_all)} baris\n")

# 2. Definisi daftar index yang harus dihapus berdasarkan hasil audit Minggu 1
# A. Dari df_master (51 baris):
master_luar_bandung = [0, 17, 36, 45] # Polda Riau, Kos Jakbar, Taman Kopo Indah (Kab. Bdg), Tamansari Bogor
master_non_kriminal = [1, 2, 5, 6, 8, 9, 44, 46, 47, 48] # Laka lalin, longsor, sejarah, rekayasa lalin, imbauan
master_duplikat_event = [7, 15, 19, 20, 21, 23, 25, 28, 30, 31, 37, 38] # Kejadian sama persis dengan df_all / internal

# B. Dari df_all (101 baris):
all_duplikat_url_teknis = [36, 51, 74, 87, 89, 94] # Duplikat artikel akibat slug URL terpotong
all_luar_bandung = [14, 54, 72] # TKP Cimahi, Polsek Kemang Bogor, Polresta Blitar
all_non_kriminal_umum = [4, 9, 25, 30, 32, 41, 43, 52, 76, 82, 92, 93] # Opini sosiolog, rilis akumulatif, imbauan
all_duplikat_event = [0, 8, 24, 26, 27, 35, 38, 42, 46, 57, 63, 67, 70, 83, 85, 90, 95] # Multi-berita untuk 1 kejadian TKP

drop_master_idx = master_luar_bandung + master_non_kriminal + master_duplikat_event
drop_all_idx = all_duplikat_url_teknis + all_luar_bandung + all_non_kriminal_umum + all_duplikat_event

# Eksekusi pemfilteran baris
m_clean = df_master.drop(index=drop_master_idx).copy()
a_clean = df_all.drop(index=drop_all_idx).copy()

print(f"[HASIL FILTERING]")
print(f"  - Sisa insiden unik dari master_events : {len(m_clean)} baris")
print(f"  - Sisa insiden unik dari all_events    : {len(a_clean)} baris")
print(f"  - Total Insiden Kriminalitas Valid     : {len(m_clean) + len(a_clean)} insiden")
```

    [INFO] Total baris gabungan mentah (3 file): 212 baris
      - bandung_crime_master_events.csv          : 51 baris
      - kriminalitas_bandung_juni_september.csv  : 60 baris (100% subset identik di dalam file all)
      - kriminalitas_jalanan_kota_bandung_all.csv: 101 baris
    
    [HASIL FILTERING]
      - Sisa insiden unik dari master_events : 25 baris
      - Sisa insiden unik dari all_events    : 63 baris
      - Total Insiden Kriminalitas Valid     : 88 insiden
    


```python
# Referensi Koordinat Centroid 30 Kecamatan Resmi Kota Bandung
KECAMATAN_COORDS = {
    'Andir': (-6.9147, 107.5794), 'Antapani': (-6.9178, 107.6628),
    'Arcamanik': (-6.9231, 107.6833), 'Astanaanyar': (-6.9317, 107.5992),
    'Babakan Ciparay': (-6.9458, 107.5794), 'Bandung Kidul': (-6.9569, 107.6306),
    'Bandung Kulon': (-6.9261, 107.5653), 'Bandung Wetan': (-6.9044, 107.6186),
    'Batununggal': (-6.9325, 107.6386), 'Bojongloa Kaler': (-6.9333, 107.5878),
    'Bojongloa Kidul': (-6.9497, 107.5967), 'Buahbatu': (-6.9567, 107.6533),
    'Cibeunying Kaler': (-6.8953, 107.6322), 'Cibeunying Kidul': (-6.9036, 107.6436),
    'Cibiru': (-6.9239, 107.7228), 'Cicendo': (-6.9042, 107.5958),
    'Cidadap': (-6.8681, 107.6033), 'Cinambo': (-6.9367, 107.6978),
    'Coblong': (-6.8856, 107.6133), 'Gedebage': (-6.9583, 107.6917),
    'Kiaracondong': (-6.9297, 107.6483), 'Lengkong': (-6.9347, 107.6178),
    'Mandalajati': (-6.9019, 107.6744), 'Panyileukan': (-6.9339, 107.7089),
    'Rancasari': (-6.9511, 107.6711), 'Regol': (-6.9417, 107.6083),
    'Sukajadi': (-6.8889, 107.5933), 'Sukasari': (-6.8611, 107.5917),
    'Sumur Bandung': (-6.9167, 107.6111), 'Ujungberung': (-6.9083, 107.7028),
    'Kota Bandung (Umum)': (-6.9175, 107.6191)
}

# Bobot Keparahan Kejahatan (Skala 1 - 5)
SEVERITY_WEIGHTS = {
    'Pembunuhan': 5,
    'Begal & Curas': 4,
    'Penganiayaan Berat / Sajam': 4,
    'Tawuran & Geng Motor': 3,
    'Curanmor': 3,
    'Penjambretan (Jambret)': 2,
    'Pencurian (Curat) & Lainnya': 2
}

# Fungsi Harmonisasi Kategori Kejahatan
def harmonize_category(raw_cat, judul, desc=""):
    text = f"{raw_cat} {judul} {desc}".lower()
    if 'pembunuhan' in raw_cat.lower() or 'meninggal dunia' in text or 'guru sd tilil' in text or 'tewas ditikam' in text:
        return 'Pembunuhan'
    if 'jambret' in text:
        return 'Penjambretan (Jambret)'
    if 'tawuran' in raw_cat.lower() or 'geng motor' in raw_cat.lower():
        return 'Tawuran & Geng Motor'
    if 'curanmor' in raw_cat.lower() or 'digondol maling' in text:
        return 'Curanmor'
    if 'keributan bersenjata tajam' in text or 'pembacokan' in raw_cat.lower() or 'penganiayaan' in raw_cat.lower() or 'lukai 16 polisi' in text or 'pengeroyok' in text:
        return 'Penganiayaan Berat / Sajam'
    if 'pemerasan' in text or 'curat' in raw_cat.lower() or 'premanisme' in raw_cat.lower() or 'tramadol' in text or 'sabu' in text:
        return 'Pencurian (Curat) & Lainnya'
    if 'begal' in raw_cat.lower() or 'curas' in raw_cat.lower():
        return 'Begal & Curas'
    return 'Pencurian (Curat) & Lainnya'

# Fungsi Normalisasi Kecamatan Berdasarkan Isi Teks & Koreksi Manual
def normalize_kecamatan(raw_loc, judul, desc=""):
    text = f"{judul} {desc}".lower()
    loc_clean = str(raw_loc).replace(', Kota Bandung', '').strip()

    # Koreksi spesifik berdasarkan jalan/kelurahan di dalam deskripsi berita
    if 'sekelimus' in text or 'banceuy' in text: return 'Bandung Kidul'
    if 'gang luna' in text or 'kelurahan kopo' in text or 'jalan peta' in text or 'bizpark kopo' in text or 'bca kopo' in text: return 'Bojongloa Kaler'
    if 'jatisari' in text or 'cijaura' in text: return 'Buahbatu'
    if 'sd tilil' in text or 'sadangserang' in text or 'dago' in text or 'dipatiukur' in loc_clean.lower(): return 'Coblong'
    if 'cigadung' in text or 'sukaluyu' in text: return 'Cibeunying Kaler'
    if 'ciparungpung' in text or 'cibeunying kidul' in text: return 'Cibeunying Kidul'
    if 'riung bandung' in text or 'riung hegar' in text or 'cipamokolan' in text: return 'Rancasari'
    if 'jalan natuna' in text: return 'Sumur Bandung'
    if 'taman maluku' in text: return 'Bandung Wetan'
    if 'sukaati' in text: return 'Regol'
    if 'dr otten' in text or 'bundaran doggy' in text: return 'Cicendo'
    if 'pasar induk caringin' in text or 'pasar caringin' in text: return 'Babakan Ciparay'
    if 'setiabudi' in text: return 'Sukasari'
    if 'pasupati' in text: return 'Coblong'
    if 'jalan sudirman' in text: return 'Astanaanyar'
    if 'soekarno hatta' in loc_clean.lower(): return 'Buahbatu'

    if loc_clean in KECAMATAN_COORDS:
        return loc_clean
    return 'Kota Bandung (Umum)'

# Fungsi Ekstraksi Jam & Fase Waktu dari Kolom Waktu dan Deskripsi
def extract_time_and_phase(waktu_str, desc_str, existing_jam="", existing_fase=""):
    combined = f"{waktu_str} {desc_str}".lower()
    jam_match = re.search(r'(\d{1,2})[:\.](\d{2})\s*wib|pukul\s*(\d{1,2})[:\.](\d{2})', combined)

    jam_val = str(existing_jam) if pd.notna(existing_jam) and str(existing_jam).strip() != '' else ""
    if not jam_val and jam_match:
        h = jam_match.group(1) or jam_match.group(3)
        m = jam_match.group(2) or jam_match.group(4)
        jam_val = f"{int(h):02d}:{m}"

    # Tentukan fase waktu berdasarkan jam atau kata kunci
    if jam_val:
        hour = int(jam_val.split(':')[0])
        if 0 <= hour < 5: fase = 'Dini Hari (00:00 - 05:00)'
        elif 5 <= hour < 11: fase = 'Pagi (05:00 - 11:00)'
        elif 11 <= hour < 15: fase = 'Siang (11:00 - 15:00)'
        elif 15 <= hour < 18 or (hour == 18 and int(jam_val.split(':')[1]) < 30): fase = 'Sore (15:00 - 18:30)'
        else: fase = 'Malam (18:30 - 24:00)'
        return jam_val, fase

    if existing_fase and existing_fase != 'Tidak Disebutkan':
        return jam_val, existing_fase

    if 'dini hari' in combined or 'tengah malam' in combined: return "", 'Dini Hari (00:00 - 05:00)'
    if 'pagi' in combined: return "", 'Pagi (05:00 - 11:00)'
    if 'siang' in combined: return "", 'Siang (11:00 - 15:00)'
    if 'sore' in combined: return "", 'Sore (15:00 - 18:30)'
    if 'malam' in combined: return "", 'Malam (18:30 - 24:00)'

    return "", 'Tidak Disebutkan'

# Susun list baris bersih gabungan
records = []
np.random.seed(42) # Seed tetap untuk jitter koordinat mikro dalam kecamatan yang sama

# A. Proses dari m_clean
for idx, r in m_clean.iterrows():
    kat = harmonize_category(r['jenis_kejahatan'], r['judul_berita'])
    kec = normalize_kecamatan(r['kecamatan'], r['judul_berita'], r['detail_tkp_jalan'])
    jam, fase = extract_time_and_phase(r['tgl_terbit_berita'], r['judul_berita'], r['jam_kejadian'], r['fase_waktu'])

    # Gunakan koordinat asli jika spesifik, atau koordinat kecamatan yang dikoreksi
    lat, lon = r['latitude'], r['longitude']
    if r['kecamatan'] in ['Kota Bandung', 'Kota Bandung (Umum)'] and kec != 'Kota Bandung (Umum)':
        lat, lon = KECAMATAN_COORDS[kec]

    tkp = r['detail_tkp_jalan'] if r['detail_tkp_jalan'] not in ['-', 'Jalan Pakai Google'] else f"Kec. {kec}"
    if 'Lengkong Besar' in r['judul_berita']: tkp = 'Jl. Lengkong Besar'

    records.append({
        'source_id': f"M{idx:02d}",
        'judul_berita': r['judul_berita'],
        'kategori_kejahatan': kat,
        'bobot_keparahan': SEVERITY_WEIGHTS[kat],
        'kecamatan': kec,
        'detail_tkp': tkp,
        'jam_kejadian': jam if jam else '-',
        'fase_waktu': fase,
        'latitude': round(lat + np.random.uniform(-0.002, 0.002), 5),
        'longitude': round(lon + np.random.uniform(-0.002, 0.002), 5),
        'waktu_berita': r['tgl_terbit_berita'],
        'sumber': 'Master Portal',
        'url': r['url_sumber']
    })

# B. Proses dari a_clean
for idx, r in a_clean.iterrows():
    kat = harmonize_category(r['perkara'], r['judul'], r['deskripsi'])
    kec = normalize_kecamatan(r['lokasi'], r['judul'], r['deskripsi'])
    jam, fase = extract_time_and_phase(r['waktu'], r['deskripsi'])
    lat, lon = KECAMATAN_COORDS.get(kec, KECAMATAN_COORDS['Kota Bandung (Umum)'])

    # Ekstrak nama jalan/kawasan dari deskripsi bila ada
    jln_match = re.search(r'(Jalan\s+[A-Za-z0-9\.\-\s]+?)(?:,|Kelurahan|Kecamatan|Kota|pada|Sabtu|Minggu|Senin|Selasa|Rabu|Kamis|Jumat)', str(r['deskripsi']))
    tkp = jln_match.group(1).strip() if jln_match else f"Kawasan {kec}"

    records.append({
        'source_id': f"A{idx:03d}",
        'judul_berita': r['judul'],
        'kategori_kejahatan': kat,
        'bobot_keparahan': SEVERITY_WEIGHTS[kat],
        'kecamatan': kec,
        'detail_tkp': tkp[:45],
        'jam_kejadian': jam if jam else '-',
        'fase_waktu': fase,
        'latitude': round(lat + np.random.uniform(-0.003, 0.003), 5),
        'longitude': round(lon + np.random.uniform(-0.003, 0.003), 5),
        'waktu_berita': f"{r['waktu']} (Bulan: {r['bulan']})",
        'sumber': r['sumber'],
        'url': r['url']
    })

df_clean = pd.DataFrame(records)
df_clean.insert(0, 'event_id', [f"E{i+1:02d}" for i in range(len(df_clean))])

# Simpan dataset bersih hasil Minggu 1 ke CSV
df_clean.to_csv('master_kriminalitas_bandung_clean.csv', index=False)
print(f"[SUKSES] File 'master_kriminalitas_bandung_clean.csv' berhasil disimpan ({len(df_clean)} baris).")
display(df_clean.head(10))
```

    [SUKSES] File 'master_kriminalitas_bandung_clean.csv' berhasil disimpan (88 baris).
    



  <div id="df-322fa2f3-d4de-4a3e-9132-99c3638b4315" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_id</th>
      <th>source_id</th>
      <th>judul_berita</th>
      <th>kategori_kejahatan</th>
      <th>bobot_keparahan</th>
      <th>kecamatan</th>
      <th>detail_tkp</th>
      <th>jam_kejadian</th>
      <th>fase_waktu</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>waktu_berita</th>
      <th>sumber</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E01</td>
      <td>M03</td>
      <td>Akhir Bahagia Pedagang Sayur Ujungberung Usai ...</td>
      <td>Curanmor</td>
      <td>3</td>
      <td>Ujungberung</td>
      <td>Kec. Ujungberung</td>
      <td>20:30</td>
      <td>Malam (18:30 - 24:00)</td>
      <td>-6.90880</td>
      <td>107.70460</td>
      <td>Jumat, 24 Jul 2026 20:30 WIB</td>
      <td>Master Portal</td>
      <td>https://www.detik.com/jabar/hukum-dan-kriminal...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02</td>
      <td>M04</td>
      <td>Maling Motor di Cibiru Ajak Mantan Istri, Bera...</td>
      <td>Curanmor</td>
      <td>3</td>
      <td>Cibiru</td>
      <td>Kec. Cibiru</td>
      <td>08:36</td>
      <td>Pagi (05:00 - 11:00)</td>
      <td>-6.92297</td>
      <td>107.72319</td>
      <td>Minggu, 21 Jun 2026 08:36 WIB</td>
      <td>Master Portal</td>
      <td>https://www.detik.com/jabar/hukum-dan-kriminal...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E03</td>
      <td>M10</td>
      <td>Pelaku Curanmor Bersenpi Ditangkap Warga di Ge...</td>
      <td>Curanmor</td>
      <td>3</td>
      <td>Gedebage</td>
      <td>Kec. Gedebage</td>
      <td>-</td>
      <td>Tidak Disebutkan</td>
      <td>-6.95968</td>
      <td>107.69032</td>
      <td>Fri, 17 Apr 2026 07:00:00 GMT</td>
      <td>Master Portal</td>
      <td>https://news.google.com/rss/articles/CBMiuwFBV...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E04</td>
      <td>M11</td>
      <td>Heboh Maling Motor Tertangkap Basah di Arcaman...</td>
      <td>Curanmor</td>
      <td>3</td>
      <td>Arcamanik</td>
      <td>Kec. Arcamanik</td>
      <td>-</td>
      <td>Tidak Disebutkan</td>
      <td>-6.92487</td>
      <td>107.68476</td>
      <td>Thu, 07 May 2026 07:00:00 GMT</td>
      <td>Master Portal</td>
      <td>https://news.google.com/rss/articles/CBMiswFBV...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E05</td>
      <td>M12</td>
      <td>Viral! Aksi Pencurian Motor Ninja di Kota Band...</td>
      <td>Curanmor</td>
      <td>3</td>
      <td>Kota Bandung (Umum)</td>
      <td>Kec. Kota Bandung (Umum)</td>
      <td>-</td>
      <td>Tidak Disebutkan</td>
      <td>-6.91710</td>
      <td>107.61993</td>
      <td>Sat, 25 Oct 2025 07:00:00 GMT</td>
      <td>Master Portal</td>
      <td>https://news.google.com/rss/articles/CBMiqAFBV...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>E06</td>
      <td>M13</td>
      <td>Perampok SPBU Setiabudi Dibekuk</td>
      <td>Begal &amp; Curas</td>
      <td>4</td>
      <td>Sukasari</td>
      <td>Setiabudi</td>
      <td>18:33</td>
      <td>Malam (18:30 - 24:00)</td>
      <td>-6.86302</td>
      <td>107.59358</td>
      <td>Selasa, 21 Des 2010 18:33 WIB</td>
      <td>Master Portal</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>E07</td>
      <td>M14</td>
      <td>Sabu 8,97 Gram Diselundupkan di Ikat Rambut Pe...</td>
      <td>Pencurian (Curat) &amp; Lainnya</td>
      <td>2</td>
      <td>Bandung Kidul</td>
      <td>Banceuy</td>
      <td>-</td>
      <td>Tidak Disebutkan</td>
      <td>-6.95557</td>
      <td>107.62945</td>
      <td>6 jam yang lalu</td>
      <td>Master Portal</td>
      <td>https://www.detik.com/jabar/hukum-dan-kriminal...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>E08</td>
      <td>M16</td>
      <td>Serangan Massa Aksi Lukai 16 Polisi di Dago Ba...</td>
      <td>Penganiayaan Berat / Sajam</td>
      <td>4</td>
      <td>Coblong</td>
      <td>Dago</td>
      <td>09:00</td>
      <td>Pagi (05:00 - 11:00)</td>
      <td>-6.88687</td>
      <td>107.61203</td>
      <td>Kamis, 22 Okt 2026 09:00 WIB</td>
      <td>Master Portal</td>
      <td>https://www.detik.com/jabar/hukum-dan-kriminal...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>E09</td>
      <td>M18</td>
      <td>Aksi Curanmor di Bandung, Pelaku Kabur Usai To...</td>
      <td>Curanmor</td>
      <td>3</td>
      <td>Cibeunying Kaler</td>
      <td>Kec. Cibeunying Kaler</td>
      <td>11:37</td>
      <td>Siang (11:00 - 15:00)</td>
      <td>-6.89608</td>
      <td>107.63230</td>
      <td>Kamis, 08 Okt 2026 11:37 WIB</td>
      <td>Master Portal</td>
      <td>https://www.detik.com/jabar/hukum-dan-kriminal...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>E10</td>
      <td>M22</td>
      <td>Pria di Cinambo Bandung Tewas Dibacok di Kontr...</td>
      <td>Penganiayaan Berat / Sajam</td>
      <td>4</td>
      <td>Cinambo</td>
      <td>Kec. Cinambo</td>
      <td>-</td>
      <td>Tidak Disebutkan</td>
      <td>-6.93697</td>
      <td>107.69696</td>
      <td>Fri, 17 Apr 2026 07:00:00 GMT</td>
      <td>Master Portal</td>
      <td>https://news.google.com/rss/articles/CBMipwFBV...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-322fa2f3-d4de-4a3e-9132-99c3638b4315')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-322fa2f3-d4de-4a3e-9132-99c3638b4315 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-322fa2f3-d4de-4a3e-9132-99c3638b4315');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Plot Distribusi Kategori Kejahatan
cat_counts = df_clean['kategori_kejahatan'].value_counts()
sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=axes[0], palette='Reds_r')
axes[0].set_title('Distribusi Insiden Berdasarkan Kategori Kejahatan (n=88)', fontweight='bold')
axes[0].set_xlabel('Jumlah Insiden')
axes[0].set_ylabel('Kategori Kejahatan')
for i, v in enumerate(cat_counts.values):
    axes[0].text(v + 0.3, i, str(v), va='center', fontweight='bold')

# 2. Plot Distribusi Fase Waktu (Peak Hours)
order_fase = [
    'Dini Hari (00:00 - 05:00)',
    'Pagi (05:00 - 11:00)',
    'Siang (11:00 - 15:00)',
    'Sore (15:00 - 18:30)',
    'Malam (18:30 - 24:00)',
    'Tidak Disebutkan'
]
fase_counts = df_clean['fase_waktu'].value_counts().reindex(order_fase).fillna(0)
sns.barplot(x=fase_counts.values, y=fase_counts.index, ax=axes[1], palette='YlOrBr_r')
axes[1].set_title('Distribusi Temporal Kejadian Kriminalitas (Peak Hours)', fontweight='bold')
axes[1].set_xlabel('Jumlah Insiden')
axes[1].set_ylabel('Fase Waktu')
for i, v in enumerate(fase_counts.values):
    axes[1].text(v + 0.3, i, str(int(v)), va='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

    /tmp/ipykernel_993/698180457.py:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=axes[0], palette='Reds_r')
    /tmp/ipykernel_993/698180457.py:22: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x=fase_counts.values, y=fase_counts.index, ax=axes[1], palette='YlOrBr_r')
    


    
![png](Untitled29_files/Untitled29_3_1.png)
    



```python
# Konversi koordinat ke radian untuk metrik Haversine (jarak permukaan bumi dalam km)
coords = df_clean[['latitude', 'longitude']].values
kms_per_radian = 6371.0088

# Parameter DBSCAN: radius (eps) = 1.5 km, minimal titik per kluster (min_samples) = 3 insiden
epsilon = 1.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

df_clean['cluster_dbscan'] = db.labels_
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = list(db.labels_).count(-1)

print(f"[HASIL DBSCAN SPATIAL CLUSTERING]")
print(f"  - Radius Ketetanggaan (eps) : 1.5 KM")
print(f"  - Min Samples per Kluster   : 3 Insiden")
print(f"  - Jumlah Kluster Hotspot    : {n_clusters} Kluster")
print(f"  - Titik Terisolasi (Noise)  : {n_noise} Insiden\n")

# Visualisasi Peta Sebaran Titik & Kluster DBSCAN
plt.figure(figsize=(11, 8))
scatter = plt.scatter(
    df_clean['longitude'], df_clean['latitude'],
    c=df_clean['cluster_dbscan'], cmap='tab10',
    s=df_clean['bobot_keparahan'] * 35, alpha=0.85, edgecolors='black', linewidth=1
)

# Tambahkan label nama Kecamatan pada pusat-pusat kerawanan utama
top_kec = df_clean['kecamatan'].value_counts().head(10).index
for kec in top_kec:
    if kec in KECAMATAN_COORDS and kec != 'Kota Bandung (Umum)':
        lat_k, lon_k = KECAMATAN_COORDS[kec]
        plt.annotate(
            kec, (lon_k, lat_k),
            fontsize=9, fontweight='bold',
            xytext=(5, 5), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6)
        )

plt.title('Peta Sebaran Titik Kriminalitas & Kluster DBSCAN Kota Bandung\n(Ukuran Titik = Bobot Keparahan)', fontweight='bold', fontsize=13)
plt.xlabel('Longitude (Bujur Timur)')
plt.ylabel('Latitude (Lintang Selatan)')
plt.colorbar(scatter, label='ID Kluster DBSCAN (-1 = Outlier/Noise)')
plt.show()
```

    [HASIL DBSCAN SPATIAL CLUSTERING]
      - Radius Ketetanggaan (eps) : 1.5 KM
      - Min Samples per Kluster   : 3 Insiden
      - Jumlah Kluster Hotspot    : 13 Kluster
      - Titik Terisolasi (Noise)  : 8 Insiden
    
    


    
![png](Untitled29_files/Untitled29_4_1.png)
    



```python
# 1. Hitung Skor Kerawanan Spasial per Kecamatan
# Rumus: Skor_Prioritas = Sum(Bobot_Keparahan) + (1.5 * Jumlah_Insiden_Malam_&_Dini_Hari)
df_spatial = df_clean[df_clean['kecamatan'] != 'Kota Bandung (Umum)'].copy()
df_spatial['is_rawan_malam'] = df_spatial['fase_waktu'].isin([
    'Malam (18:30 - 24:00)', 'Dini Hari (00:00 - 05:00)'
]).astype(int)

ranking_kecamatan = df_spatial.groupby('kecamatan').agg(
    total_insiden=('event_id', 'count'),
    skor_keparahan=('bobot_keparahan', 'sum'),
    rata_rata_bobot=('bobot_keparahan', 'mean'),
    insiden_malam_dinihari=('is_rawan_malam', 'sum')
).reset_index()

ranking_kecamatan['indeks_prioritas_patroli'] = (
    ranking_kecamatan['skor_keparahan'] + (1.5 * ranking_kecamatan['insiden_malam_dinihari'])
).round(2)

# Klasifikasi Tier Kerawanan Wilayah
def tentukan_status_rawan(skor):
    if skor >= 22: return 'ZONA MERAH (Prioritas Utama)'
    elif skor >= 12: return 'ZONA ORANYE (Siaga Tinggi)'
    else: return 'ZONA KUNING (Waspada)'

ranking_kecamatan['status_zona'] = ranking_kecamatan['indeks_prioritas_patroli'].apply(tentukan_status_rawan)
ranking_kecamatan = ranking_kecamatan.sort_values(by='indeks_prioritas_patroli', ascending=False).reset_index(drop=True)

# Simpan hasil pemodelan Minggu 2 untuk dipakai Dashboard Minggu 3
ranking_kecamatan.to_csv('skor_prioritas_kecamatan_bandung.csv', index=False)
print("=== PERINGKAT PRIORITAS PATROLI KECAMATAN KOTA BANDUNG ===")
display(ranking_kecamatan)

# 2. Matriks Heatmap Interseksi Spasial-Temporal (Kecamatan vs Fase Waktu)
df_st = df_spatial[df_spatial['fase_waktu'] != 'Tidak Disebutkan']
pivot_st = pd.pivot_table(
    df_st,
    values='bobot_keparahan',
    index='kecamatan',
    columns='fase_waktu',
    aggfunc='sum',
    fill_value=0
)

# Urutkan kolom waktu secara kronologis dan baris berdasarkan total bobot
cols_waktu = [c for c in [
    'Dini Hari (00:00 - 05:00)', 'Pagi (05:00 - 11:00)',
    'Siang (11:00 - 15:00)', 'Sore (15:00 - 18:30)', 'Malam (18:30 - 24:00)'
] if c in pivot_st.columns]
pivot_st = pivot_st[cols_waktu]
pivot_st['Total_Skor'] = pivot_st.sum(axis=1)
pivot_st = pivot_st.sort_values(by='Total_Skor', ascending=False).drop(columns='Total_Skor')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_st, annot=True, fmt='g', cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Akumulasi Bobot Keparahan'})
plt.title('Matriks Interseksi Spasial-Temporal (Kecamatan vs Shift Waktu Patroli)', fontweight='bold', fontsize=13)
plt.xlabel('Fase Waktu (Shift Patroli)')
plt.ylabel('Kecamatan')
plt.tight_layout()
plt.show()
```

    === PERINGKAT PRIORITAS PATROLI KECAMATAN KOTA BANDUNG ===
    



  <div id="df-3e494bb2-0cae-408e-ac73-4bcfc925fce3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kecamatan</th>
      <th>total_insiden</th>
      <th>skor_keparahan</th>
      <th>rata_rata_bobot</th>
      <th>insiden_malam_dinihari</th>
      <th>indeks_prioritas_patroli</th>
      <th>status_zona</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Coblong</td>
      <td>9</td>
      <td>29</td>
      <td>3.222222</td>
      <td>3</td>
      <td>33.5</td>
      <td>ZONA MERAH (Prioritas Utama)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Babakan Ciparay</td>
      <td>8</td>
      <td>26</td>
      <td>3.250000</td>
      <td>3</td>
      <td>30.5</td>
      <td>ZONA MERAH (Prioritas Utama)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rancasari</td>
      <td>5</td>
      <td>19</td>
      <td>3.800000</td>
      <td>3</td>
      <td>23.5</td>
      <td>ZONA MERAH (Prioritas Utama)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gedebage</td>
      <td>5</td>
      <td>17</td>
      <td>3.400000</td>
      <td>2</td>
      <td>20.0</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Astanaanyar</td>
      <td>4</td>
      <td>15</td>
      <td>3.750000</td>
      <td>3</td>
      <td>19.5</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bojongloa Kaler</td>
      <td>5</td>
      <td>18</td>
      <td>3.600000</td>
      <td>0</td>
      <td>18.0</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cibeunying Kaler</td>
      <td>4</td>
      <td>14</td>
      <td>3.500000</td>
      <td>2</td>
      <td>17.0</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ujungberung</td>
      <td>4</td>
      <td>10</td>
      <td>2.500000</td>
      <td>3</td>
      <td>14.5</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Regol</td>
      <td>4</td>
      <td>11</td>
      <td>2.750000</td>
      <td>2</td>
      <td>14.0</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cibiru</td>
      <td>4</td>
      <td>12</td>
      <td>3.000000</td>
      <td>1</td>
      <td>13.5</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sukajadi</td>
      <td>3</td>
      <td>12</td>
      <td>4.000000</td>
      <td>1</td>
      <td>13.5</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cicendo</td>
      <td>4</td>
      <td>13</td>
      <td>3.250000</td>
      <td>0</td>
      <td>13.0</td>
      <td>ZONA ORANYE (Siaga Tinggi)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Buahbatu</td>
      <td>3</td>
      <td>10</td>
      <td>3.333333</td>
      <td>1</td>
      <td>11.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bojongloa Kidul</td>
      <td>2</td>
      <td>8</td>
      <td>4.000000</td>
      <td>2</td>
      <td>11.0</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Antapani</td>
      <td>2</td>
      <td>8</td>
      <td>4.000000</td>
      <td>2</td>
      <td>11.0</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Arcamanik</td>
      <td>2</td>
      <td>7</td>
      <td>3.500000</td>
      <td>1</td>
      <td>8.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Panyileukan</td>
      <td>2</td>
      <td>7</td>
      <td>3.500000</td>
      <td>1</td>
      <td>8.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Bandung Kidul</td>
      <td>2</td>
      <td>7</td>
      <td>3.500000</td>
      <td>1</td>
      <td>8.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Batununggal</td>
      <td>2</td>
      <td>7</td>
      <td>3.500000</td>
      <td>1</td>
      <td>8.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lengkong</td>
      <td>2</td>
      <td>6</td>
      <td>3.000000</td>
      <td>1</td>
      <td>7.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Cinambo</td>
      <td>2</td>
      <td>7</td>
      <td>3.500000</td>
      <td>0</td>
      <td>7.0</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Kiaracondong</td>
      <td>2</td>
      <td>5</td>
      <td>2.500000</td>
      <td>1</td>
      <td>6.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Sukasari</td>
      <td>1</td>
      <td>4</td>
      <td>4.000000</td>
      <td>1</td>
      <td>5.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Cibeunying Kidul</td>
      <td>1</td>
      <td>4</td>
      <td>4.000000</td>
      <td>0</td>
      <td>4.0</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Bandung Wetan</td>
      <td>1</td>
      <td>4</td>
      <td>4.000000</td>
      <td>0</td>
      <td>4.0</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Andir</td>
      <td>1</td>
      <td>2</td>
      <td>2.000000</td>
      <td>1</td>
      <td>3.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sumur Bandung</td>
      <td>1</td>
      <td>2</td>
      <td>2.000000</td>
      <td>1</td>
      <td>3.5</td>
      <td>ZONA KUNING (Waspada)</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3e494bb2-0cae-408e-ac73-4bcfc925fce3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3e494bb2-0cae-408e-ac73-4bcfc925fce3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3e494bb2-0cae-408e-ac73-4bcfc925fce3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_6a7a072b-80c5-477e-a830-0dcc0d46236e">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('ranking_kecamatan')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_6a7a072b-80c5-477e-a830-0dcc0d46236e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('ranking_kecamatan');
      }
      })();
    </script>
  </div>

    </div>
  </div>




    
![png](Untitled29_files/Untitled29_5_2.png)
    

