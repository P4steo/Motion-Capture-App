# 🎥 Motion Capture App

Aplikacja okienkowa do przechwytywania ruchu w czasie rzeczywistym z wykorzystaniem MediaPipe. Umożliwia podgląd szkieletu, zapis danych do pliku `.json` oraz eksport do `.bvh` zgodnego z Unreal Engine 5 / Blenderem.

## 📦 Wymagania

- Python 3.9+
- Biblioteki opisane w requirements.txt
- Kamera lub CamoStudio

## 🚀 Uruchamianie
W folderze z plikami uruchom:
```bash
py -m pip install -r requirements.txt
main.py
```


---

## 🖥️ Funkcje

### Główna zakładka:

- **Rozdzielczość** – wybór rozdzielczości kamery (`640x480`, `1280x720`, `1920x1080`)
- **Opóźnienie** – czas w sekundach przed rozpoczęciem nagrania (np. 3s na przygotowanie)
- **Mapowanie** – aktualnie: `Mediapipe`, przyszłościowo inne modele
- **Start/Stop** – rozpoczyna lub zatrzymuje przechwytywanie
- **Podgląd skeletonu** – wizualizuje zarejestrowany ruch w formie uproszczonego szkieletu 2D

### Zakładka „Eksport”:

- **Zapisz JSON** – zapisuje surowe dane z landmarkami do pliku `.json`
- **Eksportuj BVH** – eksportuje dane do pliku `.bvh`

### Zakładka „Pomoc”:

Instrukcje użycia eksportowanych plików `.bvh` w Blenderze:
> File → Import → Motion Capture (.bvh)

---

## 📁 Struktura danych

### `landmark_data` – lista klatek, każda zawiera:
```json
[
  { "x": 0.5, "y": 0.6, "z": -0.2 },
  ...
  { "x": 0.4, "y": 0.7, "z": -0.1 }
]
```

### `UE5_BONE_MAP` – mapowanie nazw kości do indeksów landmarków MediaPipe, zgodne z hierarchią Unreal Engine / Blender

---

## ⚙️ Planowane funkcje

- Eksport `.fbx`
- Mapowanie do niestandardowych rigów
- Detekcja i śledzenie wielu osób
- Wybór źródła wideo (kamera, plik wideo)

---

## 🐞 Znane problemy

- Brak wsparcia dla modeli 3D – tylko podgląd 2D
- Brak możliwości zatrzymania i wznowienia sesji

---
## 📷 Zrzuty z ekranu
<img width="998" height="749" alt="image" src="https://github.com/user-attachments/assets/2d8887cd-c321-4ee2-b485-6c512f73f6ee" />

---

## 📃 Licencja

MIT © 2025 P4steo
