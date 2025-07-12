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
- **Rotacje** - uproszczone 2D lub 3D Euler (w trakcie rozwoju)
- **Szybkie nagrywanie** - możliwość nagrania fragmentu video (`5s`, `10s`, `20s`)
- **Start/Stop** – rozpoczyna lub zatrzymuje przechwytywanie
- **Podgląd skeletonu** – wizualizuje zarejestrowany ruch w formie uproszczonego szkieletu 2D

### Zakładka „Eksport”:
**Wybór formatu eksportu**
- **Eksportuj JSON** – zapisuje surowe dane z landmarkami do pliku `.json`
- **Eksportuj BVH** – eksportuje dane do pliku `.bvh`
- **Eksportuj FBX** - wyświetla okno dialogowe z informacją, jak to zrobić w Blenderze `easteregg` XD

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

- Brak wsparcia dla liczenia rotacji za pomocą Eulera, liczne błędy
- Brak możliwości zatrzymania i wznowienia sesji
- Skalowalność

---
## 📷 Zrzuty z ekranu
<img width="1919" height="1026" alt="image" src="https://github.com/user-attachments/assets/b32bbecc-310a-405e-87c1-2d198b8dcfff" />

---
## 🧨 Wskazówki
- Ustaw opcję **Rotacje** na `Uproszczona 2D`
- Ustaw opóźnienie startu na wystarczająco długo aby:
  - Ustawić się swobodnie przed kamerą
  - Stanąć w T-Pose `XD`
- **T-Pose jest wymagany tylko w pierwszej klatce**
- Polecam korzystać z opcji **Szybkie nagrywanie**

---

## 📃 Licencja

MIT © 2025 P4steo
