import numpy as np
import os

file_path = "MultiMatrix.npz" # Schimbă calea dacă e în alt folder

if not os.path.exists(file_path):
    print(f"[EROARE] Nu am găsit fișierul: {file_path}")
else:
    data = np.load(file_path, allow_pickle=True)
    print("=" * 60)
    print(f"   CONȚINUT FIȘIER CALIBRARE: {file_path}")
    print("=" * 60)
    
    for cheie in data.files:
        valoare = data[cheie]
        print(f"🔑 Cheie: {cheie}")
        print(f"   Shape: {valoare.shape if hasattr(valoare, 'shape') else 'N/A'}")
        print(f"   Tip date: {valoare.dtype if hasattr(valoare, 'dtype') else type(valoare)}")
        
        # Afișăm frumos matricea camerei și coeficienții
        if cheie in ["camMatrix", "distCoef"]:
            print("   Conținut:\n", valoare)
        elif cheie in ["pattern", "boardDimensions", "squareSize"]:
            print(f"   Valoare: {valoare}")
        else:
            print("   [Array de puncte sau vectori - ascuns pentru lizibilitate]")
        print("-" * 60)
