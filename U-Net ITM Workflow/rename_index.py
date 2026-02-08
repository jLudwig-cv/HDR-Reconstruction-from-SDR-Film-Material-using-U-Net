import os
import re

def rename_files_with_index(directory, start_index=3):
    # Muster für Dateinamen mit 4-stelliger Indexierung %04d
    pattern = re.compile(r"^(.*?)(\d{4})(\.\w+)$")
    
    # Alle Dateien im Verzeichnis auflisten und filtern
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Nur Dateien berücksichtigen, die zum Muster passen und eine 4-stellige Zahl enthalten
    matched_files = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            prefix, index_str, suffix = match.groups()
            index = int(index_str)
            matched_files.append((filename, prefix, index, suffix))
    
    # Sortieren: nach Index absteigend, um Konflikte zu vermeiden
    matched_files.sort(key=lambda x: x[2], reverse=True)
    
    for filename, prefix, index, suffix in matched_files:
        # Nur Dateien umbenennen, deren Index >= start_index ist
        if index >= start_index:
            new_index_str = f"{index + 1:04d}"
            new_filename = f"{prefix}{new_index_str}{suffix}"
            
            # Pfade definieren
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # Datei umbenennen
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")


# Beispielaufruf der Funktion
rename_files_with_index(r"C:\Users\rippthehorror\Desktop\Datensatz_digital_Morbius\Wall-E_Mobius", start_index=32)
