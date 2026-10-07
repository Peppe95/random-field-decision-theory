from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'production_code.zip'
with zipfile.ZipFile(ARCHIVE) as z:
    z.extractall(HERE)
print(f'Extracted {ARCHIVE.name} under {HERE}')
