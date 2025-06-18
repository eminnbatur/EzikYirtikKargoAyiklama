import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def download_and_prepare_dataset():
    # Create dataset directory
    os.makedirs('dataset', exist_ok=True)
    
    print("""
    Lütfen şu adımları izleyin:
    
    1. https://www.mvtec.com/company/research/datasets/mvtec-ad adresine gidin
    2. 'Download' butonuna tıklayın
    3. Formu doldurun (ücretsiz kayıt)
    4. İndirme linkini alın
    5. İndirilen ZIP dosyasını bu klasöre çıkartın
    6. Çıkartılan dosyaları 'dataset' klasörüne taşıyın
    
    Veri seti hazır olduğunda Enter tuşuna basın...
    """)
    input()
    
    # Klasör yapısını oluştur
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/test', exist_ok=True)
    
    # Kategorileri oluştur
    for category in ['damaged', 'undamaged']:
        os.makedirs(f'dataset/train/{category}', exist_ok=True)
        os.makedirs(f'dataset/test/{category}', exist_ok=True)
    
    print("Veri seti başarıyla hazırlandı!")

if __name__ == '__main__':
    download_and_prepare_dataset() 