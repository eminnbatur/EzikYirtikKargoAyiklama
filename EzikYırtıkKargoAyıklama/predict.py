import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_preprocess_image(image_path):
    try:
        # Görüntüyü yükle ve boyutlandır
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        print(f"Hata: {image_path} yüklenirken bir sorun oluştu: {str(e)}")
        return None, None

def predict_damage(image_path, model):
    # Görüntüyü hazırla
    img_array, original_img = load_and_preprocess_image(image_path)
    if img_array is None:
        return None, None, None
    
    # Tahmin yap
    prediction = model.predict(img_array, verbose=0)
    probability = prediction[0][0]
    
    # Sonucu belirle
    result = "Hasarlı" if probability > 0.5 else "Hasarsız"
    confidence = probability if probability > 0.5 else 1 - probability
    
    return result, confidence, original_img

def visualize_prediction(image, result, confidence, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Sonuç: {result}\nGüven: %{confidence*100:.2f}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Modeli yükle
    try:
        model = tf.keras.models.load_model('package_damage_model.h5')
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Model yüklenirken bir sorun oluştu: {str(e)}")
        return

    # Test klasöründeki tüm görüntüleri kontrol et
    test_dir = 'dataset/test'
    results_dir = 'prediction_results'
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nTest görüntüleri kontrol ediliyor...")
    print("-" * 50)
    
    total_images = 0
    correct_predictions = 0
    
    for category in ['damaged', 'undamaged']:
        category_dir = os.path.join(test_dir, category)
        if os.path.exists(category_dir):
            print(f"\n{category.upper()} klasöründeki görüntüler:")
            for img_file in os.listdir(category_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    total_images += 1
                    img_path = os.path.join(category_dir, img_file)
                    print(f"\nGörüntü: {img_file}")
                    
                    result, confidence, image = predict_damage(img_path, model)
                    if result is not None:
                        # Sonucu görselleştir ve kaydet
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(results_dir, f"{timestamp}_{img_file}")
                        visualize_prediction(image, result, confidence, save_path)
                        
                        # Doğruluk kontrolü
                        expected_result = "Hasarlı" if category == "damaged" else "Hasarsız"
                        if result == expected_result:
                            correct_predictions += 1
                            print(f"✓ Doğru tahmin! (Beklenen: {expected_result})")
                        else:
                            print(f"✗ Yanlış tahmin! (Beklenen: {expected_result})")
                        print(f"Güven: %{confidence*100:.2f}")
    
    # Genel sonuçları göster
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print("\n" + "="*50)
        print(f"Toplam test görüntüsü: {total_images}")
        print(f"Doğru tahmin: {correct_predictions}")
        print(f"Yanlış tahmin: {total_images - correct_predictions}")
        print(f"Genel doğruluk: %{accuracy:.2f}")
        print("="*50)
        print(f"\nTahmin sonuçları '{results_dir}' klasörüne kaydedildi.")

if __name__ == '__main__':
    main() 