import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import datetime

# NumPy uyumluluğu için uyarı
print(f'Kullanılan NumPy sürümü: {np.__version__}')
if not np.__version__.startswith('1.24'):
    print('UYARI: NumPy 1.24.x sürümü önerilir!')

# Veri yollarını kontrol et
train_dir = 'dataset/train'
test_dir = 'dataset/test'

total_train = 0
total_test = 0
print("Veri seti kontrol ediliyor...")
for split in ['train', 'test']:
    print(f"\n{split.upper()} veri seti:")
    for category in ['damaged', 'undamaged']:
        path = os.path.join('dataset', split, category)
        if os.path.exists(path):
            num_files = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{category}: {num_files} görüntü")
            if split == 'train':
                total_train += num_files
            else:
                total_test += num_files
        else:
            print(f"HATA: {path} klasörü bulunamadı!")
print(f"\nToplam eğitim verisi: {total_train}")
print(f"Toplam test verisi: {total_test}")

# Veri artırma ve normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,  # Dikey çevirme kapalı
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleyicileri
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("\nSınıf eşleştirmeleri:")
for class_name, class_index in train_generator.class_indices.items():
    print(f"{class_name}: {class_index}")

# Model oluşturma
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Model derleme
model.compile(
    optimizer=Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\nModel eğitimi başlıyor...")
# Model eğitimi
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
)

# Test sonuçları
print("\nTest sonuçları hesaplanıyor...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")

# Sınıflandırma raporu
print("\nSınıflandırma raporu oluşturuluyor...")
preds = model.predict(test_generator)
y_true = test_generator.classes
y_pred = (preds > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Model kaydetme
model.save('package_damage_model.h5')
print("\nModel başarıyla kaydedildi: package_damage_model.h5")
