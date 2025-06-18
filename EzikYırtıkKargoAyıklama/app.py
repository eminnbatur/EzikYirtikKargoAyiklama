import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime
import logging
import tensorflow as tf

# GPU bellek yönetimi
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.warning(f"GPU ayarları yapılandırılamadı: {e}")

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.secret_key = 'kargo_hasar_tespiti_2024'  # Flash mesajları için gerekli
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Upload klasörünü oluştur
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Model yükleme
try:
    model = load_model('package_damage_model.h5', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Model başarıyla yüklendi.")
except Exception as e:
    logging.error(f"Model yüklenirken hata oluştu: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Yüklenen dosyayı benzersiz bir isimle kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename, filepath

def process_image(filepath):
    """Görüntüyü işle ve model için hazırla"""
    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logging.error(f"Görüntü işlenirken hata: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        # Model kontrolü
        if model is None:
            flash('Model yüklenemedi. Lütfen sistem yöneticisi ile iletişime geçin.', 'error')
            return render_template('index.html', prediction=None)

        # Dosya kontrolü
        if 'file' not in request.files:
            flash('Dosya yüklenmedi.', 'error')
            return render_template('index.html', prediction=None)
        
        file = request.files['file']
        if file.filename == '':
            flash('Dosya seçilmedi.', 'error')
            return render_template('index.html', prediction=None)

        # Dosya tipi kontrolü
        if not allowed_file(file.filename):
            flash('Geçersiz dosya formatı. Lütfen PNG, JPG veya JPEG formatında bir dosya yükleyin.', 'error')
            return render_template('index.html', prediction=None)

        try:
            # Dosyayı kaydet
            filename, filepath = save_uploaded_file(file)
            logging.info(f"Dosya kaydedildi: {filename}")

            # Görüntüyü işle
            img_array = process_image(filepath)

            # Tahmin yap
            with tf.device('/CPU:0'):  # CPU'da tahmin yap
                prediction = model.predict(img_array, verbose=0)[0][0]
            
            # Sonucu belirle
            if prediction > 0.5:
                label = "Hasarlı"
                confidence = prediction
                severity = "Yüksek" if confidence > 0.8 else "Orta"
            else:
                label = "Hasarsız"
                confidence = 1 - prediction
                severity = "Düşük"

            prediction_result = {
                'filename': filename,
                'label': label,
                'confidence': round(confidence * 100, 2),
                'severity': severity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logging.info(f"Tahmin sonucu: {prediction_result}")

        except Exception as e:
            logging.error(f"Tahmin sırasında hata: {str(e)}")
            flash(f'Bir hata oluştu: {str(e)}', 'error')
            return render_template('index.html', prediction=None)

    return render_template('index.html', prediction=prediction_result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('Dosya boyutu çok büyük. Maksimum 16MB yükleyebilirsiniz.', 'error')
    return render_template('index.html', prediction=None), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('Sunucu hatası oluştu. Lütfen daha sonra tekrar deneyin.', 'error')
    return render_template('index.html', prediction=None), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
