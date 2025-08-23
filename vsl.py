import gradio as gr
import cv2
import numpy as np
import joblib
import os
import time
from collections import deque, Counter
import threading
from skimage.feature import hog, local_binary_pattern
import warnings
from PIL import Image
from datetime import datetime
import json
warnings.filterwarnings('ignore')

class VSLGradioPredictor:
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
        self.model = None
        self.imputer = None
        self.scaler = None
        self.pca = None
        self.config = None
        
        # Real-time settings
        self.prediction_history = deque(maxlen=10)
        self.stable_prediction = None
        self.stable_confidence = 0
        self.prediction_threshold = 0.6
        self.stability_threshold = 0.7
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_prediction_time = time.time()
        self.total_predictions = 0
        self.start_time = time.time()
        
        # Status tracking
        self.is_processing = False
        self.last_status = "Waiting..."
        
        # Config classes (từ notebook của bạn)
        self.LETTER_CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
        self.NUMBER_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.ALL_CLASSES = self.LETTER_CLASSES + self.NUMBER_CLASSES
        self.TARGET_SIZE = (96, 96)
        
        # Load model
        self.model_loaded = self.load_model()
        
    def load_model(self):
        """Load trained model và các components"""
        try:
            model_files = {
                'model': 'vsl_model.pkl',
                'imputer': 'vsl_imputer.pkl', 
                'scaler': 'vsl_scaler.pkl',
                'pca': 'vsl_pca.pkl',
                'config': 'vsl_config.pkl'
            }
            
            print(f"🔄 Loading model từ {self.model_dir}...")
            
            for name, filename in model_files.items():
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, name, joblib.load(filepath))
                    print(f"   ✅ {name} loaded")
                else:
                    print(f"   ❌ {filename} not found")
                    return False
            
            print(f"🎯 Model accuracy: {self.config['test_accuracy']:.1%}")
            print(f"📊 Có {len(self.config['all_classes'])} classes")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return False
    
    def detect_hand(self, image):
        """Detect hand region (từ notebook)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        try:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

                contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    total_area = gray.shape[0] * gray.shape[1]

                    if area > total_area * 0.05:
                        x, y, w, h = cv2.boundingRect(largest)
                        padding = 0.15

                        x1 = max(0, int(x - w * padding))
                        y1 = max(0, int(y - h * padding))
                        x2 = min(gray.shape[1], int(x + w * (1 + padding)))
                        y2 = min(gray.shape[0], int(y + h * (1 + padding)))

                        return {
                            'x_center': (x1 + x2) / (2 * gray.shape[1]),
                            'y_center': (y1 + y2) / (2 * gray.shape[0]),
                            'width': (x2 - x1) / gray.shape[1],
                            'height': (y2 - y1) / gray.shape[0]
                        }, True, (x1, y1, x2, y2)
        except:
            pass

        return {'x_center': 0.5, 'y_center': 0.5, 'width': 0.7, 'height': 0.7}, False, None

    def enhance_image(self, image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return enhanced
        except:
            return gray if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def crop_image(self, image, bbox, target_size):
        try:
            h, w = image.shape[:2]
            x_center = bbox['x_center'] * w
            y_center = bbox['y_center'] * h
            width = bbox['width'] * w
            height = bbox['height'] * h

            x1 = max(0, int(x_center - width/2))
            y1 = max(0, int(y_center - height/2))
            x2 = min(w, int(x_center + width/2))
            y2 = min(h, int(y_center + height/2))

            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
                return self.enhance_image(resized)
            else:
                return self.enhance_image(cv2.resize(image, target_size))
        except:
            return self.enhance_image(cv2.resize(image, target_size))

    def extract_hog_features(self, image):
        try:
            features = hog(
                image,
                orientations=8,
                pixels_per_cell=(12, 12),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                feature_vector=True
            )
            return np.nan_to_num(features, 0.0)
        except:
            return np.zeros(50)

    def extract_lbp_features(self, image):
        try:
            lbp = local_binary_pattern(image, 16, 2, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=18)
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            return np.nan_to_num(hist, 0.0)
        except:
            return np.zeros(18)

    def extract_statistical_features(self, image):
        try:
            flat = image.ravel().astype(float)
            features = [
                np.mean(flat),
                np.std(flat),
                np.median(flat),
                np.min(flat),
                np.max(flat)
            ]
            return np.nan_to_num(features, 0.0)
        except:
            return np.zeros(5)

    def extract_features(self, image):
        if image is None or image.size == 0:
            return np.zeros(80)

        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hog_feat = self.extract_hog_features(image)
            lbp_feat = self.extract_lbp_features(image)
            stat_feat = self.extract_statistical_features(image)

            all_features = np.concatenate([hog_feat, lbp_feat, stat_feat])
            return np.nan_to_num(all_features, 0.0)
        except:
            return np.zeros(80)

    def calculate_fps(self):
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) >= 2:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0] + 1e-6)
            return fps
        return 0

    def predict_frame(self, frame):
        if not self.model_loaded:
            return None
        
        try:
            # Mark processing start
            self.is_processing = True
            start_time = time.time()
            
            # Convert từ RGB sang BGR nếu cần (Gradio dùng RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Detect hand và crop
            bbox, hand_detected, hand_coords = self.detect_hand(frame_bgr)
            cropped = self.crop_image(frame_bgr, bbox, self.TARGET_SIZE)
            
            # Extract features
            features = self.extract_features(cropped).reshape(1, -1)
            
            # Apply preprocessing pipeline
            features = self.imputer.transform(features)
            features = self.scaler.transform(features)
            features = self.pca.transform(features)
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            top_3 = [(self.config['all_classes'][i], probabilities[i]) for i in top_3_idx]
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.total_predictions += 1
            self.last_prediction_time = time.time()
            fps = self.calculate_fps()
            
            self.is_processing = False
            
            return {
                'predicted_letter': self.config['all_classes'][prediction],
                'confidence': confidence,
                'top_3': top_3,
                'hand_detected': hand_detected,
                'hand_coords': hand_coords,
                'bbox': bbox,
                'processing_time': processing_time,
                'fps': fps,
                'total_predictions': self.total_predictions
            }
            
        except Exception as e:
            self.is_processing = False
            print(f"Prediction error: {e}")
            return None

    def draw_prediction_overlay(self, frame, prediction_result):
        """Draw enhanced prediction overlay"""
        if frame is None:
            return frame
        
        # Convert sang BGR để vẽ
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display_frame = frame.copy()
        
        height, width = display_frame.shape[:2]
        
        # Draw processing indicator
        if self.is_processing:
            # Pulsing circle khi đang process
            center = (width - 30, 30)
            radius = 15
            cv2.circle(display_frame, center, radius, (0, 255, 255), -1)
            cv2.putText(display_frame, "AI", (center[0] - 8, center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            # Static circle khi không process
            center = (width - 30, 30)
            radius = 12
            cv2.circle(display_frame, center, radius, (0, 128, 255), 2)
        
        # Draw hand detection box
        if prediction_result and prediction_result['hand_detected'] and prediction_result['hand_coords']:
            x1, y1, x2, y2 = prediction_result['hand_coords']
            
            # Confidence-based color
            conf = prediction_result['confidence']
            if conf > 0.8:
                color = (0, 255, 0)  # Xanh lá
            elif conf > 0.6:
                color = (0, 255, 255)  # Vàng
            else:
                color = (0, 165, 255)  # Cam
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1-2, y1-2), (x2+2, y2+2), color, 3)
            
            # Hand status
            text = f"Hand: {conf:.1%}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1-25), (x1 + text_w + 10, y1), (0, 0, 0), -1)
            cv2.putText(display_frame, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Main prediction display
        if prediction_result:
            letter = prediction_result['predicted_letter']
            conf = prediction_result['confidence']
            
            # Large prediction display
            main_text = f"{letter}"
            conf_text = f"{conf:.1%}"
            
            # Background
            (text_w, text_h), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
            cv2.rectangle(display_frame, (10, 10), (text_w + 80, text_h + 60), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 10), (text_w + 80, text_h + 60), (0, 255, 255), 3)
            
            # Main letter
            cv2.putText(display_frame, main_text, (25, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
            
            # Confidence
            cv2.putText(display_frame, conf_text, (25, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Performance info
            if 'fps' in prediction_result and 'processing_time' in prediction_result:
                fps = prediction_result['fps']
                proc_time = prediction_result['processing_time']
                total = prediction_result['total_predictions']
                
                perf_text = f"FPS: {fps:.1f} | Time: {proc_time*1000:.0f}ms | Total: {total}"
                cv2.putText(display_frame, perf_text, (10, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Top 3 predictions
            y_offset = 100
            cv2.putText(display_frame, "Top 3:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            for i, (l, p) in enumerate(prediction_result['top_3'][:3]):
                y_offset += 30
                
                # Text
                text = f"{i+1}. {l}: {p:.1%}"
                cv2.putText(display_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Progress bar
                bar_width = int(200 * p)
                cv2.rectangle(display_frame, (150, y_offset - 15), (350, y_offset - 5), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (150, y_offset - 15), (150 + bar_width, y_offset - 5), 
                             (0, 255, 255) if i == 0 else (0, 165, 255), -1)
        
        # Live status indicator
        status_color = (0, 255, 0) if prediction_result else (0, 0, 255)
        status_text = "LIVE" if prediction_result else "WAIT"
        cv2.putText(display_frame, status_text, (width - 80, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Convert lại RGB cho Gradio
        if len(display_frame.shape) == 3:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        return display_frame

# Khởi tạo predictor global
predictor = VSLGradioPredictor('./models')

def predict_uploaded_image(image):
    """Predict từ uploaded image"""
    if image is None:
        return None, "❌ Không có ảnh được upload"
    
    if not predictor.model_loaded:
        return None, "❌ Model chưa được load. Kiểm tra thư mục models."
    
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Predict
        result = predictor.predict_frame(image)
        
        if result:
            # Draw prediction lên ảnh
            annotated_image = predictor.draw_prediction_overlay(image, result)
            
            # Tạo text kết quả
            result_text = f"""
                🎯 **Dự đoán: {result['predicted_letter']}**
                📊 **Độ tin cậy: {result['confidence']:.1%}**
                🔍 **Phát hiện tay: {'✅ Có' if result['hand_detected'] else '⚠️ Không'}**
                ⚡ **Thời gian xử lý: {result['processing_time']*1000:.0f}ms**

                📈 **Top 3 dự đoán:**
                """
            for i, (letter, prob) in enumerate(result['top_3'], 1):
                char_type = "Letter" if letter in predictor.LETTER_CLASSES else "Number"
                result_text += f"\n{i}. {letter} ({char_type}): {prob:.1%}"
            
            return annotated_image, result_text
        else:
            return image, "❌ Không thể xử lý ảnh"
            
    except Exception as e:
        return image, f"❌ Lỗi: {str(e)}"

def predict_webcam_frame(image):
    """Predict từ webcam frame"""
    if image is None:
        return None, "❌ Không có frame từ webcam"
    
    if not predictor.model_loaded:
        return None, "❌ Model chưa được load"
    
    try:
        # Predict
        result = predictor.predict_frame(image)
        
        if result:
            # Update prediction history
            if result['confidence'] >= predictor.prediction_threshold:
                predictor.prediction_history.append(result['predicted_letter'])
            
            # Check stable prediction
            stable_text = ""
            if len(predictor.prediction_history) >= 5:
                counter = Counter(predictor.prediction_history)
                most_common = counter.most_common(1)[0]
                letter, count = most_common
                stability_ratio = count / len(predictor.prediction_history)
                
                if stability_ratio >= predictor.stability_threshold:
                    stable_text = f"\n🔒 **Dự đoán ổn định: {letter}** (Độ ổn định: {stability_ratio:.1%})"
            
            # Draw prediction overlay
            annotated_frame = predictor.draw_prediction_overlay(image, result)
            
            # Tạo result text
            current_time = datetime.now().strftime("%H:%M:%S")
            result_text = f"""
                🔴 **LIVE PREDICTION** - {current_time}

                🎯 **Hiện tại: {result['predicted_letter']}** (Tin cậy: {result['confidence']:.1%})
                🔍 **Tay: {'✅ Detected' if result['hand_detected'] else '⚠️ Not detected'}**
                ⚡ **FPS: {result['fps']:.1f}** | **Thời gian: {result['processing_time']*1000:.0f}ms**
                📊 **Total predictions: {result['total_predictions']}**{stable_text}

                📈 **Top 3:**
                """
            for i, (letter, prob) in enumerate(result['top_3'], 1):
                char_type = "Letter" if letter in predictor.LETTER_CLASSES else "Number"
                result_text += f"\n{i}. {letter} ({char_type}): {prob:.1%}"
            
            # Add history
            if len(predictor.prediction_history) > 0:
                history_str = " → ".join(list(predictor.prediction_history)[-5:])
                result_text += f"\n\n📝 **Lịch sử gần đây:** {history_str}"
            
            return annotated_frame, result_text
        else:
            return image, f"❌ Không thể xử lý frame - {datetime.now().strftime('%H:%M:%S')}"
            
    except Exception as e:
        return image, f"❌ Lỗi: {str(e)} - {datetime.now().strftime('%H:%M:%S')}"

def reset_prediction_history():
    predictor.prediction_history.clear()
    predictor.total_predictions = 0
    predictor.start_time = time.time()
    predictor.fps_counter.clear()
    return "🔄 Đã reset lịch sử dự đoán và thống kê"

def get_model_info():
    """Lấy thông tin model"""
    if predictor.model_loaded:
        uptime = time.time() - predictor.start_time
        uptime_str = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
        
        return f"""
            ✅ **Model đã load thành công!**

            📊 **Thông tin model:**
            - Độ chính xác test: {predictor.config['test_accuracy']:.1%}
            - Độ chính xác validation: {predictor.config.get('val_accuracy', 0):.1%}
            - Model type: {predictor.config.get('model_name', 'SVM')}
            - Số classes: {len(predictor.config['all_classes'])}

            🔤 **Các ký tự có thể nhận diện:**
            **Chữ cái:** {', '.join(predictor.LETTER_CLASSES)}
            **Số:** {', '.join(predictor.NUMBER_CLASSES)}

            🎯 **Thiết lập:**
            - Ngưỡng tin cậy: {predictor.prediction_threshold:.1%}
            - Ngưỡng ổn định: {predictor.stability_threshold:.1%}
            - Lịch sử tối đa: {predictor.prediction_history.maxlen} frames

            📈 **Thống kê real-time:**
            - Uptime: {uptime_str}
            - Total predictions: {predictor.total_predictions}
            - Average FPS: {len(predictor.fps_counter) / (uptime + 1e-6):.1f}
            """
    else:
        return """
            ❌ **Model chưa được load!**

            🔧 **Cách khắc phục:**
            1. Kiểm tra thư mục `./models` có chứa các file:
            - vsl_model.pkl
            - vsl_imputer.pkl
            - vsl_scaler.pkl
            - vsl_pca.pkl
            - vsl_config.pkl

            2. Đảm bảo path trong code đúng với thư mục chứa model

            3. Restart app sau khi sửa path
            """

# ===== GRADIO INTERFACE =====
def create_gradio_app():
    """Tạo Gradio interface"""
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        border: none;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .gr-button:hover {
        background: linear-gradient(45deg, #45a049, #4CAF50);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .live-indicator {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    """
    
    with gr.Blocks(css=css, title="🎯 VSL Real-time Prediction", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.Markdown("""
        # 🎯 Vietnamese Sign Language (VSL) Real-time Recognition
        ### Hệ thống nhận diện ký hiệu tay Việt Nam theo thời gian thực
        
        **Features:**
        - 🔴 **Real-time Prediction**: Nhận diện trực tiếp từ webcam
        - 📁 **Upload Image**: Test với ảnh có sẵn
        - ⚡ **Performance Tracking**: FPS và thời gian xử lý
        - 🎯 **Enhanced Visualization**: Hiển thị chi tiết với bounding box
        """)
        
        # Model status
        with gr.Row():
            model_status = gr.Markdown(get_model_info())
            refresh_btn = gr.Button("🔄 Refresh Model Info", variant="secondary")
            refresh_btn.click(get_model_info, outputs=model_status)
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: Real-time Webcam
            with gr.TabItem("📷 Webcam Real-time", id="webcam"):
                gr.Markdown("""
                ### 🔴 Live Prediction từ webcam
                **Hướng dẫn:**
                1. Cho phép truy cập camera khi được yêu cầu
                2. Đưa tay vào giữa khung hình
                3. Tạo các ký hiệu VSL để nhận diện real-time
                4. Quan sát visual indicators và performance metrics
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        webcam_input = gr.Image(
                            sources=["webcam"], 
                            streaming=True,
                            label="📹 Webcam Feed",
                            format="numpy"
                        )
                        
                        with gr.Row():
                            reset_btn = gr.Button("🔄 Reset History & Stats", variant="secondary")
                            reset_status = gr.Textbox(label="Status", interactive=False)
                            reset_btn.click(reset_prediction_history, outputs=reset_status)
                    
                    with gr.Column(scale=1):
                        webcam_output = gr.Image(label="🎯 Real-time Prediction Result")
                        webcam_text = gr.Markdown("👋 Đưa tay vào trước camera để bắt đầu nhận diện...")
                
                # Streaming setup
                webcam_input.stream(
                    predict_webcam_frame,
                    inputs=webcam_input,
                    outputs=[webcam_output, webcam_text],
                    stream_every=0.5,  # 0.5 giây interval
                    show_progress=False
                )
            
            # Tab 2: Upload Image
            with gr.TabItem("🖼️ Upload Image", id="upload"):
                gr.Markdown("### Test với ảnh upload")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_input = gr.Image(
                            type="pil",
                            label="📁 Upload ảnh VSL"
                        )
                        predict_btn = gr.Button("🎯 Predict", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        upload_output = gr.Image(label="🎯 Prediction Result")
                        upload_text = gr.Markdown("📝 Kết quả sẽ hiển thị ở đây...")
                
                # Event handlers
                predict_btn.click(
                    predict_uploaded_image,
                    inputs=upload_input,
                    outputs=[upload_output, upload_text]
                )
                
                upload_input.change(
                    predict_uploaded_image,
                    inputs=upload_input,
                    outputs=[upload_output, upload_text]
                )
            
            # Tab 3: Model Information
            with gr.TabItem("📊 Model Info & Performance", id="info"):
                gr.Markdown("### Thông tin chi tiết về model và hiệu suất")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        #### 🎯 Các ký tự được hỗ trợ
                        """)
                        
                        if predictor.model_loaded:
                            gr.Markdown("**Chữ cái:** " + ", ".join([f"**{c}**" for c in predictor.LETTER_CLASSES]))
                            gr.Markdown("**Số:** " + ", ".join([f"**{c}**" for c in predictor.NUMBER_CLASSES]))
                        else:
                            gr.Markdown("❌ *Model chưa được load*")
                    
                    with gr.Column():
                        gr.Markdown("""
                        #### ⚙️ Real-time Settings
                        """)
                        
                        settings_text = f"""
                        - **Ngưỡng tin cậy**: {predictor.prediction_threshold:.1%}
                        - **Ngưỡng ổn định**: {predictor.stability_threshold:.1%}
                        - **Lịch sử tối đa**: {predictor.prediction_history.maxlen} frames
                        - **Stream interval**: 0.5 seconds
                        - **Target size**: {predictor.TARGET_SIZE}
                        - **Feature extraction**: HOG + LBP + Statistical
                        """
                        gr.Markdown(settings_text)
                
                gr.Markdown("""
                #### 💡 Tips để có kết quả tốt nhất:
                1. **Ánh sáng tốt**: Đảm bảo ánh sáng đủ và đều
                2. **Background đơn giản**: Nền không quá phức tạp
                3. **Tay rõ ràng**: Đưa tay vào giữa khung hình
                4. **Giữ ổn định**: Giữ tay ổn định vài giây để có prediction ổn định
                5. **Khoảng cách phù hợp**: Cách camera 30-50cm
                6. **Theo dõi indicators**: Quan sát các visual indicators trên video
                
                #### 🔴 Visual Indicators giải thích:
                - **LIVE/WAIT**: Trạng thái system
                - **AI Circle**: Pulsing khi đang xử lý
                - **Hand Box**: Màu thay đổi theo confidence level (Xanh = cao, Vàng = trung bình, Cam = thấp)
                - **Progress Bars**: Top 3 predictions với tỷ lệ tin cậy
                - **FPS Counter**: Tốc độ xử lý real-time
                """)
        
        # Footer
        gr.Markdown("""
        ---
        ### 🚀 VSL Real-time Recognition System
        **Technology Stack**: OpenCV, Scikit-learn, HOG+LBP Features, Gradio
        
        **Model**: Support Vector Machine với 32 classes (22 letters + 10 numbers)
        
        **Performance**: Real-time prediction với visual feedback và performance monitoring
        """)
    
    return app

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Tạo và chạy app
    app = create_gradio_app()
    
    # Launch app
    app.launch(
        server_name="0.0.0.0",      # Cho phép truy cập từ mạng
        server_port=7860,           # Port mặc định
        share=True,                # Set True nếu muốn public link
        debug=True,                 # Debug mode
        show_error=True,            # Hiển thị lỗi
        inbrowser=True,             # Tự mở browser
        max_threads=10              # Số threads xử lý
    )