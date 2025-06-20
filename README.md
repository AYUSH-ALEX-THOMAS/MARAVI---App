# MARAVI: Cognitive Risk Detection through Handwriting Analysis

**MARAVI** is a stylus-based web application built for mobile and tablet devices to assess cognitive health through handwriting and drawing patterns. It captures fine-grained motion data and applies machine learning to predict early cognitive risk — particularly for conditions like Alzheimer's and Parkinson’s.

---

## 🔧 Features

- Interactive canvas with **15 custom handwriting tasks**
- Real-time capture of stylus movement data
- Extraction of 255+ features per test
- **XGBoost-based ML model** for risk scoring
- Visual feedback with color-coded risk bands
- User authentication and session management
- Dashboard with historical score tracking
- Lightweight, extensible, and research-friendly

---

## 🧠 Handwriting Tasks

All tasks are designed to run on a canvas and are optimized for stylus input. Examples include:

- Signature drawing  
- Drawing lines and circles  
- Copying letters and words  
- Writing in reverse  
- Memory-based word recall  
- Drawing shapes (e.g., star, clock, house)

---

## 📊 Feature Extraction

Each of the 15 tasks yields 17 distinct features:

- **Temporal**: `air_time`, `paper_time`, `total_time`  
- **Spatial**: `dispersion_index`, `max_x_extension`, `max_y_extension`  
- **Kinematic**: `mean_speed`, `mean_acceleration`, `mean_jerk` (in air and on paper)  
- **Geometric**: `gmrt`, `gmrt_in_air`, `gmrt_on_paper`  
- **Pen Events**: `num_of_pendown`  

**Total features per test:** 15 × 17 = **255**

---

## 🧪 Machine Learning Pipeline

- **Model**: `XGBoostClassifier`
- **Input**: 255 extracted features  
- **Output**: Risk percentage (0–100%)  
- **Preprocessing**: `StandardScaler`  
- **Training Data**: CSV with 255 features + class label (`H` or `P`)  
- **Inference**: Scaled feature vector passed to model for prediction  

---

## 🧭 User Flow

1. User signs up or logs in
2. Completes 15 canvas-based handwriting tasks
3. Features are extracted and fed to the ML model
4. A risk percentage and band (Low / Moderate / High) are displayed
5. Result is stored and visualized on the dashboard

---

## 🏗️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript (Jinja templates)
- **Backend**: Python (Flask)
- **Database**: Supabase (PostgreSQL)
- **ML**: XGBoost, scikit-learn
- **Storage**: Supabase + optional local CSV logs

---

📌 Future Improvements
Improve mobile/tablet UX with responsive design

Add more cognitive games (e.g., Stroop Test, Memory Matrix)

Enhance real-user data support for model training

Admin dashboard for researchers and clinicians

Multilingual support and exportable reports

📄 License
This project is licensed under the MIT License.

⚠️ Disclaimer
MARAVI is a research and experimental tool. It is not a certified medical diagnostic tool. Use results for educational or preliminary screening purposes only. Always consult with a medical professional for clinical diagnosis.



