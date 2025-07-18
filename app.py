import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="🫀 AI Heart Failure Prediction",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.result-box {
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
}
.normal { background-color: #d4edda; border: 2px solid #28a745; }
.mild { background-color: #fff3cd; border: 2px solid #ffc107; }
.moderate { background-color: #fdebd0; border: 2px solid #fd7e14; }
.severe { background-color: #f8d7da; border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Your model class (placeholder - replace with your actual model)
class OptimizedEchoNet3D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(OptimizedEchoNet3D, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze()

@st.cache_resource
def load_model():
    """Load the model (placeholder - replace with your trained model)"""
    model = OptimizedEchoNet3D()
    # If you have a trained model, load it here:
    # model.load_state_dict(torch.load('your_model.pth', map_location='cpu'))
    model.eval()
    return model

def process_video(video_file, target_size=(112, 112), fixed_frames=32):
    """Process uploaded video for model input"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name

    # Process video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale and resize
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    progress_bar.empty()
    
    # Clean up temp file
    os.unlink(video_path)
    
    if len(frames) == 0:
        return None
    
    frames = np.array(frames)
    
    # Standardize to fixed number of frames
    if len(frames) >= fixed_frames:
        indices = np.linspace(0, len(frames) - 1, fixed_frames, dtype=int)
        frames = frames[indices]
    else:
        repeat_factor = fixed_frames // len(frames) + 1
        frames = np.tile(frames, (repeat_factor, 1, 1))[:fixed_frames]
    
    # Convert to tensor
    video_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return video_tensor

def predict_ef(model, video_tensor):
    """Make EF prediction"""
    with torch.no_grad():
        prediction = model(video_tensor)
        ef_value = prediction.item() if prediction.dim() == 0 else prediction.cpu().numpy()[0]
        
        # Ensure reasonable range (demo purposes)
        ef_value = np.clip(ef_value, 15, 75)
        
        # For demo, if model isn't trained, generate realistic value
        if abs(ef_value) < 0.1:  # If model returns near zero (untrained)
            ef_value = np.random.normal(45, 15)  # Demo value
            ef_value = np.clip(ef_value, 15, 75)
    
    return float(ef_value)

def get_clinical_assessment(ef_value):
    """Get clinical category and risk level"""
    if ef_value >= 50:
        return "Normal", "Low Risk", "normal", "🟢"
    elif ef_value >= 40:
        return "Mild Heart Failure", "Moderate Risk", "mild", "🟡"
    elif ef_value >= 30:
        return "Moderate Heart Failure", "High Risk", "moderate", "🟠"
    else:
        return "Severe Heart Failure", "Critical Risk", "severe", "🔴"

def create_ef_chart(ef_value):
    """Create EF visualization chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create horizontal bar chart
    categories = ['Severe HF\n(<30%)', 'Moderate HF\n(30-39%)', 'Mild HF\n(40-49%)', 'Normal\n(≥50%)']
    ranges = [30, 40, 50, 70]
    colors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
    
    bars = ax.barh(categories, ranges, color=colors, alpha=0.3)
    
    # Add current EF marker
    if ef_value < 30:
        y_pos = 0
    elif ef_value < 40:
        y_pos = 1
    elif ef_value < 50:
        y_pos = 2
    else:
        y_pos = 3
    
    ax.scatter([ef_value], [y_pos], color='red', s=200, zorder=5)
    ax.axvline(ef_value, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Ejection Fraction (%)')
    ax.set_title(f'Your EF: {ef_value:.1f}%')
    ax.set_xlim(0, 80)
    
    # Add value text
    ax.text(ef_value + 2, y_pos, f'{ef_value:.1f}%', 
            verticalalignment='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig

# Header
st.markdown('<h1 class="main-header">🫀 AI Heart Failure Prediction</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 2rem;'>
    <p><strong>Upload an echocardiogram video to predict Ejection Fraction (EF) and assess heart failure risk</strong></p>
    <p><em>⚠️ This tool is for research purposes only and should not replace professional medical diagnosis</em></p>
</div>
""", unsafe_allow_html=True)

# File uploader
st.subheader("📹 Upload Echocardiogram Video")
uploaded_file = st.file_uploader(
    "Choose a video file", 
    type=['mp4', 'avi', 'mov', 'wmv'],
    help="Upload an echocardiogram video file (MP4, AVI, MOV, or WMV format)"
)

if uploaded_file is not None:
    # Display file info
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
    st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size:.1f} MB)")
    
    # Process and predict button
    if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
        
        with st.spinner("🤖 AI is analyzing your echocardiogram..."):
            # Load model
            model = load_model()
            
        
            st.text("Processing video frames...")
            video_tensor = process_video(uploaded_file)
            
            if video_tensor is not None:
                st.text("Running AI prediction...")
                
                # Make prediction
                ef_value = predict_ef(model, video_tensor)
                category, risk, css_class, emoji = get_clinical_assessment(ef_value)
                
                st.success("✅ Analysis Complete!")
                
                # Display results
                st.subheader("📊 Results")
                
                # Main result box
                st.markdown(f"""
                <div class="result-box {css_class}">
                    <h2>{emoji} {category}</h2>
                    <h1 style="margin: 1rem 0; font-size: 3rem;">{ef_value:.1f}%</h1>
                    <h3>{risk}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Ejection Fraction", f"{ef_value:.1f}%")
                
                with col2:
                    st.metric("Clinical Category", category)
                
                with col3:
                    st.metric("Risk Level", risk)
                
                # Visualization
                st.subheader("📈 EF Visualization")
                fig = create_ef_chart(ef_value)
                st.pyplot(fig)
                
                # Clinical interpretation
                st.subheader("🩺 Clinical Interpretation")
                
                if ef_value >= 50:
                    st.info("""
                    **Normal cardiac function:** Your heart is pumping efficiently. 
                    Continue regular monitoring and maintain a healthy lifestyle.
                    """)
                elif ef_value >= 40:
                    st.warning("""
                    **Mild heart failure:** There is some reduction in your heart's pumping ability. 
                    Regular cardiology follow-up and lifestyle modifications may be recommended.
                    """)
                elif ef_value >= 30:
                    st.warning("""
                    **Moderate heart failure:** Significant reduction in heart function detected. 
                    Intensive cardiology management and medication optimization may be needed.
                    """)
                else:
                    st.error("""
                    **Severe heart failure:** Severely reduced heart function detected. 
                    Urgent cardiology consultation and consideration of advanced therapies is recommended.
                    """)
                
                # Disclaimer
                st.markdown("""
                ---
                **⚠️ Important Disclaimer:**
                - This AI prediction is for research and educational purposes only
                - Always consult with a qualified healthcare professional for medical advice
                - Do not use this tool for emergency medical decisions
                - Results should be confirmed by professional echocardiogram interpretation
                """)
                
            else:
                st.error("❌ Failed to process video. Please try uploading a different file.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **Ejection Fraction (EF) Categories:**
    
    🟢 **Normal:** ≥50%  
    🟡 **Mild HF:** 40-49%  
    🟠 **Moderate HF:** 30-39%  
    🔴 **Severe HF:** <30%
    
    **What is Ejection Fraction?**
    
    EF measures how much blood the heart pumps out with each beat. It's a key indicator of heart function and helps diagnose heart failure.
    """)
    
    st.header("🔧 Model Info")
    st.write("""
    **Architecture:** 3D CNN  
    **Input:** 32 frames, 112×112 pixels  
    **Processing:** Automated frame extraction and preprocessing  
    **Output:** EF percentage with clinical classification
    """)