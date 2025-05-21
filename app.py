import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load models
gun_model = YOLO('best_m_gun.pt')
uniform_model = YOLO('best_m_uni.pt')

# Page settings
st.set_page_config(
    page_title="Gun & Militant Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom dark-themed styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0f0f0f;
            color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
        }
        .css-1v0mbdj, .css-1dp5vir {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #6a0dad;
            border: none;
            border-radius: 4px;
        }
        .stAlert {
            background-color: #222 !important;
            color: #fff !important;
        }
        .uploadedFile {
            color: #aaa;
        }
        .threat-level {
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 15px;
        }
        .threat-low {
            background-color: #2e7d32;
        }
        .threat-medium {
            background-color: #ff9800;
            color: #000;
        }
        .threat-high {
            background-color: #d32f2f;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .detection-guide {
            background-color: #1a1a1a;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .guide-title {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 SafeScope")
st.markdown("Detects firearms and flags non-uniformed individuals.")

st.divider()

# Sidebar for file upload and detection guide
st.sidebar.markdown("## Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

# Add Detection Guide in sidebar
with st.sidebar.expander("📋 Detection Guide", expanded=False):
    st.markdown("""
    ### Understanding Detections
    
    - **Gun Detection**: Identifies firearms in the image
    - **Uniform Detection**: Identifies individuals in official uniforms
    - **Militant Detection**: Non-uniformed individuals with firearms
    
    ### Threat Levels
    
    - **Low** (Green): No firearms or all firearms with uniformed personnel
    - **Medium** (Yellow): Firearms detected but not clearly associated with people
    - **High** (Red): Firearms detected with non-uniformed individuals
    
    ### Recommended Actions
    
    - **Low**: Normal monitoring
    - **Medium**: Increased vigilance and verification
    - **High**: Immediate response protocol and authority notification
    """)

# Helper functions
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_boxes(image, detections, label, color):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def display_threat_level(level):
    if level == "LOW":
        st.markdown(f'<div class="threat-level threat-low">THREAT LEVEL: {level}</div>', unsafe_allow_html=True)
    elif level == "MEDIUM":
        st.markdown(f'<div class="threat-level threat-medium">THREAT LEVEL: {level}</div>', unsafe_allow_html=True)
    elif level == "HIGH":
        st.markdown(f'<div class="threat-level threat-high">THREAT LEVEL: {level}</div>', unsafe_allow_html=True)

def display_action_guide(level):
    if level == "LOW":
        guide_text = """
        - Continue normal monitoring
        - No immediate action required
        - Document detection for records
        """
    elif level == "MEDIUM":
        guide_text = """
        - Increase monitoring frequency
        - Verify identities if possible
        - Prepare for escalation if situation changes
        - Consider informing security personnel
        """
    elif level == "HIGH":
        guide_text = """
        - Implement immediate response protocol
        - Alert proper authorities
        - Evacuate civilians from the area if necessary
        - Maintain visual on subjects if safe to do so
        - Document all observations
        """
    
    st.markdown('<div class="detection-guide">', unsafe_allow_html=True)
    st.markdown('<div class="guide-title">🚨 Recommended Actions:</div>', unsafe_allow_html=True)
    st.markdown(guide_text)
    st.markdown('</div>', unsafe_allow_html=True)

# Main logic
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    with st.spinner("🔎 Analyzing image..."):
        gun_results = gun_model(img)[0]
        gun_dets = [box.cpu().numpy() for box in gun_results.boxes.data if int(box[5]) == 0]

        threat_level = "LOW"
        action_guide = ""

        if not gun_dets:
            st.success("✅ No firearms detected.")
            display_threat_level(threat_level)
        else:
            uniform_results = uniform_model(img)[0]
            uniform_dets = [box.cpu().numpy() for box in uniform_results.boxes.data if int(box[5]) == 0]
            normal_dets = [box.cpu().numpy() for box in uniform_results.boxes.data if int(box[5]) == 1]

            alert_flag = False
            overlapped_normals = []
            overlapped_uniforms = []
            
            # Check guns with normal people
            for gun in gun_dets:
                for person in normal_dets:
                    # Use the same expanded bounding box approach for consistency
                    gun_x1, gun_y1, gun_x2, gun_y2 = map(int, gun[:4])
                    person_x1, person_y1, person_x2, person_y2 = map(int, person[:4])
                    
                    # Check if gun is near or overlapping with non-uniform person
                    gun_center_x = (gun_x1 + gun_x2) / 2
                    gun_center_y = (gun_y1 + gun_y2) / 2
                    
                    # Check if gun center is within or near person box with expanded boundaries
                    expand_factor = 0.2  # Expand person box by 20%
                    width = person_x2 - person_x1
                    height = person_y2 - person_y1
                    expanded_x1 = person_x1 - width * expand_factor
                    expanded_y1 = person_y1 - height * expand_factor
                    expanded_x2 = person_x2 + width * expand_factor
                    expanded_y2 = person_y2 + height * expand_factor
                    
                    if (expanded_x1 <= gun_center_x <= expanded_x2 and 
                        expanded_y1 <= gun_center_y <= expanded_y2):
                        alert_flag = True
                        overlapped_normals.append(person)
            
            # Check guns with uniformed personnel
            for gun in gun_dets:
                for person in uniform_dets:
                    # Use a more relaxed IOU threshold for uniform detection
                    # Also check if the gun is contained within the uniform bounding box
                    gun_x1, gun_y1, gun_x2, gun_y2 = map(int, gun[:4])
                    person_x1, person_y1, person_x2, person_y2 = map(int, person[:4])
                    
                    # Check if gun is near or overlapping with uniform person (relaxed spatial relationship)
                    gun_center_x = (gun_x1 + gun_x2) / 2
                    gun_center_y = (gun_y1 + gun_y2) / 2
                    
                    # Check if gun center is within or near person box with expanded boundaries
                    expand_factor = 0.2  # Expand person box by 20%
                    width = person_x2 - person_x1
                    height = person_y2 - person_y1
                    expanded_x1 = person_x1 - width * expand_factor
                    expanded_y1 = person_y1 - height * expand_factor
                    expanded_x2 = person_x2 + width * expand_factor
                    expanded_y2 = person_y2 + height * expand_factor
                    
                    if (expanded_x1 <= gun_center_x <= expanded_x2 and 
                        expanded_y1 <= gun_center_y <= expanded_y2):
                        overlapped_uniforms.append(person)

            img_annotated = img.copy()
            draw_boxes(img_annotated, gun_dets, label='Gun', color=(0, 0, 255))
            
            # Determine threat level
            if alert_flag:
                threat_level = "HIGH"
                draw_boxes(img_annotated, overlapped_normals, label='Militant', color=(0, 255, 255))
                st.error("🚨 ALERT: Gun detected with a non-uniformed person.")
            elif len(overlapped_uniforms) > 0:
                threat_level = "LOW"
                draw_boxes(img_annotated, uniform_dets, label='Uniform', color=(0, 255, 0))
                st.success("✅ All firearms are associated with uniformed personnel.")
            else:
                threat_level = "MEDIUM"
                draw_boxes(img_annotated, uniform_dets, label='Uniform', color=(0, 255, 0))
                draw_boxes(img_annotated, normal_dets, label='Person', color=(255, 0, 255))
                st.warning("⚠️ Firearms detected but not clearly associated with any person.")

            display_threat_level(threat_level)
            display_action_guide(threat_level)

            # Calculate detection statistics
            gun_count = len(gun_dets)
            uniform_count = len(uniform_dets)
            civilian_count = len(normal_dets)
            
            # Display detection metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Guns Detected", gun_count)
            with col2:
                st.metric("Uniformed Personnel", uniform_count)
            with col3:
                st.metric("Civilians", civilian_count)

            st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB),
                     caption="🖼️ Detection Output",
                     use_container_width=True)
            
            # Add detection details expander
            with st.expander("📊 Detection Details", expanded=False):
                st.markdown("### Object Confidences")
                
                if gun_dets:
                    st.markdown("#### 🔫 Guns")
                    for i, det in enumerate(gun_dets):
                        st.progress(float(det[4]))
                        st.text(f"Gun #{i+1}: Confidence {det[4]:.2f}")
                
                if uniform_dets:
                    st.markdown("#### 👮 Uniformed Personnel")
                    for i, det in enumerate(uniform_dets):
                        st.progress(float(det[4]))
                        st.text(f"Uniform #{i+1}: Confidence {det[4]:.2f}")
                
                if normal_dets:
                    st.markdown("#### 👤 Civilians")
                    for i, det in enumerate(normal_dets):
                        st.progress(float(det[4]))
                        st.text(f"Civilian #{i+1}: Confidence {det[4]:.2f}")
          
else:
    st.info("⬅️ Please upload an image from the sidebar to begin.")
    
    # Show sample threat levels when no image is uploaded
    st.markdown("### Sample Threat Levels")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="threat-level threat-low">THREAT LEVEL: LOW</div>', unsafe_allow_html=True)
        st.markdown("No firearms or all firearms with uniformed personnel")
    with col2:
        st.markdown('<div class="threat-level threat-medium">THREAT LEVEL: MEDIUM</div>', unsafe_allow_html=True)
        st.markdown("Firearms detected but not clearly associated with people")
    with col3:
        st.markdown('<div class="threat-level threat-high">THREAT LEVEL: HIGH</div>', unsafe_allow_html=True)
        st.markdown("Firearms detected with non-uniformed individuals")
