import streamlit as st
import zipfile
import os
import tempfile
import shutil
from pathlib import Path
import base64
from PIL import Image, ImageEnhance
import io
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

# Optional imports with fallbacks
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI not available. Install with: pip install google-generativeai")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("MediaPipe not available. Install with: pip install mediapipe")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  

GEMINI_API_KEY = "AIzaSyAq8DtMZ1TEefrfYi871YS_OG8hZIhfa_o" 

class PhotoQuality(Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good" 
    ACCEPTABLE = "Acceptable"
    NEEDS_REVIEW = "Needs Review"

@dataclass
class PhotoAnalysis:
    filename: str
    quality: PhotoQuality
    confidence: float
    ai_reasoning: str
    faces_detected: int
    faces_with_open_eyes: int
    faces_smiling: int
    animals_detected: bool
    key_features: List[str]
    technical_metrics: Dict[str, float]
    thumbnail_path: str = ""
    user_override: Optional[PhotoQuality] = None
    processing_time: float = 0.0
    analysis_method: str = "Unknown"


class HybridPhotoAnalyzer:
    """Enhanced photo analyzer combining Gemini AI with computer vision"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        self.gemini_model = None
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
            except Exception as e:
                st.error(f"Failed to initialize Gemini: {e}")
        
        self.mp_face = None
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face = mp.solutions.face_mesh
                self.face_mesh = self.mp_face.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=15,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                
                self.left_eye_indices = [33, 160, 158, 133, 153, 144]
                self.right_eye_indices = [263, 387, 385, 362, 380, 373]
                self.mouth_indices = [61, 291, 13, 14]
                
                self.ear_threshold = 0.15  # Eye aspect ratio
                self.mar_threshold = 0.25  # Mouth aspect ratio
                
            except Exception as e:
                st.error(f"Failed to initialize MediaPipe: {e}")
        
        # Gemini analysis prompt
        self.gemini_prompt = """
        Analyze this photo and categorize it for sorting photos. Rate it as one of these categories:
        the concept here is to sort out the photos based on if the photo is good or bad ,
        by bad photos i mean the ones that get photobombed , blurriness, lack of focus, distracting elements, and poor technical execution,bad framing and poor lighting 
        if any of the photos include animals or pets check if the subject is focused or not as animals might move quickly , if the subject (animals) are moving fast 
        then let that photo goes under NEEDS_REVIEW category also give count on how many animals where detected from the total amount of photos

        🌟 EXCELLENT: Perfect shots - great quality, everyone looks good, sharp, well-lit, good composition, 
        👍 GOOD: Great photos with minor issues - maybe one person has eyes closed, slight blur, or minor lighting issues  
        👌 ACCEPTABLE: Decent photos that are usable - noticeable issues but still worth keeping
        ⚠️ NEEDS_REVIEW: Poor quality - very blurry, bad lighting, multiple issues, or unflattering

        Consider these factors:
        - Photo sharpness and focus
        - Lighting and exposure  
        - For people: Are eyes open? Are they smiling? Good expressions?
        - For animals: Are they moving/blurry? Good pose?
        - Overall composition and framing
        - Group dynamics (if multiple people)
        - Any technical issues or artifacts
        - Photo bomb or not 


        Respond in this EXACT JSON format:
        {
            "category": "EXCELLENT/GOOD/ACCEPTABLE/NEEDS_REVIEW",
            "confidence": 0.85,
            "reasoning": "Detailed explanation of why this category was chosen, mentioning specific observations",
            "faces_count": 2,
            "animals_detected": false,
            "key_features": ["sharp focus", "everyone smiling", "good lighting", "nice composition"],
            "technical_issues": ["slight motion blur", "overexposed sky"]
        }
        
        Be thorough in your reasoning and realistic in your assessment.
        """
    
    def analyze_photo(self, image_path: str, temp_dir: str) -> PhotoAnalysis:
        """Analyze a single photo using the hybrid method"""
        start_time = time.time()
        
        try:
            # Load image
            pil_image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            if pil_image is None or cv_image is None:
                return self._create_error_analysis(image_path, "Could not load image", time.time() - start_time)
            
            # Create thumbnail
            thumbnail_path = self._create_thumbnail(pil_image, image_path, temp_dir)
            
            # Run both analyses
            gemini_result = None
            cv_result = None
            
            if self.gemini_model:
                gemini_result = self._analyze_with_gemini(pil_image)
            if self.face_mesh:
                cv_result = self._analyze_with_cv(cv_image)
            
            # Combine results
            final_result = self._combine_analyses(gemini_result, cv_result, image_path, thumbnail_path)
            final_result.processing_time = time.time() - start_time
            
            return final_result
            
        except Exception as e:
            return self._create_error_analysis(image_path, f"Analysis error: {str(e)}", time.time() - start_time)
    
    def _analyze_with_gemini(self, image: Image.Image) -> Dict:
        """Analyze photo using Gemini AI"""
        try:
            # Resize image if too large
            image = self._resize_for_api(image)
            
            # Send to Gemini with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content([self.gemini_prompt, image])
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1 * (attempt + 1))
            
            # Parse response
            return self._parse_gemini_response(response.text)
            
        except Exception as e:
            return {"error": f"Gemini analysis failed: {str(e)}"}
    
    def _analyze_with_cv(self, image: np.ndarray) -> Dict:
        """Analyze photo using computer vision"""
        try:
            # Technical metrics
            sharpness = self._calculate_sharpness(image)
            brightness = self._calculate_brightness(image)
            contrast = self._calculate_contrast(image)
            
            # Face analysis
            faces_data = self._analyze_faces(image)
            
            # Quality assessment
            quality, confidence, issues = self._assess_cv_quality(sharpness, brightness, contrast, faces_data)
            
            return {
                "quality": quality,
                "confidence": confidence,
                "faces_count": len(faces_data),
                "faces_with_open_eyes": sum(1 for f in faces_data if f.get('eyes_open', False)),
                "faces_smiling": sum(1 for f in faces_data if f.get('smiling', False)),
                "technical_metrics": {
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "contrast": contrast
                },
                "issues": issues,
                "method": "Computer Vision"
            }
            
        except Exception as e:
            return {"error": f"CV analysis failed: {str(e)}"}
    
    def _combine_analyses(self, gemini_result: Dict, cv_result: Dict, 
                         image_path: str, thumbnail_path: str) -> PhotoAnalysis:
        """Combine Gemini and CV analyses into final result"""
        
        # Handle error cases
        if gemini_result and "error" in gemini_result and cv_result and "error" in cv_result:
            return self._create_error_analysis(image_path, "Both analyses failed", 0)
        
        # Create hybrid analysis combining both results
        return self._create_hybrid_analysis(gemini_result, cv_result, image_path, thumbnail_path)
    
    def _create_hybrid_analysis(self, gemini_result: Dict, cv_result: Dict, 
                               image_path: str, thumbnail_path: str) -> PhotoAnalysis:
        """Create hybrid analysis combining both results"""
        # Use Gemini for overall assessment and reasoning, CV for detailed metrics
        
        if gemini_result and "error" not in gemini_result:
            base_analysis = self._create_analysis_from_gemini(gemini_result, image_path, thumbnail_path)
        else:
            base_analysis = self._create_analysis_from_cv(cv_result, image_path, thumbnail_path)
        
        # Enhance with CV metrics if available
        if cv_result and "error" not in cv_result:
            base_analysis.faces_with_open_eyes = cv_result.get('faces_with_open_eyes', 0)
            base_analysis.faces_smiling = cv_result.get('faces_smiling', 0)
            base_analysis.technical_metrics = cv_result.get('technical_metrics', {})
            
            # Combine confidence scores (weighted average)
            if gemini_result and "error" not in gemini_result:
                gemini_conf = gemini_result.get('confidence', 0.5)
                cv_conf = cv_result.get('confidence', 0.5)
                base_analysis.confidence = (gemini_conf * 0.7 + cv_conf * 0.3)  # Weight Gemini higher
        
        base_analysis.analysis_method = "Hybrid (AI + CV)"
        return base_analysis
    
    def _create_analysis_from_gemini(self, result: Dict, image_path: str, thumbnail_path: str) -> PhotoAnalysis:
        """Create PhotoAnalysis from Gemini result"""
        if "error" in result:
            return self._create_error_analysis(image_path, result["error"], 0)
        
        return PhotoAnalysis(
            filename=Path(image_path).name,
            quality=result.get('quality', PhotoQuality.NEEDS_REVIEW),
            confidence=result.get('confidence', 0.5),
            ai_reasoning=result.get('reasoning', 'Gemini AI analysis completed'),
            faces_detected=result.get('faces_count', 0),
            faces_with_open_eyes=0,  # Gemini doesn't provide this detail
            faces_smiling=0,  # Gemini doesn't provide this detail
            animals_detected=result.get('animals_detected', False),
            key_features=result.get('key_features', []),
            technical_metrics={},
            thumbnail_path=thumbnail_path,
            analysis_method="Gemini AI"
        )
    
    def _create_analysis_from_cv(self, result: Dict, image_path: str, thumbnail_path: str) -> PhotoAnalysis:
        """Create PhotoAnalysis from CV result"""
        if "error" in result:
            return self._create_error_analysis(image_path, result["error"], 0)
        
        # Create reasoning from CV metrics
        reasoning_parts = []
        if result.get('technical_metrics'):
            metrics = result['technical_metrics']
            if metrics.get('sharpness', 0) > 50:
                reasoning_parts.append("image is sharp")
            elif metrics.get('sharpness', 0) < 15:
                reasoning_parts.append("image appears blurry")
            
            brightness = metrics.get('brightness', 0)
            if brightness > 200:
                reasoning_parts.append("image is bright")
            elif brightness < 50:
                reasoning_parts.append("image is dark")
        
        if result.get('faces_count', 0) > 0:
            reasoning_parts.append(f"detected {result['faces_count']} face(s)")
            if result.get('faces_with_open_eyes', 0) == result.get('faces_count', 0):
                reasoning_parts.append("all eyes are open")
            if result.get('faces_smiling', 0) > 0:
                reasoning_parts.append(f"{result['faces_smiling']} person(s) smiling")
        
        reasoning = "Computer vision analysis: " + ", ".join(reasoning_parts) if reasoning_parts else "Technical analysis completed"
        
        return PhotoAnalysis(
            filename=Path(image_path).name,
            quality=result.get('quality', PhotoQuality.NEEDS_REVIEW),
            confidence=result.get('confidence', 0.5),
            ai_reasoning=reasoning,
            faces_detected=result.get('faces_count', 0),
            faces_with_open_eyes=result.get('faces_with_open_eyes', 0),
            faces_smiling=result.get('faces_smiling', 0),
            animals_detected=False,  # CV doesn't detect animals in this implementation
            key_features=result.get('issues', []),
            technical_metrics=result.get('technical_metrics', {}),
            thumbnail_path=thumbnail_path,
            analysis_method="Computer Vision"
        )
    
    # Helper methods
    def _create_thumbnail(self, image: Image.Image, original_path: str, temp_dir: str) -> str:
        """Create thumbnail for preview"""
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            filename = Path(original_path).stem + "_thumb.jpg"
            thumbnail_path = os.path.join(temp_dir, filename)
            thumbnail.save(thumbnail_path, "JPEG", quality=85)
            
            return thumbnail_path
        except:
            return ""
    
    def _resize_for_api(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image if too large for API"""
        width, height = image.size
        
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's JSON response"""
        try:
            # Clean response
            clean_response = response_text.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            # Parse JSON
            data = json.loads(clean_response)
            
            # Map category to enum
            category_mapping = {
                'EXCELLENT': PhotoQuality.EXCELLENT,
                'GOOD': PhotoQuality.GOOD,
                'ACCEPTABLE': PhotoQuality.ACCEPTABLE,
                'NEEDS_REVIEW': PhotoQuality.NEEDS_REVIEW
            }
            
            quality = category_mapping.get(data.get('category', 'NEEDS_REVIEW'), PhotoQuality.NEEDS_REVIEW)
            
            return {
                'quality': quality,
                'confidence': float(data.get('confidence', 0.5)),
                'reasoning': data.get('reasoning', 'AI analysis completed'),
                'faces_count': int(data.get('faces_count', 0)),
                'animals_detected': bool(data.get('animals_detected', False)),
                'key_features': data.get('key_features', []) + data.get('technical_issues', [])
            }
            
        except Exception as e:
            # Fallback parsing
            response_lower = response_text.lower()
            
            if 'excellent' in response_lower:
                quality = PhotoQuality.EXCELLENT
            elif 'good' in response_lower:
                quality = PhotoQuality.GOOD
            elif 'acceptable' in response_lower:
                quality = PhotoQuality.ACCEPTABLE
            else:
                quality = PhotoQuality.NEEDS_REVIEW
            
            return {
                'quality': quality,
                'confidence': 0.6,
                'reasoning': f"Fallback analysis: {response_text[:200]}",
                'faces_count': 0,
                'animals_detected': False,
                'key_features': ['AI analysis completed']
            }
    
    # Computer Vision helper methods
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def _analyze_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze all faces in the image"""
        if not self.face_mesh:
            return []
        
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        faces_data = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = {}
                for idx, lm in enumerate(face_landmarks.landmark):
                    landmarks[idx] = (int(lm.x * w), int(lm.y * h))
                
                face_data = self._analyze_single_face(landmarks)
                faces_data.append(face_data)
        
        return faces_data
    
    def _analyze_single_face(self, landmarks: Dict[int, Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze a single face for eyes and smile"""
        # Eye aspect ratios
        left_ear = self._eye_aspect_ratio([landmarks[i] for i in self.left_eye_indices])
        right_ear = self._eye_aspect_ratio([landmarks[i] for i in self.right_eye_indices])
        avg_ear = (left_ear + right_ear) / 2
        
        # Mouth analysis
        mouth_points = [landmarks[i] for i in self.mouth_indices]
        is_smiling = self._detect_smile(mouth_points)
        
        return {
            'eyes_open': avg_ear > self.ear_threshold,
            'smiling': is_smiling,
            'ear_score': avg_ear,
            'confidence': min(1.0, avg_ear / 0.3)
        }
    
    def _eye_aspect_ratio(self, eye_points: List[Tuple[int, int]]) -> float:
        """Calculate Eye Aspect Ratio"""
        if len(eye_points) < 6:
            return 0.0
        
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0
    
    def _detect_smile(self, mouth_points: List[Tuple[int, int]]) -> bool:
        """Detect smile based on mouth shape"""
        if len(mouth_points) < 4:
            return False
        
        left, right, top, bottom = mouth_points
        width = np.linalg.norm(np.array(right) - np.array(left))
        height = np.linalg.norm(np.array(bottom) - np.array(top))
        
        return (height / width) > self.mar_threshold if width != 0 else False
    
    def _assess_cv_quality(self, sharpness: float, brightness: float, 
                          contrast: float, faces_data: List[Dict]) -> Tuple[PhotoQuality, float, List[str]]:
        """Assess quality using computer vision metrics"""
        score = 70
        issues = []
        
        # Technical quality
        if sharpness < 15:
            score -= 20
            issues.append("Image appears blurry")
        elif sharpness > 50:
            score += 10
        
        if not (30 <= brightness <= 230):
            if brightness < 50:
                score -= 15
                issues.append("Image is too dark")
            elif brightness > 200:
                score -= 10
                issues.append("Image is very bright")
        
        if contrast < 15:
            score -= 10
            issues.append("Low contrast")
        
        # Face analysis
        if len(faces_data) == 0:
            score -= 10
            issues.append("No faces detected")
        else:
            open_eyes_ratio = sum(1 for f in faces_data if f.get('eyes_open', False)) / len(faces_data)
            smile_ratio = sum(1 for f in faces_data if f.get('smiling', False)) / len(faces_data)
            
            if open_eyes_ratio >= 0.8:
                score += 15
            elif open_eyes_ratio < 0.5:
                score -= 20
                issues.append("Many people have closed eyes")
            
            if smile_ratio >= 0.6:
                score += 10
            elif smile_ratio == 0 and len(faces_data) > 1:
                score -= 5
                issues.append("No one appears to be smiling")
        
        confidence = max(0.1, min(1.0, score / 100.0))
        
        if score >= 85:
            quality = PhotoQuality.EXCELLENT
        elif score >= 70:
            quality = PhotoQuality.GOOD
        elif score >= 55:
            quality = PhotoQuality.ACCEPTABLE
        else:
            quality = PhotoQuality.NEEDS_REVIEW
        
        return quality, confidence, issues
    
    def _create_error_analysis(self, filepath: str, error_msg: str, processing_time: float) -> PhotoAnalysis:
        """Create analysis object for errors"""
        return PhotoAnalysis(
            filename=Path(filepath).name,
            quality=PhotoQuality.NEEDS_REVIEW,
            confidence=0.0,
            ai_reasoning=f"Error during analysis: {error_msg}",
            faces_detected=0,
            faces_with_open_eyes=0,
            faces_smiling=0,
            animals_detected=False,
            key_features=[],
            technical_metrics={},
            processing_time=processing_time,
            analysis_method="Error"
        )

# ============================================================================
# Utility Functions
# ============================================================================

def extract_photos_from_zip(uploaded_file, temp_dir: str) -> List[str]:
    """Extract photos from zip file"""
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.raw', '.cr2', '.nef', '.arw'}
    image_files = []
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def analyze_all_photos(image_files: List[str], analyzer: HybridPhotoAnalyzer, temp_dir: str) -> List[PhotoAnalysis]:
    """Analyze all photos with progress tracking"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    total_files = len(image_files)
    total_time = 0
    
    for i, image_path in enumerate(image_files):
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"🔍 Analyzing: {Path(image_path).name} ({i+1}/{total_files})")
        
        analysis = analyzer.analyze_photo(image_path, temp_dir)
        results.append(analysis)
        
        total_time += analysis.processing_time
        avg_time = total_time / (i + 1)
        remaining_time = avg_time * (total_files - i - 1)
        
        time_text.text(f"⏱️ Avg: {avg_time:.1f}s/photo | Est. remaining: {remaining_time:.0f}s | Method: {analysis.analysis_method}")
        
        # Rate limiting for API calls
        time.sleep(0.2)
    
    progress_bar.empty()
    status_text.empty()
    time_text.empty()
    
    return results

def create_sorted_download(results: List[PhotoAnalysis], temp_dir: str, 
                         selected_qualities: List[PhotoQuality], original_files: List[str]) -> str:
    """Create download zip with sorted photos"""
    download_dir = os.path.join(temp_dir, "ai_sorted_photos")
    os.makedirs(download_dir, exist_ok=True)
    
    # Create quality folders
    for quality in PhotoQuality:
        os.makedirs(os.path.join(download_dir, quality.value), exist_ok=True)
    
    # Create filename to path mapping
    file_mapping = {os.path.basename(path): path for path in original_files}
    
    # Copy files to appropriate folders
    for result in results:
        final_quality = result.user_override if result.user_override else result.quality
        
        if final_quality in selected_qualities:
            source_file = file_mapping.get(result.filename)
            
            if source_file and os.path.exists(source_file):
                dest_folder = os.path.join(download_dir, final_quality.value)
                dest_file = os.path.join(dest_folder, result.filename)
                try:
                    shutil.copy2(source_file, dest_file)
                except Exception as e:
                    st.warning(f"Could not copy {result.filename}: {e}")
    
    # Create zip
    zip_path = os.path.join(temp_dir, "ai_sorted_photos.zip")
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(download_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, download_dir)
                    zipf.write(file_path, arcname)
    except Exception as e:
        st.error(f"Error creating zip: {e}")
        return None
    
    return zip_path



def main():
    st.set_page_config(
        page_title="AI Photo Sorter",
        page_icon="🤖",
        layout="wide"
    )
    
    # Enhanced styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .photo-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .photo-card:hover {
        transform: translateY(-2px);
    }
    .quality-excellent { border-left: 6px solid #28a745; }
    .quality-good { border-left: 6px solid #17a2b8; }
    .quality-acceptable { border-left: 6px solid #ffc107; }
    .quality-needs-review { border-left: 6px solid #dc3545; }
    .ai-reasoning {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .ai-reasoning {
    color: black;
    background: rgb(243 244 247);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 1rem 0;
    }
                
    .technical-metrics {
        background: #f5f5f5;
        padding: 0.8rem;
        border-radius: 6px;
        font-size: 0.9em;
        color : black ;
    }
                
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Photo Sorter</h1>
        <h3>Powered by Hybrid AI + Computer Vision</h3>
        <p>Upload your photos and let advanced AI automatically analyze quality, faces, expressions, and composition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    if 'user_overrides' not in st.session_state:
        st.session_state.user_overrides = {}
    if 'original_files' not in st.session_state:
        st.session_state.original_files = []
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Photos")
        
        uploaded_file = st.file_uploader(
            "Select ZIP file with your photos",
            type=['zip'],
            help="Upload a zip file containing photos in any format (JPG, PNG, HEIC, RAW, etc.)"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024*1024)
            st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size:.1f} MB)")
            
            # Validation
            can_analyze = True
            error_messages = []
            
            if not GEMINI_AVAILABLE or not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
                can_analyze = False
                error_messages.append("❌ Please set your Gemini API key in the code")
            
            if not MEDIAPIPE_AVAILABLE:
                can_analyze = False
                error_messages.append("❌ MediaPipe package not installed")
            
            for error in error_messages:
                st.error(error)
            
            if can_analyze and st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                # Create temp directory
                temp_dir = tempfile.mkdtemp()
                st.session_state.temp_dir = temp_dir
                
                try:
                    # Extract photos
                    with st.spinner("📦 Extracting photos from zip..."):
                        image_files = extract_photos_from_zip(uploaded_file, temp_dir)
                    
                    if not image_files:
                        st.error("❌ No image files found in the uploaded zip!")
                    else:
                        st.info(f"📸 Found {len(image_files)} photos to analyze with Hybrid AI")
                        
                        # Store original files
                        st.session_state.original_files = image_files
                        
                        # Initialize analyzer
                        analyzer = HybridPhotoAnalyzer(api_key=GEMINI_API_KEY)
                        
                        # Analyze all photos
                        results = analyze_all_photos(image_files, analyzer, temp_dir)
                        
                        st.session_state.results = results
                        st.session_state.user_overrides = {}
                        
                        # Success message with stats
                        total_time = sum(r.processing_time for r in results)
                        avg_time = total_time / len(results)
                        
                        st.balloons()
                        success_msg = f"""
                        🎉 **AI Analysis Complete!**
                        - **{len(results)} photos** analyzed using Hybrid AI + CV
                        - **Total time:** {total_time:.1f} seconds  
                        - **Average:** {avg_time:.1f}s per photo
                        """
                        
                        st.success(success_msg)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        
        # Instructions
        if not uploaded_file:
            st.markdown("""
            **How to use:**
            1. Upload a ZIP file with photos
            2. Click "Start AI Analysis"
            3. Review AI + computer vision results
            4. Override any categories if needed
            5. Download intelligently sorted photos
            
            **Supported formats:** JPG, PNG, HEIC, RAW, TIFF, WebP, BMP
            
            **Analysis Features:**
            - 🧠 **AI Understanding**: Context, composition, artistic quality
            - 👁️ **Computer Vision**: Technical metrics, face detection
            - 😊 **Expression Analysis**: Smiles, open/closed eyes
            - 🐾 **Animal Detection**: Pets and wildlife
            - 📊 **Quality Assessment**: Comprehensive photo scoring
            """)
    
    with col2:
        if st.session_state.results:
            st.subheader("📊 AI Analysis Results")
            
            results = st.session_state.results
            
            # Stats with overrides
            quality_counts = defaultdict(int)
            total_faces = 0
            total_open_eyes = 0
            total_smiles = 0
            photos_with_animals = 0
            
            for result in results:
                final_quality = st.session_state.user_overrides.get(result.filename, result.quality)
                quality_counts[final_quality] += 1
                total_faces += result.faces_detected
                total_open_eyes += result.faces_with_open_eyes
                total_smiles += result.faces_smiling
                if result.animals_detected:
                    photos_with_animals += 1
            
            # Display stats
            col2a, col2b, col2c, col2d = st.columns(4)
            
            with col2a:
                st.metric("🌟 Excellent", quality_counts[PhotoQuality.EXCELLENT])
            with col2b:
                st.metric("👍 Good", quality_counts[PhotoQuality.GOOD])
            with col2c:
                st.metric("👌 Acceptable", quality_counts[PhotoQuality.ACCEPTABLE])
            with col2d:
                st.metric("⚠️ Needs Review", quality_counts[PhotoQuality.NEEDS_REVIEW])
            
            # Additional insights
            st.markdown("---")
            col_insights1, col_insights2, col_insights3, col_insights4 = st.columns(4)
            
            with col_insights1:
                st.metric("👥 Total Faces", total_faces)
            with col_insights2:
                st.metric("👁️ Open Eyes", total_open_eyes)
            with col_insights3:
                st.metric("😊 Smiling", total_smiles)
            with col_insights4:
                st.metric("🐾 With Animals", photos_with_animals)
    
    # Photo review section
    if st.session_state.results:
        st.markdown("---")
        st.header("🔍 Review AI Analysis")
        
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            view_quality = st.selectbox(
                "Filter by category:",
                options=["All"] + [q.value for q in PhotoQuality],
                index=0
            )
        
        with col_filter2:
            show_technical = st.checkbox("Show technical metrics", value=True)
        
        with col_filter3:
            show_ai_reasoning = st.checkbox("Show AI reasoning", value=True)
        
        # Filter results
        if view_quality == "All":
            filtered_results = st.session_state.results
        else:
            target_quality = next(q for q in PhotoQuality if q.value == view_quality)
            filtered_results = []
            for result in st.session_state.results:
                current_quality = st.session_state.user_overrides.get(result.filename, result.quality)
                if current_quality == target_quality:
                    filtered_results.append(result)
        
        st.write(f"📋 Showing {len(filtered_results)} photos")
        
        # Photo display
        if filtered_results:
            for result in filtered_results:
                current_quality = st.session_state.user_overrides.get(result.filename, result.quality)
                quality_class = f"quality-{current_quality.value.lower().replace(' ', '-')}"
                
                st.markdown(f'<div class="photo-card {quality_class}">', unsafe_allow_html=True)
                
                col_photo, col_details = st.columns([1, 2])
                
                with col_photo:
                    # Show thumbnail
                    if result.thumbnail_path and os.path.exists(result.thumbnail_path):
                        image = Image.open(result.thumbnail_path)
                        st.image(image, caption=result.filename, use_column_width=True)
                    else:
                        st.markdown(f"📷 **{result.filename}**")
                
                with col_details:
                    # Title and category
                    st.markdown(f"### {result.filename}")
                    st.markdown(f"**Method:** {result.analysis_method} | **Time:** {result.processing_time:.1f}s")
                    
                    # Current category with override indicator
                    if result.filename in st.session_state.user_overrides:
                        st.markdown(f"**Current:** {current_quality.value} *(manually set)*")
                        st.markdown(f"**AI Original:** {result.quality.value}")
                    else:
                        st.markdown(f"**AI Category:** {current_quality.value}")
                    
                    # Override selector
                    quality_values = [q.value for q in PhotoQuality]
                    current_index = quality_values.index(current_quality.value)
                    
                    new_quality = st.selectbox(
                        "Override category:",
                        options=quality_values,
                        index=current_index,
                        key=f"override_{result.filename}"
                    )
                    
                    # Handle override
                    selected_quality = next(q for q in PhotoQuality if q.value == new_quality)
                    if selected_quality != result.quality:
                        st.session_state.user_overrides[result.filename] = selected_quality
                    elif result.filename in st.session_state.user_overrides and selected_quality == result.quality:
                        del st.session_state.user_overrides[result.filename]
                    
                    # Stats
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.markdown(f"**🎯 Confidence:** {result.confidence:.0%}")
                        if result.faces_detected > 0:
                            st.markdown(f"**👥 Faces:** {result.faces_detected}")
                    
                    with col_stats2:
                        if result.faces_detected > 0:
                            st.markdown(f"**👁️ Open Eyes:** {result.faces_with_open_eyes}/{result.faces_detected}")
                            st.markdown(f"**😊 Smiling:** {result.faces_smiling}/{result.faces_detected}")
                    
                    with col_stats3:
                        if result.animals_detected:
                            st.markdown("**🐾 Animals:** Detected")
                        if result.technical_metrics:
                            avg_metric = sum(result.technical_metrics.values()) / len(result.technical_metrics)
                            st.markdown(f"**📊 Tech Score:** {avg_metric:.0f}")
                    
                    # Key features
                    if result.key_features:
                        st.markdown("**✨ Key Features:**")
                        for feature in result.key_features[:4]:  # Show top 4
                            st.markdown(f"• {feature}")
                
                # Technical metrics (full width)
                if show_technical and result.technical_metrics:
                    st.markdown(f"""
                    <div class="technical-metrics">
                        <strong>🔧 Technical Metrics:</strong><br>
                        Sharpness: {result.technical_metrics.get('sharpness', 0):.1f} | 
                        Brightness: {result.technical_metrics.get('brightness', 0):.1f} | 
                        Contrast: {result.technical_metrics.get('contrast', 0):.1f}
                    </div>
                    """, unsafe_allow_html=True)
                
                # AI reasoning (full width)
                if show_ai_reasoning:
                    st.markdown(f"""
                    <div class="ai-reasoning">
                        <strong>🤖 AI Analysis:</strong><br>
                        {result.ai_reasoning}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    
    # Download section
    if st.session_state.results:
        st.header("📥 Download AI-Sorted Photos")
        
        # Recalculate counts with overrides
        final_quality_counts = {q: 0 for q in PhotoQuality}
        for result in st.session_state.results:
            final_quality = st.session_state.user_overrides.get(result.filename, result.quality)
            final_quality_counts[final_quality] += 1
        
        st.write("Select categories to include in download:")
        
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        
        download_selection = {}
        with col_dl1:
            download_selection[PhotoQuality.EXCELLENT] = st.checkbox(
                f"🌟 Excellent ({final_quality_counts[PhotoQuality.EXCELLENT]})",
                value=True,
                key="dl_excellent"
            )
        with col_dl2:
            download_selection[PhotoQuality.GOOD] = st.checkbox(
                f"👍 Good ({final_quality_counts[PhotoQuality.GOOD]})",
                value=True,
                key="dl_good"
            )
        with col_dl3:
            download_selection[PhotoQuality.ACCEPTABLE] = st.checkbox(
                f"👌 Acceptable ({final_quality_counts[PhotoQuality.ACCEPTABLE]})",
                value=False,
                key="dl_acceptable"
            )
        with col_dl4:
            download_selection[PhotoQuality.NEEDS_REVIEW] = st.checkbox(
                f"⚠️ Needs Review ({final_quality_counts[PhotoQuality.NEEDS_REVIEW]})",
                value=False,
                key="dl_review"
            )
        
        selected_qualities = [q for q, selected in download_selection.items() if selected]
        total_selected = sum(final_quality_counts[q] for q in selected_qualities)
        
        if total_selected > 0:
            st.success(f"✅ Ready to download {total_selected} AI-sorted photos")
            
            # Override summary
            override_count = len(st.session_state.user_overrides)
            if override_count > 0:
                st.info(f"📝 {override_count} photos manually overridden")
            
            if st.button("📦 Create AI Photo Package", type="primary", use_container_width=True):
                if st.session_state.temp_dir and st.session_state.original_files:
                    with st.spinner("📦 Creating your AI-sorted photo package..."):
                        try:
                            # Update results with overrides
                            updated_results = []
                            for result in st.session_state.results:
                                if result.filename in st.session_state.user_overrides:
                                    result.user_override = st.session_state.user_overrides[result.filename]
                                updated_results.append(result)
                            
                            zip_path = create_sorted_download(
                                updated_results,
                                st.session_state.temp_dir,
                                selected_qualities,
                                st.session_state.original_files
                            )
                            
                            if zip_path:
                                # Download button
                                with open(zip_path, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Download AI-Sorted Photos",
                                        data=f.read(),
                                        file_name="sorted_photos.zip",
                                        mime="application/zip",
                                        type="primary",
                                        use_container_width=True
                                    )
                                
                                st.success("🎉 Your AI-sorted photos are ready!")
                            else:
                                st.error("❌ Failed to create download package")
                                
                        except Exception as e:
                            st.error(f"❌ Error creating download: {str(e)}")
        else:
            st.warning("⚠️ Please select at least one category to download")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;">
        <p><strong>AI Photo Sorter</strong> - Powered by Hybrid AI + Computer Vision</p>
        <p>Combining Gemini AI with MediaPipe for intelligent photo analysis</p>
        <p>🤖 <strong>Automatic hybrid analysis</strong> provides the most comprehensive photo sorting experience</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

   #not bad but has issues in the interface and with face countingt