import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Dict, List, Optional
from app.schemas.movement_analysis_video import MovementType

mp_pose = mp.solutions.pose

def analyze_video(file_path: str, movement_type: MovementType) -> dict:
    """
    Procesa el video usando MediaPipe y extrae puntos clave para el análisis biomecánico.
    """
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    cap = cv2.VideoCapture(file_path)
    
    # Verificar que el video se abrió correctamente
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {file_path}")
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    keypoints_log = []
    processed_frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar solo cada 5 frames para eficiencia
            if frame_count % 5 == 0:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        landmarks = extract_landmarks(results.pose_landmarks)
                        if landmarks:  # Solo agregar si hay landmarks válidos
                            keypoints_log.append({
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps if fps > 0 else 0,
                                'landmarks': landmarks
                            })
                            processed_frames += 1
                            
                except Exception as e:
                    print(f"Error procesando frame {frame_count}: {str(e)}")
                    continue

            frame_count += 1

    finally:
        cap.release()
        pose.close()

    # Verificar que se procesaron frames
    if not keypoints_log:
        raise ValueError("No se pudieron extraer poses del video. Verifica que el video contenga personas visibles.")

    # Evaluar biomecánica
    biomechanical_issues = evaluate_keypoints(keypoints_log, movement_type)

    return {
        "frames_analyzed": processed_frames,
        "total_frames": total_frames,
        "movement_detected": movement_type.value,
        "issues_found": biomechanical_issues,
        "video_duration": total_frames / fps if fps > 0 else 0,
        "fps": fps
    }


def extract_landmarks(pose_landmarks) -> Optional[Dict]:
    """
    Extrae los landmarks de MediaPipe en un formato más manejable.
    """
    try:
        landmarks = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmark_name = mp_pose.PoseLandmark(idx).name
            landmarks[landmark_name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            }
        return landmarks
    except Exception as e:
        print(f"Error extrayendo landmarks: {str(e)}")
        return None


def evaluate_keypoints(keypoints_sequence: List[Dict], movement_type: MovementType) -> List[str]:
    """
    Evalúa los keypoints del video en base al movimiento específico.
    """
    issues = []

    if not keypoints_sequence:
        return ["No se detectaron posturas válidas en el video"]

    try:
        if movement_type == MovementType.TAKE_OFF:
            issues.extend(analyze_takeoff_movement(keypoints_sequence))
        elif movement_type == MovementType.BOTTOM_TURN:
            issues.extend(analyze_bottom_turn_movement(keypoints_sequence))
        else:
            issues.extend(analyze_general_posture(keypoints_sequence))

    except Exception as e:
        issues.append(f"Error en el análisis biomecánico: {str(e)}")

    # Si no se encontraron problemas específicos
    if not issues:
        issues.append("Análisis completado. No se detectaron problemas biomecánicos significativos.")

    return issues


def analyze_takeoff_movement(keypoints_sequence: List[Dict]) -> List[str]:
    """
    Analiza específicamente el movimiento de take-off.
    """
    issues = []
    valid_frames = 0

    for frame_data in keypoints_sequence:
        landmarks = frame_data.get('landmarks', {})
        
        # Verificar landmarks necesarios
        required_landmarks = ['LEFT_HIP', 'RIGHT_HIP', 'NOSE', 'LEFT_KNEE', 'RIGHT_KNEE']
        if not all(landmark in landmarks for landmark in required_landmarks):
            continue

        valid_frames += 1

        # Análisis de alineación cabeza-cadera
        left_hip = landmarks['LEFT_HIP']
        right_hip = landmarks['RIGHT_HIP']
        nose = landmarks['NOSE']

        # Verificar visibilidad
        if all(landmark['visibility'] > 0.5 for landmark in [left_hip, right_hip, nose]):
            avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
            head_hip_diff = abs(nose['y'] - avg_hip_y)
            
            if head_hip_diff < 0.05:  # Muy poca diferencia vertical
                issues.append("Se detectó posible encorvamiento: cabeza muy baja respecto a las caderas")
                break

        # Análisis de rodillas
        left_knee = landmarks['LEFT_KNEE']
        right_knee = landmarks['RIGHT_KNEE']
        
        if left_knee['visibility'] > 0.5 and right_knee['visibility'] > 0.5:
            knee_alignment_diff = abs(left_knee['y'] - right_knee['y'])
            if knee_alignment_diff > 0.1:  # Diferencia significativa en altura
                issues.append("Se detectó desalineación en las rodillas")
                break

    if valid_frames == 0:
        issues.append("No se pudieron analizar suficientes frames con poses válidas")

    return issues


def analyze_bottom_turn_movement(keypoints_sequence: List[Dict]) -> List[str]:
    """
    Analiza específicamente el movimiento de bottom turn.
    """
    issues = []
    valid_frames = 0

    for frame_data in keypoints_sequence:
        landmarks = frame_data.get('landmarks', {})
        
        required_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
        if not all(landmark in landmarks for landmark in required_landmarks):
            continue

        valid_frames += 1

        # Análisis de rotación de hombros
        left_shoulder = landmarks['LEFT_SHOULDER']
        right_shoulder = landmarks['RIGHT_SHOULDER']
        
        if left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
            shoulder_rotation = abs(left_shoulder['y'] - right_shoulder['y'])
            if shoulder_rotation < 0.02:  # Muy poca rotación
                issues.append("Se detectó poca rotación en los hombros durante el giro")
                break

    if valid_frames == 0:
        issues.append("No se pudieron analizar suficientes frames para el bottom turn")

    return issues


def analyze_general_posture(keypoints_sequence: List[Dict]) -> List[str]:
    """
    Análisis general de postura para movimientos no específicos.
    """
    issues = []
    
    # Contar frames con detección válida
    valid_frames = sum(1 for frame in keypoints_sequence 
                      if frame.get('landmarks') and 
                      any(landmark.get('visibility', 0) > 0.5 
                          for landmark in frame['landmarks'].values()))
    
    if valid_frames < len(keypoints_sequence) * 0.5:
        issues.append("Detección de pose limitada. Asegúrate de que la persona esté claramente visible en el video.")
    
    return issues