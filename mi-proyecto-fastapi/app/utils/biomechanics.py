import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import math

@dataclass
class BodyKeypoint:
    """Punto clave del cuerpo con coordenadas y confianza"""
    x: float
    y: float
    z: float
    confidence: float

    def __post_init__(self):
        # Validar que los valores sean numéricos
        self.x = float(self.x) if not math.isnan(float(self.x)) else 0.0
        self.y = float(self.y) if not math.isnan(float(self.y)) else 0.0
        self.z = float(self.z) if not math.isnan(float(self.z)) else 0.0
        self.confidence = float(self.confidence) if not math.isnan(float(self.confidence)) else 0.0


class BiomechanicsCalculator:
    """Calculadora de métricas biomecánicas usando landmarks de MediaPipe"""
    
    def __init__(self):
        # Índices de landmarks importantes de MediaPipe Pose
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_FOOT_INDEX = 31
        self.RIGHT_FOOT_INDEX = 32

    def validate_keypoints_quality(self, landmarks: List[BodyKeypoint], min_confidence: float = 0.7) -> bool:
        """Valida la calidad de los keypoints basado en confianza"""
        if not landmarks or len(landmarks) < 33:
            return False
        
        # Verificar confianza de puntos clave críticos
        critical_points = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_KNEE, self.RIGHT_KNEE,
            self.LEFT_ANKLE, self.RIGHT_ANKLE
        ]
        
        for idx in critical_points:
            if idx < len(landmarks) and landmarks[idx].confidence < min_confidence:
                return False
        
        return True

    def calculate_distance(self, point1: BodyKeypoint, point2: BodyKeypoint) -> float:
        """Calcula distancia euclidiana entre dos puntos"""
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        dz = point2.z - point1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def get_midpoint(self, point1: BodyKeypoint, point2: BodyKeypoint) -> BodyKeypoint:
        """Obtiene el punto medio entre dos keypoints"""
        mid_x = (point1.x + point2.x) / 2
        mid_y = (point1.y + point2.y) / 2
        mid_z = (point1.z + point2.z) / 2
        mid_conf = (point1.confidence + point2.confidence) / 2
        
        return BodyKeypoint(mid_x, mid_y, mid_z, mid_conf)

    def calculate_angle(self, point1: BodyKeypoint, vertex: BodyKeypoint, point2: BodyKeypoint) -> float:
        """Calcula el ángulo formado por tres puntos (vertex es el vértice)"""
        try:
            # Vectores desde el vértice a los otros puntos
            v1 = np.array([point1.x - vertex.x, point1.y - vertex.y, point1.z - vertex.z])
            v2 = np.array([point2.x - vertex.x, point2.y - vertex.y, point2.z - vertex.z])
            
            # Calcular ángulo usando producto punto
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            cos_angle = dot_product / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar errores numéricos
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def calculate_head_inclination(self, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la inclinación de la cabeza respecto al horizonte"""
        try:
            if len(landmarks) <= max(self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER):
                return 0.0
            
            nose = landmarks[self.NOSE]
            shoulder_center = self.get_midpoint(landmarks[self.LEFT_SHOULDER], landmarks[self.RIGHT_SHOULDER])
            
            # Calcular ángulo de inclinación
            dx = nose.x - shoulder_center.x
            dy = nose.y - shoulder_center.y
            
            angle_rad = math.atan2(abs(dy), abs(dx))
            angle_deg = math.degrees(angle_rad)
            
            return float(angle_deg)
        except:
            return 0.0

    def calculate_arm_extension(self, side: str, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la extensión del brazo (ángulo del codo)"""
        try:
            if side == 'left':
                shoulder_idx, elbow_idx, wrist_idx = self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST
            else:
                shoulder_idx, elbow_idx, wrist_idx = self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST
            
            if len(landmarks) <= max(shoulder_idx, elbow_idx, wrist_idx):
                return 0.0
            
            shoulder = landmarks[shoulder_idx]
            elbow = landmarks[elbow_idx]
            wrist = landmarks[wrist_idx]
            
            return self.calculate_angle(shoulder, elbow, wrist)
        except:
            return 0.0

    def calculate_knee_flexion(self, side: str, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la flexión de la rodilla"""
        try:
            if side == 'left':
                hip_idx, knee_idx, ankle_idx = self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE
            else:
                hip_idx, knee_idx, ankle_idx = self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE
            
            if len(landmarks) <= max(hip_idx, knee_idx, ankle_idx):
                return 0.0
            
            hip = landmarks[hip_idx]
            knee = landmarks[knee_idx]
            ankle = landmarks[ankle_idx]
            
            return self.calculate_angle(hip, knee, ankle)
        except:
            return 0.0

    def calculate_body_inclination(self, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la inclinación corporal usando hombros y caderas"""
        try:
            if len(landmarks) <= max(self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP):
                return 0.0
            
            shoulder_center = self.get_midpoint(landmarks[self.LEFT_SHOULDER], landmarks[self.RIGHT_SHOULDER])
            hip_center = self.get_midpoint(landmarks[self.LEFT_HIP], landmarks[self.RIGHT_HIP])
            
            # Calcular ángulo de inclinación del torso
            dx = shoulder_center.x - hip_center.x
            dy = shoulder_center.y - hip_center.y
            
            angle_rad = math.atan2(abs(dx), abs(dy))
            angle_deg = math.degrees(angle_rad)
            
            return float(angle_deg)
        except:
            return 0.0

    def calculate_spine_curvature(self, landmarks: List[BodyKeypoint]) -> float:
        """Estima la curvatura de la columna"""
        try:
            if len(landmarks) <= max(self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP):
                return 0.0
            
            # Usar puntos de referencia para estimar curvatura
            nose = landmarks[self.NOSE]
            shoulder_center = self.get_midpoint(landmarks[self.LEFT_SHOULDER], landmarks[self.RIGHT_SHOULDER])
            hip_center = self.get_midpoint(landmarks[self.LEFT_HIP], landmarks[self.RIGHT_HIP])
            
            # Calcular desviación de la línea recta hombro-cadera
            # Vector de hombro a cadera
            shoulder_to_hip = np.array([hip_center.x - shoulder_center.x, hip_center.y - shoulder_center.y])
            # Vector de hombro a nariz
            shoulder_to_nose = np.array([nose.x - shoulder_center.x, nose.y - shoulder_center.y])
            
            # Proyección de la nariz sobre la línea hombro-cadera
            if np.linalg.norm(shoulder_to_hip) == 0:
                return 0.0
            
            projection = np.dot(shoulder_to_nose, shoulder_to_hip) / np.linalg.norm(shoulder_to_hip)
            projection_point = shoulder_center.x + projection * shoulder_to_hip[0] / np.linalg.norm(shoulder_to_hip), \
                              shoulder_center.y + projection * shoulder_to_hip[1] / np.linalg.norm(shoulder_to_hip)
            
            # Distancia perpendicular (curvatura)
            curvature_distance = math.sqrt((nose.x - projection_point[0])**2 + (nose.y - projection_point[1])**2)
            
            # Convertir a grados (aproximación)
            curvature_angle = math.degrees(math.atan(curvature_distance / max(0.1, np.linalg.norm(shoulder_to_hip))))
            
            return float(curvature_angle)
        except:
            return 10.0  # Valor por defecto

    def get_body_center(self, landmarks: List[BodyKeypoint]) -> BodyKeypoint:
        """Calcula el centro de masa corporal aproximado"""
        try:
            if len(landmarks) <= max(self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP):
                return BodyKeypoint(0.5, 0.5, 0, 0.5)
            
            # Usar puntos clave para aproximar centro de masa
            key_points = [
                landmarks[self.LEFT_SHOULDER],
                landmarks[self.RIGHT_SHOULDER],
                landmarks[self.LEFT_HIP],
                landmarks[self.RIGHT_HIP]
            ]
            
            # Peso ponderado (hombros menos peso que caderas para centro de masa)
            weights = [0.2, 0.2, 0.3, 0.3]
            
            center_x = sum(point.x * weight for point, weight in zip(key_points, weights))
            center_y = sum(point.y * weight for point, weight in zip(key_points, weights))
            center_z = sum(point.z * weight for point, weight in zip(key_points, weights))
            avg_confidence = sum(point.confidence for point in key_points) / len(key_points)
            
            return BodyKeypoint(center_x, center_y, center_z, avg_confidence)
        except:
            return BodyKeypoint(0.5, 0.5, 0, 0.5)

    def calculate_hip_rotation(self, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la rotación de caderas"""
        try:
            if len(landmarks) <= max(self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_SHOULDER, self.RIGHT_SHOULDER):
                return 0.0
            
            left_hip = landmarks[self.LEFT_HIP]
            right_hip = landmarks[self.RIGHT_HIP]
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            # Vector de caderas
            hip_vector = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y])
            # Vector de hombros
            shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
            
            # Calcular ángulo entre vectores
            if np.linalg.norm(hip_vector) == 0 or np.linalg.norm(shoulder_vector) == 0:
                return 0.0
            
            cos_angle = np.dot(hip_vector, shoulder_vector) / (np.linalg.norm(hip_vector) * np.linalg.norm(shoulder_vector))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
        except:
            return 0.0

    def smooth_keypoints(self, keypoints_sequence: List[List[BodyKeypoint]], window_size: int = 5) -> List[List[BodyKeypoint]]:
        """Suaviza la secuencia de keypoints usando promedio móvil"""
        if len(keypoints_sequence) < window_size:
            return keypoints_sequence
        
        smoothed_sequence = []
        
        for i in range(len(keypoints_sequence)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(keypoints_sequence), i + window_size // 2 + 1)
            
            # Suavizar cada keypoint
            smoothed_frame = []
            
            for kp_idx in range(len(keypoints_sequence[i])):
                x_values = []
                y_values = []
                z_values = []
                conf_values = []
                
                for frame_idx in range(start_idx, end_idx):
                    if kp_idx < len(keypoints_sequence[frame_idx]):
                        kp = keypoints_sequence[frame_idx][kp_idx]
                        x_values.append(kp.x)
                        y_values.append(kp.y)
                        z_values.append(kp.z)
                        conf_values.append(kp.confidence)
                
                if x_values:
                    smoothed_kp = BodyKeypoint(
                        x=np.mean(x_values),
                        y=np.mean(y_values),
                        z=np.mean(z_values),
                        confidence=np.mean(conf_values)
                    )
                    smoothed_frame.append(smoothed_kp)
                else:
                    smoothed_frame.append(keypoints_sequence[i][kp_idx])
            
            smoothed_sequence.append(smoothed_frame)
        
        return smoothed_sequence

    def detect_movement_phase(self, current_landmarks: List[BodyKeypoint], previous_landmarks: List[BodyKeypoint]) -> str:
        """Detecta la fase del movimiento basado en cambios de posición"""
        try:
            if not current_landmarks or not previous_landmarks:
                return "unknown"
            
            # Calcular movimiento del centro de masa
            current_center = self.get_body_center(current_landmarks)
            previous_center = self.get_body_center(previous_landmarks)
            
            # Velocidad de movimiento
            dx = current_center.x - previous_center.x
            dy = current_center.y - previous_center.y
            velocity = math.sqrt(dx*dx + dy*dy)
            
            # Clasificar fase basada en velocidad y posición
            if velocity < 0.01:
                return "static"
            elif velocity < 0.05:
                return "transition"
            elif dy < -0.02:  # Movimiento hacia arriba
                return "rising"
            elif dy > 0.02:   # Movimiento hacia abajo
                return "descending"
            else:
                return "dynamic"
                
        except:
            return "unknown"

    def calculate_shoulder_alignment(self, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la alineación de hombros (grados respecto al horizonte)"""
        try:
            if len(landmarks) <= max(self.LEFT_SHOULDER, self.RIGHT_SHOULDER):
                return 0.0
            
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            # Calcular ángulo de inclinación de hombros
            dx = right_shoulder.x - left_shoulder.x
            dy = right_shoulder.y - left_shoulder.y
            
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            return float(abs(angle_deg))
        except:
            return 0.0

    def calculate_foot_separation_ratio(self, landmarks: List[BodyKeypoint]) -> float:
        """Calcula la ratio de separación de pies respecto al ancho de hombros"""
        try:
            if len(landmarks) <= max(self.LEFT_ANKLE, self.RIGHT_ANKLE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER):
                return 1.0
            
            # Distancia entre pies
            foot_distance = self.calculate_distance(landmarks[self.LEFT_ANKLE], landmarks[self.RIGHT_ANKLE])
            
            # Distancia entre hombros
            shoulder_distance = self.calculate_distance(landmarks[self.LEFT_SHOULDER], landmarks[self.RIGHT_SHOULDER])
            
            if shoulder_distance == 0:
                return 1.0
            
            ratio = foot_distance / shoulder_distance
            return float(ratio)
        except:
            return 1.0