import numpy as np
import math
from typing import List, Tuple
from ..schemas.movement import BodyKeypoint


class BiomechanicsCalculator:
    """
    Calculadora de biomecánica para análisis de movimientos de surf
    """
    
    def __init__(self):
        # MediaPipe Pose keypoint indices
        self.KEYPOINT_MAPPING = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def calculate_distance(self, point1: BodyKeypoint, point2: BodyKeypoint) -> float:
        """
        Calcula la distancia euclidiana entre dos puntos
        """
        dx = point1.x - point2.x
        dy = point1.y - point2.y
        if point1.z is not None and point2.z is not None:
            dz = point1.z - point2.z
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        return math.sqrt(dx*dx + dy*dy)
    
    def calculate_angle(self, point1: BodyKeypoint, vertex: BodyKeypoint, point2: BodyKeypoint) -> float:
        """
        Calcula el ángulo formado por tres puntos (vertex es el punto central)
        Retorna el ángulo en grados
        """
        # Vectores desde el vértice a los otros puntos
        v1 = np.array([point1.x - vertex.x, point1.y - vertex.y])
        v2 = np.array([point2.x - vertex.x, point2.y - vertex.y])
        
        # Calcular el ángulo usando producto punto
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar errores de precisión
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def get_midpoint(self, point1: BodyKeypoint, point2: BodyKeypoint) -> BodyKeypoint:
        """
        Calcula el punto medio entre dos keypoints
        """
        mid_x = (point1.x + point2.x) / 2
        mid_y = (point1.y + point2.y) / 2
        mid_z = None
        if point1.z is not None and point2.z is not None:
            mid_z = (point1.z + point2.z) / 2
        
        confidence = min(point1.confidence, point2.confidence)
        
        return BodyKeypoint(x=mid_x, y=mid_y, z=mid_z, confidence=confidence)
    
    def get_body_center(self, keypoints: List[BodyKeypoint]) -> BodyKeypoint:
        """
        Calcula el centro de masa aproximado del cuerpo
        """
        # Usar puntos principales: hombros y caderas con pesos
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        
        # Centro de hombros y caderas
        shoulder_center = self.get_midpoint(left_shoulder, right_shoulder)
        hip_center = self.get_midpoint(left_hip, right_hip)
        
        # Centro de masa aproximado (40% hombros, 60% caderas)
        center_x = shoulder_center.x * 0.4 + hip_center.x * 0.6
        center_y = shoulder_center.y * 0.4 + hip_center.y * 0.6
        center_z = None
        if shoulder_center.z is not None and hip_center.z is not None:
            center_z = shoulder_center.z * 0.4 + hip_center.z * 0.6
        
        confidence = min(shoulder_center.confidence, hip_center.confidence)
        
        return BodyKeypoint(x=center_x, y=center_y, z=center_z, confidence=confidence)
    
    def calculate_body_inclination(self, keypoints: List[BodyKeypoint]) -> float:
        """
        Calcula la inclinación lateral del cuerpo en grados
        """
        # Usar línea de hombros para calcular inclinación
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        
        # Calcular ángulo de la línea de hombros respecto a la horizontal
        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_knee_flexion(self, side: str, keypoints: List[BodyKeypoint]) -> float:
        """
        Calcula la flexión de rodilla (lado: 'left' o 'right')
        """
        if side == 'left':
            hip = keypoints[23]
            knee = keypoints[25]
            ankle = keypoints[27]
        else:
            hip = keypoints[24]
            knee = keypoints[26]
            ankle = keypoints[28]
        
        # Calcular ángulo de flexión de rodilla
        return self.calculate_angle(hip, knee, ankle)
    
    def calculate_spine_curvature(self, keypoints: List[BodyKeypoint]) -> float:
        """
        Calcula la curvatura de la columna vertebral
        """
        # Usar puntos de referencia: cabeza, hombros, caderas
        nose = keypoints[0]
        shoulder_center = self.get_midpoint(keypoints[11], keypoints[12])
        hip_center = self.get_midpoint(keypoints[23], keypoints[24])
        
        # Calcular ángulo de la columna
        return self.calculate_angle(nose, shoulder_center, hip_center)
    
    def calculate_arm_extension(self, side: str, keypoints: List[BodyKeypoint]) -> float:
        """
        Calcula la extensión del brazo (lado: 'left' o 'right')
        """
        if side == 'left':
            shoulder = keypoints[11]
            elbow = keypoints[13]
            wrist = keypoints[15]
        else:
            shoulder = keypoints[12]
            elbow = keypoints[14]
            wrist = keypoints[16]
        
        # Calcular ángulo de extensión del brazo
        return self.calculate_angle(shoulder, elbow, wrist)
    
    def detect_movement_phase(self, current_frame: List[BodyKeypoint], previous_frame: List[BodyKeypoint]) -> str:
        """
        Detecta la fase del movimiento comparando frames consecutivos
        """
        if not previous_frame:
            return "initial"
        
        # Calcular cambios en posiciones clave
        current_center = self.get_body_center(current_frame)
        previous_center = self.get_body_center(previous_frame)
        
        # Calcular velocidad vertical (cambio en Y)
        vertical_velocity = current_center.y - previous_center.y
        
        # Calcular cambio en altura de manos
        current_hands_height = (current_frame[15].y + current_frame[16].y) / 2
        previous_hands_height = (previous_frame[15].y + previous_frame[16].y) / 2
        hands_velocity = current_hands_height - previous_hands_height
        
        # Determinar fase basada en movimientos
        if hands_velocity < -0.02:  # Manos subiendo (Y decrece en imagen)
            return "pushing_up"
        elif vertical_velocity < -0.01:  # Cuerpo subiendo
            return "rising"
        elif abs(vertical_velocity) < 0.005:  # Posición estable
            return "stable"
        else:
            return "transitioning"
    
    def smooth_keypoints(self, keypoint_sequence: List[List[BodyKeypoint]], window_size: int = 3) -> List[List[BodyKeypoint]]:
        """
        Suaviza una secuencia de keypoints usando ventana móvil
        """
        if len(keypoint_sequence) < window_size:
            return keypoint_sequence
        
        smoothed_sequence = []
        half_window = window_size // 2
        
        for i in range(len(keypoint_sequence)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(keypoint_sequence), i + half_window + 1)
            
            # Promediar keypoints en la ventana
            smoothed_frame = []
            for kp_idx in range(len(keypoint_sequence[0])):
                x_values = [keypoint_sequence[j][kp_idx].x for j in range(start_idx, end_idx)]
                y_values = [keypoint_sequence[j][kp_idx].y for j in range(start_idx, end_idx)]
                confidences = [keypoint_sequence[j][kp_idx].confidence for j in range(start_idx, end_idx)]
                
                smoothed_kp = BodyKeypoint(
                    x=np.mean(x_values),
                    y=np.mean(y_values),
                    z=None,  # Por simplicidad, no suavizamos Z
                    confidence=np.mean(confidences)
                )
                smoothed_frame.append(smoothed_kp)
            
            smoothed_sequence.append(smoothed_frame)
        
        return smoothed_sequence
    
    def validate_keypoints_quality(self, keypoints: List[BodyKeypoint], min_confidence: float = 0.5) -> bool:
        """
        Valida la calidad de los keypoints detectados
        """
        # Keypoints críticos para análisis de surf
        critical_keypoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        
        for idx in critical_keypoints:
            if idx < len(keypoints) and keypoints[idx].confidence < min_confidence:
                return False
        
        return True
    
    def calculate_movement_velocity(self, keypoints1: List[BodyKeypoint], keypoints2: List[BodyKeypoint], time_diff: float) -> float:
        """
        Calcula la velocidad del movimiento entre dos frames
        """
        center1 = self.get_body_center(keypoints1)
        center2 = self.get_body_center(keypoints2)
        
        distance = self.calculate_distance(center1, center2)
        velocity = distance / time_diff if time_diff > 0 else 0
        
        return velocity