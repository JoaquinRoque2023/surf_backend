import numpy as np
from typing import List, Dict, Tuple
from ..schemas.movement import BodyFrame, MovementCheckpoint, MovementDirection
from ..utils.biomechanics import BiomechanicsCalculator
from ..core.config import settings


class TakeoffAnalyzer:
    """
    Analizador específico para el movimiento Take Off
    """
    
    def __init__(self):
        self.biomech = BiomechanicsCalculator()
        self.max_duration = settings.TAKEOFF_MAX_DURATION
    
    def analyze_takeoff(
        self, 
        body_frames: List[BodyFrame], 
        direction: MovementDirection
    ) -> Dict:
        """
        Analiza el movimiento de take off completo
        """
        if len(body_frames) < 2:
            raise ValueError("Se necesitan al menos 2 frames para analizar el takeoff")
        
        duration = body_frames[-1].timestamp - body_frames[0].timestamp
        
        # Calcular checkpoints específicos
        checkpoints = self._analyze_checkpoints(body_frames, direction)
        
        # Calcular scores detallados
        scores = self._calculate_scores(body_frames, checkpoints, duration)
        
        # Generar feedback
        feedback = self._generate_feedback(checkpoints, scores, duration, direction)
        
        return {
            "checkpoints": checkpoints,
            "scores": scores,
            "duration": duration,
            "feedback": feedback,
            "is_correct": scores["overall"] >= 70  # 70% threshold for correct execution
        }
    
    def _analyze_checkpoints(
        self, 
        body_frames: List[BodyFrame], 
        direction: MovementDirection
    ) -> Dict[str, MovementCheckpoint]:
        """
        Analiza cada checkpoint específico del takeoff
        """
        checkpoints = {}
        
        # 1. Espalda recta (straight back)
        checkpoints["straight_back"] = self._check_straight_back(body_frames)
        
        # 2. Pecho abierto (open chest)
        checkpoints["open_chest"] = self._check_open_chest(body_frames)
        
        # 3. Manos cerca del pecho (hands near chest)
        checkpoints["hands_near_chest"] = self._check_hands_position(body_frames)
        
        # 4. Cuerpo centrado entre los pies (body centered)
        checkpoints["body_centered"] = self._check_body_centering(body_frames)
        
        # 5. Timing correcto (timing correct)
        checkpoints["timing_correct"] = self._check_timing(body_frames)
        
        # 6. Posicionamiento de brazos específico por dirección
        checkpoints["arm_positioning"] = self._check_arm_positioning(body_frames, direction)
        
        return checkpoints
    
    def _check_straight_back(self, body_frames: List[BodyFrame]) -> MovementCheckpoint:
        """
        Verifica si la espalda se mantiene recta durante el takeoff
        """
        back_angles = []
        
        for frame in body_frames:
            # Keypoints: 11-shoulder_left, 12-shoulder_right, 23-hip_left, 24-hip_right
            shoulder_center = self.biomech.get_midpoint(
                frame.keypoints[11], frame.keypoints[12]
            )
            hip_center = self.biomech.get_midpoint(
                frame.keypoints[23], frame.keypoints[24]
            )
            neck = frame.keypoints[0]  # nose as reference for head
            
            # Calcular ángulo de la espalda
            back_angle = self.biomech.calculate_angle(
                neck, shoulder_center, hip_center
            )
            back_angles.append(back_angle)
        
        avg_angle = np.mean(back_angles)
        # Espalda recta debe estar entre 160-180 grados
        is_straight = 160 <= avg_angle <= 180
        score = max(0, min(1, (avg_angle - 160) / 20)) if avg_angle >= 160 else 0
        
        return MovementCheckpoint(
            checkpoint_name="straight_back",
            passed=is_straight,
            score=score,
            details=f"Ángulo promedio de espalda: {avg_angle:.1f}°"
        )
    
    def _check_open_chest(self, body_frames: List[BodyFrame]) -> MovementCheckpoint:
        """
        Verifica si el pecho está abierto (hombros hacia atrás)
        """
        chest_openness_scores = []
        
        for frame in body_frames:
            # Keypoints: 11-left_shoulder, 12-right_shoulder
            left_shoulder = frame.keypoints[11]
            right_shoulder = frame.keypoints[12]
            
            # Calcular la amplitud del pecho basada en la distancia entre hombros
            shoulder_width = self.biomech.calculate_distance(left_shoulder, right_shoulder)
            chest_openness_scores.append(shoulder_width)
        
        avg_openness = np.mean(chest_openness_scores)
        # Normalizar score basado en amplitud típica
        score = min(1.0, avg_openness / 0.3)  # 0.3 es una distancia típica normalizada
        is_open = score > 0.6
        
        return MovementCheckpoint(
            checkpoint_name="open_chest",
            passed=is_open,
            score=score,
            details=f"Amplitud de pecho: {avg_openness:.3f}"
        )
    
    def _check_hands_position(self, body_frames: List[BodyFrame]) -> MovementCheckpoint:
        """
        Verifica si las manos están cerca del pecho al impulsarse
        """
        hand_distances = []
        
        for frame in body_frames[:len(body_frames)//2]:  # Solo primeros frames (impulso)
            # Keypoints: 15-left_wrist, 16-right_wrist, 11-left_shoulder, 12-right_shoulder
            left_wrist = frame.keypoints[15]
            right_wrist = frame.keypoints[16]
            chest_center = self.biomech.get_midpoint(
                frame.keypoints[11], frame.keypoints[12]
            )
            
            # Distancia promedio de manos al pecho
            left_dist = self.biomech.calculate_distance(left_wrist, chest_center)
            right_dist = self.biomech.calculate_distance(right_wrist, chest_center)
            avg_dist = (left_dist + right_dist) / 2
            hand_distances.append(avg_dist)
        
        avg_distance = np.mean(hand_distances)
        # Las manos deben estar cerca del pecho (distancia < 0.2)
        is_near = avg_distance < 0.2
        score = max(0, 1 - (avg_distance / 0.2))
        
        return MovementCheckpoint(
            checkpoint_name="hands_near_chest",
            passed=is_near,
            score=score,
            details=f"Distancia promedio manos-pecho: {avg_distance:.3f}"
        )
    
    def _check_body_centering(self, body_frames: List[BodyFrame]) -> MovementCheckpoint:
        """
        Verifica si el cuerpo se mantiene centrado entre los pies
        """
        centering_scores = []
        
        for frame in body_frames:
            # Keypoints: 27-left_ankle, 28-right_ankle, torso center
            left_ankle = frame.keypoints[27]
            right_ankle = frame.keypoints[28]
            foot_center = self.biomech.get_midpoint(left_ankle, right_ankle)
            
            # Centro del torso
            torso_center = self.biomech.get_midpoint(
                frame.keypoints[11], frame.keypoints[12]  # shoulders
            )
            
            # Calcular desviación lateral
            lateral_deviation = abs(torso_center.x - foot_center.x)
            centering_scores.append(lateral_deviation)
        
        avg_deviation = np.mean(centering_scores)
        # El cuerpo debe mantenerse centrado (desviación < 0.1)
        is_centered = avg_deviation < 0.1
        score = max(0, 1 - (avg_deviation / 0.1))
        
        return MovementCheckpoint(
            checkpoint_name="body_centered",
            passed=is_centered,
            score=score,
            details=f"Desviación lateral promedio: {avg_deviation:.3f}"
        )
    
    def _check_timing(self, body_frames: List[BodyFrame]) -> MovementCheckpoint:
        """
        Verifica si el timing del takeoff es correcto (< 1.2 segundos)
        """
        duration = body_frames[-1].timestamp - body_frames[0].timestamp
        is_within_time = duration <= self.max_duration
        score = max(0, 1 - (duration / self.max_duration)) if duration > 0 else 1
        
        return MovementCheckpoint(
            checkpoint_name="timing_correct",
            passed=is_within_time,
            score=score,
            details=f"Duración: {duration:.2f}s (máximo: {self.max_duration}s)"
        )
    
    def _check_arm_positioning(
        self, 
        body_frames: List[BodyFrame], 
        direction: MovementDirection
    ) -> MovementCheckpoint:
        """
        Verifica el posicionamiento específico de brazos según la dirección
        """
        arm_scores = []
        
        for frame in body_frames[-len(body_frames)//2:]:  # Últimos frames (ya de pie)
            # Keypoints: 13-left_elbow, 14-right_elbow, 15-left_wrist, 16-right_wrist
            right_elbow = frame.keypoints[14]
            right_wrist = frame.keypoints[16]
            left_elbow = frame.keypoints[13]
            left_wrist = frame.keypoints[15]
            
            # Evaluar posicionamiento según dirección
            if direction == MovementDirection.FRONTSIDE:
                # Brazo derecho hacia adelante, izquierdo lateral
                right_forward = right_wrist.x > right_elbow.x
                left_lateral = abs(left_wrist.x - left_elbow.x) > 0.05
                score = (int(right_forward) + int(left_lateral)) / 2
            else:  # BACKSIDE
                # Posicionamiento para mayor visibilidad
                # Similar lógica pero con diferentes criterios
                right_position = right_wrist.y < right_elbow.y  # brazo más elevado
                left_position = left_wrist.x < left_elbow.x   # brazo hacia atrás
                score = (int(right_position) + int(left_position)) / 2
            
            arm_scores.append(score)
        
        avg_score = np.mean(arm_scores) if arm_scores else 0
        is_correct = avg_score > 0.6
        
        return MovementCheckpoint(
            checkpoint_name="arm_positioning",
            passed=is_correct,
            score=avg_score,
            details=f"Posicionamiento de brazos para {direction.value}: {avg_score:.2f}"
        )
    
    def _calculate_scores(
        self, 
        body_frames: List[BodyFrame], 
        checkpoints: Dict[str, MovementCheckpoint],
        duration: float
    ) -> Dict[str, float]:
        """
        Calcula los scores detallados del análisis
        """
        # Score de alineación corporal
        alignment_score = np.mean([
            checkpoints["straight_back"].score,
            checkpoints["open_chest"].score,
            checkpoints["body_centered"].score
        ]) * 100
        
        # Score de timing
        timing_score = checkpoints["timing_correct"].score * 100
        
        # Score de estabilidad (basado en variabilidad de posiciones)
        stability_score = self._calculate_stability_score(body_frames)
        
        # Score de técnica (posicionamiento específico)
        technique_score = np.mean([
            checkpoints["hands_near_chest"].score,
            checkpoints["arm_positioning"].score
        ]) * 100
        
        # Score general
        overall_score = np.mean([
            alignment_score, timing_score, stability_score, technique_score
        ])
        
        return {
            "body_alignment": alignment_score,
            "timing": timing_score,
            "stability": stability_score,
            "technique": technique_score,
            "overall": overall_score
        }
    
    def _calculate_stability_score(self, body_frames: List[BodyFrame]) -> float:
        """
        Calcula el score de estabilidad basado en la variabilidad del movimiento
        """
        if len(body_frames) < 3:
            return 100.0
        
        # Calcular variabilidad del centro de masa
        center_positions = []
        for frame in body_frames:
            center = self.biomech.get_body_center(frame.keypoints)
            center_positions.append([center.x, center.y])
        
        positions_array = np.array(center_positions)
        stability_variance = np.var(positions_array, axis=0).mean()
        
        # Convertir varianza a score (menos varianza = más estabilidad)
        stability_score = max(0, 100 - (stability_variance * 1000))
        return min(100, stability_score)
    
    def _generate_feedback(
        self, 
        checkpoints: Dict[str, MovementCheckpoint],
        scores: Dict[str, float],
        duration: float,
        direction: MovementDirection
    ) -> Dict[str, any]:
        """
        Genera feedback personalizado basado en el análisis
        """
        feedback_message = []
        recommendations = []
        
        # Feedback de timing
        if not checkpoints["timing_correct"].passed:
            feedback_message.append(f"Tu takeoff tomó {duration:.2f}s, intenta ser más rápido (máximo {self.max_duration}s)")
            recommendations.append("Practica el movimiento de popup más explosivo")
        
        # Feedback de postura
        if not checkpoints["straight_back"].passed:
            feedback_message.append("Mantén la espalda más recta durante el takeoff")
            recommendations.append("Enfócate en mantener el core activado y la columna alineada")
        
        if not checkpoints["open_chest"].passed:
            feedback_message.append("Abre más el pecho, lleva los hombros hacia atrás")
            recommendations.append("Practica ejercicios de apertura de pecho fuera del agua")
        
        # Feedback de técnica
        if not checkpoints["hands_near_chest"].passed:
            feedback_message.append("Coloca las manos más cerca del pecho al impulsarte")
            recommendations.append("Practica la posición correcta de las manos en tierra")
        
        if not checkpoints["body_centered"].passed:
            feedback_message.append("Mantén el cuerpo centrado entre los pies")
            recommendations.append("Trabaja en ejercicios de equilibrio y propiocepción")
        
        # Feedback específico por dirección
        if not checkpoints["arm_positioning"].passed:
            if direction == MovementDirection.FRONTSIDE:
                feedback_message.append("Brazo derecho hacia adelante, izquierdo como contrapeso lateral")
                recommendations.append("Practica la coordinación de brazos para takeoff frontside")
            else:
                feedback_message.append("Ajusta la posición de brazos para mejor visibilidad en backside")
                recommendations.append("Enfócate en la rotación del torso para mantener visión de la ola")
        
        # Mensaje de éxito
        if scores["overall"] >= 80:
            feedback_message.insert(0, "¡Excelente takeoff! Tu técnica es muy buena")
        elif scores["overall"] >= 70:
            feedback_message.insert(0, "Buen takeoff, con pequeños ajustes será perfecto")
        else:
            feedback_message.insert(0, "Tu takeoff necesita trabajo, pero estás en el camino correcto")
        
        return {
            "message": " ".join(feedback_message),
            "recommendations": recommendations,
            "priority_areas": self._get_priority_areas(scores)
        }
    
    def _get_priority_areas(self, scores: Dict[str, float]) -> List[str]:
        """
        Identifica las áreas prioritarias para mejorar
        """
        priority_areas = []
        
        if scores["timing"] < 60:
            priority_areas.append("Velocidad de ejecución")
        if scores["body_alignment"] < 60:
            priority_areas.append("Alineación corporal")
        if scores["stability"] < 60:
            priority_areas.append("Estabilidad y control")
        if scores["technique"] < 60:
            priority_areas.append("Técnica de brazos y manos")
        
        return priority_areas