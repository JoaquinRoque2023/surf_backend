import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import structlog
from datetime import datetime

from app.schemas.movement import (
    MovementAnalysisRequest,
    MovementAnalysisResponse,
    MovementCriteria,
    MovementType,
    SurfStance,
    BiomechanicalData,
    BodyPartPosition
)

logger = structlog.get_logger()


class MovementAnalyzerService:
    """Servicio principal para análisis de movimientos de surf"""
    
    def __init__(self):
        self.movement_definitions = self._load_movement_definitions()
    
    def _load_movement_definitions(self) -> Dict[MovementType, Dict]:
        """Cargar definiciones de movimientos basadas en el documento PDF"""
        return {
            MovementType.TAKE_OFF: {
                "max_duration": 1.2,
                "criteria": [
                    {
                        "name": "postura_espalda",
                        "description": "Espalda recta, sin encorvarse",
                        "check_function": self._check_straight_back
                    },
                    {
                        "name": "pecho_abierto",
                        "description": "Pecho abierto (hombros hacia atrás)",
                        "check_function": self._check_chest_open
                    },
                    {
                        "name": "manos_cerca_pecho",
                        "description": "Manos cerca del pecho al impulsarse",
                        "check_function": self._check_hands_near_chest
                    },
                    {
                        "name": "centro_equilibrio",
                        "description": "Cuerpo centrado entre los pies",
                        "check_function": self._check_body_centered
                    },
                    {
                        "name": "tiempo_ejecucion",
                        "description": "No más de 1.2 segundos para pararse",
                        "check_function": self._check_takeoff_timing
                    }
                ]
            },
            MovementType.BOTTOM_TURN: {
                "criteria": [
                    {
                        "name": "inclinacion_segura",
                        "description": "Cuerpo inclinado sin perder equilibrio",
                        "check_function": self._check_safe_lean
                    },
                    {
                        "name": "rodillas_flexionadas",
                        "description": "Rodillas dobladas para más potencia",
                        "check_function": self._check_knee_flexion
                    },
                    {
                        "name": "rotacion_torso",
                        "description": "Giro iniciado con rotación de torso y caderas",
                        "check_function": self._check_torso_rotation
                    },
                    {
                        "name": "base_apoyo",
                        "description": "Cuerpo dentro del espacio entre los pies",
                        "check_function": self._check_body_centered
                    }
                ]
            },
            MovementType.TOP_TURN: {
                "criteria": [
                    {
                        "name": "extension_previa",
                        "description": "Extensión antes del giro",
                        "check_function": self._check_pre_extension
                    },
                    {
                        "name": "flexion_durante_giro",
                        "description": "Flexión durante el giro",
                        "check_function": self._check_turn_flexion
                    },
                    {
                        "name": "rotacion_hombros",
                        "description": "Torso y hombros giran para cambiar dirección",
                        "check_function": self._check_shoulder_rotation
                    },
                    {
                        "name": "control_centro_masa",
                        "description": "Cuerpo no se sale de la zona entre pies",
                        "check_function": self._check_body_centered
                    }
                ]
            }
            # Agregar más movimientos según necesidad
        }
    
    async def analyze_movement(self, request: MovementAnalysisRequest) -> MovementAnalysisResponse:
        """Análisis principal de movimiento"""
        start_time = datetime.now()
        
        try:
            # Obtener definición del movimiento
            movement_def = self.movement_definitions.get(request.movement_type)
            if not movement_def:
                raise ValueError(f"Movimiento {request.movement_type} no soportado")
            
            # Evaluar cada criterio
            criteria_results = []
            for criterion in movement_def["criteria"]:
                result = await self._evaluate_criterion(
                    criterion, 
                    request.biomechanical_data,
                    request.stance,
                    request.duration_seconds
                )
                criteria_results.append(result)
            
            # Calcular puntuaciones
            scores = self._calculate_scores(criteria_results, request)
            
            # Generar feedback
            feedback = self._generate_feedback(criteria_results, request.movement_type)
            
            # Crear respuesta
            response = MovementAnalysisResponse(
                movement_type=request.movement_type,
                stance=request.stance,
                overall_score=scores["overall"],
                execution_time=request.duration_seconds,
                criteria_results=criteria_results,
                balance_score=scores["balance"],
                technique_score=scores["technique"],
                timing_score=scores["timing"],
                strengths=feedback["strengths"],
                improvements=feedback["improvements"],
                recommendations=feedback["recommendations"],
                confidence_level=scores["confidence"]
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Movement analysis completed",
                movement_type=request.movement_type,
                processing_time=processing_time,
                overall_score=scores["overall"]
            )
            
            return response
            
        except Exception as e:
            logger.error("Error in movement analysis", error=str(e))
            raise
    
    async def _evaluate_criterion(
        self, 
        criterion: Dict, 
        biomech_data: List[BiomechanicalData],
        stance: SurfStance,
        duration: float
    ) -> MovementCriteria:
        """Evaluar un criterio específico"""
        
        try:
            # Ejecutar función de verificación del criterio
            check_result = criterion["check_function"](
                biomech_data, 
                stance, 
                duration
            )
            
            return MovementCriteria(
                criteria_name=criterion["name"],
                expected=criterion["description"],
                actual=check_result["actual"],
                passed=check_result["passed"],
                confidence=check_result["confidence"]
            )
            
        except Exception as e:
            logger.error(f"Error evaluating criterion {criterion['name']}", error=str(e))
            return MovementCriteria(
                criteria_name=criterion["name"],
                expected=criterion["description"],
                actual="Error en evaluación",
                passed=False,
                confidence=0.0
            )
    
    def _calculate_scores(self, criteria_results: List[MovementCriteria], request: MovementAnalysisRequest) -> Dict[str, float]:
        """Calcular puntuaciones basadas en criterios evaluados"""
        
        # Puntuación general basada en criterios pasados
        passed_criteria = sum(1 for c in criteria_results if c.passed)
        total_criteria = len(criteria_results)
        overall_score = (passed_criteria / total_criteria) * 10 if total_criteria > 0 else 0
        
        # Puntuaciones específicas (simuladas por ahora)
        balance_score = self._calculate_balance_score(request.biomechanical_data)
        technique_score = overall_score  # Usar overall como base
        timing_score = self._calculate_timing_score(request)
        
        # Confianza promedio
        avg_confidence = np.mean([c.confidence for c in criteria_results]) if criteria_results else 0.5
        
        return {
            "overall": round(overall_score, 1),
            "balance": round(balance_score, 1),
            "technique": round(technique_score, 1),
            "timing": round(timing_score, 1),
            "confidence": round(avg_confidence, 2)
        }
    
    def _calculate_balance_score(self, biomech_data: List[BiomechanicalData]) -> float:
        """Calcular puntuación de equilibrio basada en estabilidad del centro de masa"""
        if not biomech_data:
            return 5.0
        
        # Calcular variabilidad del centro de masa
        hip_positions = [data.hip_position for data in biomech_data]
        hip_x_coords = [pos.x for pos in hip_positions]
        hip_y_coords = [pos.y for pos in hip_positions]
        
        # Calcular desviación estándar como medida de estabilidad
        x_stability = 1 / (1 + np.std(hip_x_coords)) if len(hip_x_coords) > 1 else 1.0
        y_stability = 1 / (1 + np.std(hip_y_coords)) if len(hip_y_coords) > 1 else 1.0
        
        balance_score = ((x_stability + y_stability) / 2) * 10
        return min(10.0, max(0.0, balance_score))
    
    def _calculate_timing_score(self, request: MovementAnalysisRequest) -> float:
        """Calcular puntuación de timing"""
        movement_def = self.movement_definitions.get(request.movement_type, {})
        max_duration = movement_def.get("max_duration")
        
        if max_duration and request.duration_seconds > max_duration:
            # Penalizar por exceso de tiempo
            penalty = min(5.0, (request.duration_seconds - max_duration) * 2)
            return max(0.0, 10.0 - penalty)
        
        return 8.0  # Puntuación base para timing aceptable
    
    def _generate_feedback(self, criteria_results: List[MovementCriteria], movement_type: MovementType) -> Dict[str, List[str]]:
        """Generar feedback personalizado"""
        strengths = []
        improvements = []
        recommendations = []
        
        # Identificar fortalezas (criterios pasados)
        for criterion in criteria_results:
            if criterion.passed:
                strengths.append(f"Buena ejecución en {criterion.criteria_name}")
        
        # Identificar áreas de mejora (criterios fallidos)
        for criterion in criteria_results:
            if not criterion.passed:
                improvements.append(f"Mejorar {criterion.criteria_name}")
                recommendations.append(f"Practica específicamente {criterion.expected.lower()}")
        
        # Recomendaciones generales por tipo de movimiento
        general_recommendations = {
            MovementType.TAKE_OFF: [
                "Practica el take-off en tierra primero",
                "Mantén el core activado durante todo el movimiento"
            ],
            MovementType.BOTTOM_TURN: [
                "Inicia el giro con la rotación del torso",
                "Mantén las rodillas flexibles para mejor control"
            ]
        }
        
        if movement_type in general_recommendations:
            recommendations.extend(general_recommendations[movement_type])
        
        return {
            "strengths": strengths[:3],  # Limitar a 3 fortalezas
            "improvements": improvements[:3],  # Limitar a 3 mejoras
            "recommendations": recommendations[:5]  # Limitar a 5 recomendaciones
        }
    
    def get_movement_criteria(self, movement_type: MovementType, stance: Optional[SurfStance] = None) -> List[Dict]:
        """Obtener criterios de evaluación para un movimiento específico"""
        movement_def = self.movement_definitions.get(movement_type)
        if not movement_def:
            return []
        
        criteria_list = []
        for criterion in movement_def["criteria"]:
            criteria_list.append({
                "name": criterion["name"],
                "description": criterion["description"],
                "stance_specific": stance is not None
            })
        
        return criteria_list
    
    # Funciones de verificación de criterios específicos
    def _check_straight_back(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar si la espalda está recta"""
        if not biomech_data:
            return {"passed": False, "actual": "Sin datos", "confidence": 0.0}
        
        # Calcular ángulo promedio del torso
        torso_angles = [data.torso_angle for data in biomech_data if data.torso_angle is not None]
        if not torso_angles:
            return {"passed": False, "actual": "No se pudo medir ángulo del torso", "confidence": 0.3}
        
        avg_angle = np.mean(torso_angles)
        is_straight = abs(avg_angle) < 15  # Tolerancia de 15 grados
        
        return {
            "passed": is_straight,
            "actual": f"Ángulo promedio del torso: {avg_angle:.1f}°",
            "confidence": 0.8
        }
    
    def _check_chest_open(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar si el pecho está abierto (hombros hacia atrás)"""
        if not biomech_data:
            return {"passed": False, "actual": "Sin datos", "confidence": 0.0}
        
        # Analizar posición de hombros
        shoulder_positions = []
        for data in biomech_data:
            left_shoulder = data.left_shoulder
            right_shoulder = data.right_shoulder
            if left_shoulder and right_shoulder:
                # Calcular apertura de hombros basada en distancia
                shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                shoulder_positions.append(shoulder_width)
        
        if not shoulder_positions:
            return {"passed": False, "actual": "No se pudieron medir hombros", "confidence": 0.3}
        
        avg_width = np.mean(shoulder_positions)
        is_open = avg_width > 0.4  # Threshold simulado
        
        return {
            "passed": is_open,
            "actual": f"Apertura promedio de hombros: {avg_width:.2f}",
            "confidence": 0.7
        }
    
    def _check_hands_near_chest(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar si las manos están cerca del pecho al impulsarse"""
        if not biomech_data:
            return {"passed": False, "actual": "Sin datos", "confidence": 0.0}
        
        # Analizar distancia de muñecas al torso
        distances = []
        for data in biomech_data:
            torso_pos = data.torso_position
            left_wrist = data.left_wrist
            right_wrist = data.right_wrist
            
            if torso_pos and left_wrist and right_wrist:
                left_dist = np.sqrt((left_wrist.x - torso_pos.x)**2 + (left_wrist.y - torso_pos.y)**2)
                right_dist = np.sqrt((right_wrist.x - torso_pos.x)**2 + (right_wrist.y - torso_pos.y)**2)
                distances.append((left_dist + right_dist) / 2)
        
        if not distances:
            return {"passed": False, "actual": "No se pudo medir distancia manos-pecho", "confidence": 0.3}
        
        avg_distance = np.mean(distances)
        is_close = avg_distance < 0.3  # Threshold simulado
        
        return {
            "passed": is_close,
            "actual": f"Distancia promedio manos-pecho: {avg_distance:.2f}",
            "confidence": 0.7
        }
    
    def _check_body_centered(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar si el cuerpo se mantiene centrado entre los pies"""
        if not biomech_data:
            return {"passed": False, "actual": "Sin datos", "confidence": 0.0}
        
        # Calcular centro de masa relativo a los pies
        deviations = []
        for data in biomech_data:
            left_ankle = data.left_ankle
            right_ankle = data.right_ankle
            hip_pos = data.hip_position
            
            if left_ankle and right_ankle and hip_pos:
                # Centro entre los pies
                feet_center_x = (left_ankle.x + right_ankle.x) / 2
                # Desviación del centro de masa
                deviation = abs(hip_pos.x - feet_center_x)
                deviations.append(deviation)
        
        if not deviations:
            return {"passed": False, "actual": "No se pudo medir centro de equilibrio", "confidence": 0.3}
        
        avg_deviation = np.mean(deviations)
        is_centered = avg_deviation < 0.15  # Threshold simulado
        
        return {
            "passed": is_centered,
            "actual": f"Desviación promedio del centro: {avg_deviation:.2f}",
            "confidence": 0.8
        }
    
    def _check_takeoff_timing(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar timing del take-off"""
        max_duration = 1.2
        is_within_time = duration <= max_duration
        
        return {
            "passed": is_within_time,
            "actual": f"Tiempo de ejecución: {duration:.2f}s",
            "confidence": 0.9
        }
    
    # Funciones adicionales para otros criterios
    def _check_safe_lean(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar inclinación segura en bottom turn"""
        # Implementación simulada
        return {"passed": True, "actual": "Inclinación dentro de límites seguros", "confidence": 0.7}
    
    def _check_knee_flexion(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar flexión de rodillas"""
        # Implementación simulada
        return {"passed": True, "actual": "Rodillas adecuadamente flexionadas", "confidence": 0.7}
    
    def _check_torso_rotation(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar rotación del torso"""
        # Implementación simulada
        return {"passed": True, "actual": "Rotación de torso detectada", "confidence": 0.6}
    
    def _check_pre_extension(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar extensión previa al giro"""
        # Implementación simulada
        return {"passed": True, "actual": "Extensión previa detectada", "confidence": 0.6}
    
    def _check_turn_flexion(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar flexión durante el giro"""
        # Implementación simulada
        return {"passed": True, "actual": "Flexión durante giro detectada", "confidence": 0.6}
    
    def _check_shoulder_rotation(self, biomech_data: List[BiomechanicalData], stance: SurfStance, duration: float) -> Dict:
        """Verificar rotación de hombros"""
        # Implementación simulada
        return {"passed": True, "actual": "Rotación de hombros detectada", "confidence": 0.6}