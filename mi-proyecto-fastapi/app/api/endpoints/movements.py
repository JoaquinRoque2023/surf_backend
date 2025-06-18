from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import structlog

from app.schemas.movement import (
    MovementType, 
    SurfStance, 
    MovementAnalysisRequest,
    MovementAnalysisResponse
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("/types", response_model=List[Dict[str, Any]])
async def get_movement_types():
    """
    Obtener todos los tipos de movimientos disponibles con sus descripciones
    """
    movement_descriptions = {
        MovementType.TAKE_OFF: {
            "name": "Take Off",
            "description": "Ponerse de pie en la tabla",
            "level": "Principiante",
            "key_points": [
                "Espalda recta, sin encorvarse",
                "Pecho abierto (hombros hacia atrás)",
                "Manos cerca del pecho al impulsarse",
                "Cuerpo centrado entre los pies",
                "Máximo 1.2 segundos para pararse"
            ]
        },
        MovementType.BOTTOM_TURN: {
            "name": "Bottom Turn",
            "description": "Giro en la base de la ola",
            "level": "Intermedio",
            "key_points": [
                "Cuerpo inclinado sin perder equilibrio",
                "Rodillas dobladas para más potencia",
                "Rotación de torso y caderas",
                "Mantener base entre los pies"
            ]
        },
        MovementType.TOP_TURN: {
            "name": "Top Turn",
            "description": "Giro en la parte alta de la ola",
            "level": "Intermedio-Avanzado",
            "key_points": [
                "Extensión antes del giro",
                "Flexión durante el giro",
                "Rotación de torso y hombros",
                "Control del centro de masa"
            ]
        },
        MovementType.CUTBACK: {
            "name": "Cutback",
            "description": "Volver hacia la espuma",
            "level": "Avanzado",
            "key_points": [
                "Giro amplio y con buena curva",
                "Inclinación hacia el centro del giro",
                "Rotación coordinada torso-caderas",
                "Mantener equilibrio en la base"
            ]
        },
        MovementType.FLOATER: {
            "name": "Floater",
            "description": "Deslizarse por encima de la espuma",
            "level": "Intermedio-Avanzado",
            "key_points": [
                "Cuerpo recto y alineado",
                "Centro de equilibrio estable",
                "Flexión al aterrizar",
                "Control del centro de masa"
            ]
        }
        # Agregar más movimientos según necesidad
    }
    
    result = []
    for movement_type in MovementType:
        if movement_type in movement_descriptions:
            result.append({
                "id": movement_type.value,
                "type": movement_type,
                **movement_descriptions[movement_type]
            })
        else:
            result.append({
                "id": movement_type.value,
                "type": movement_type,
                "name": movement_type.value.replace("_", " ").title(),
                "description": f"Análisis de movimiento {movement_type.value}",
                "level": "Avanzado",
                "key_points": []
            })
    
    logger.info("Retrieved movement types", count=len(result))
    return result


@router.get("/types/{movement_type}")
async def get_movement_details(movement_type: MovementType):
    """
    Obtener detalles específicos de un tipo de movimiento
    """
    # Simulación de datos detallados por movimiento
    movement_details = {
        "type": movement_type,
        "biomechanical_profile": {},
        "common_errors": [],
        "tips": []
    }
    
    if movement_type == MovementType.TAKE_OFF:
        movement_details.update({
            "biomechanical_profile": {
                "arm_position": "Brazo derecho extendido al frente, izquierdo lateral",
                "core_activation": "Zona media activa para explosividad",
                "hip_extension": "Extensión de caderas y rodillas al pararse",
                "head_alignment": "Cabeza alineada, mirada al frente"
            },
            "common_errors": [
                "Espalda encorvada",
                "Manos muy separadas del pecho",
                "Demora excesiva (>1.2s)",
                "Pérdida del centro de equilibrio"
            ],
            "tips": [
                "Practica el movimiento en tierra",
                "Mantén el core activado",
                "Explosividad controlada",
                "Visión hacia adelante"
            ]
        })
    elif movement_type == MovementType.BOTTOM_TURN:
        movement_details.update({
            "biomechanical_profile": {
                "hip_position": "Cadera baja, rodillas flexionadas",
                "torso_inclination": "Inclinación lateral alineada con tabla",
                "weight_distribution": "Peso mayor en pierna trasera",
                "arm_coordination": "Brazo derecho guía el giro"
            },
            "common_errors": [
                "Pérdida de equilibrio por inclinación excesiva",
                "Rodillas rígidas",
                "Falta de rotación de torso",
                "Salirse de la base de apoyo"
            ],
            "tips": [
                "Inicia el giro con la rotación de torso",
                "Mantén rodillas flexibles",
                "Controla la inclinación progresivamente",
                "Usa los brazos para equilibrio"
            ]
        })
    
    logger.info("Retrieved movement details", movement_type=movement_type)
    return movement_details