from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import structlog
from app.schemas.movement_analysis_video import MovementType as VideoMovementType

from app.schemas.movement import (
    MovementType
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
        },
        MovementType.REENTRY: {
            "name": "Reentry",
            "description": "Reentrada en la ola tras un giro",
            "level": "Intermedio",
            "key_points": [
                "Giro rápido de cabeza",
                "Extensión y compresión de rodillas",
                "Aterrizaje equilibrado",
                "Control de brazos"
            ]
        },
        MovementType.SNAP: {
            "name": "Snap",
            "description": "Giro brusco y corto en la cresta de la ola",
            "level": "Avanzado",
            "key_points": [
                "Explosividad en el giro",
                "Transferencia rápida de peso",
                "Control de la tabla en el borde",
                "Recuperación rápida tras el giro"
            ]
        },
        MovementType.CARVING: {
            "name": "Carving",
            "description": "Giro amplio y fluido",
            "level": "Intermedio",
            "key_points": [
                "Giro leve de cabeza",
                "Flexión de hombros",
                "Rotación de caderas",
                "Distribución de peso 60/40"
            ]
        },
        MovementType.ROUNDHOUSE_CUTBACK: {
            "name": "Roundhouse Cutback",
            "description": "Cutback circular",
            "level": "Avanzado",
            "key_points": [
                "Giro amplio de cabeza",
                "Torsión sostenida de torso",
                "Compresión y extensión de caderas",
                "Pivote sincronizado"
            ]
        },
        MovementType.FOAM_CLIMB: {
            "name": "Foam Climb",
            "description": "Subir por la espuma de la ola",
            "level": "Intermedio",
            "key_points": [
                "Flexión de rodillas al subir",
                "Centro de masa bajo",
                "Mirada hacia adelante",
                "Control de la velocidad al descender"
            ]
        },
        MovementType.CLOSEOUT_REENTRY: {
            "name": "Closeout Reentry",
            "description": "Reentrada en secciones cerradas de la ola",
            "level": "Avanzado",
            "key_points": [
                "Sincronización precisa",
                "Extensión rápida al impactar",
                "Aterrizaje controlado",
                "Preparación para posible caída"
            ]
        },
        MovementType.TAIL_SLIDE: {
            "name": "Tail Slide",
            "description": "Deslizamiento del tail de la tabla",
            "level": "Avanzado",
            "key_points": [
                "Transferencia de peso al tail",
                "Rotación explosiva de caderas",
                "Control de la tabla en el giro",
                "Recuperación tras el deslizamiento"
            ]
        },
        MovementType.AGARRAR_VELOCIDAD: {
            "name": "Agarrar Velocidad",
            "description": "Generar velocidad bombeando la tabla",
            "level": "Principiante-Intermedio",
            "key_points": [
                "Flexión y extensión alternada de piernas",
                "Movimientos coordinados de brazos",
                "Mantener el centro de masa bajo",
                "Sincronización con la ola"
            ]
        },
        MovementType.VELOCIDAD_HORIZONTAL: {
            "name": "Velocidad Horizontal",
            "description": "Mantener velocidad a lo largo de la ola",
            "level": "Principiante",
            "key_points": [
                "Posición estable sobre la tabla",
                "Distribución uniforme del peso",
                "Mirada hacia la dirección del recorrido",
                "Ajuste de postura según la sección de la ola"
            ]
        }
        # Agregar más movimientos según necesidad
    }

    # Integrar movimientos adicionales desde movement_analysis_video si existen
    if hasattr(VideoMovementType, "__members__"):
        for video_movement in VideoMovementType:
            if video_movement not in movement_descriptions:
                movement_descriptions[video_movement] = {
                    "name": video_movement.value.replace("_", " ").title(),
                    "description": f"Análisis de movimiento {video_movement.value}",
                    "level": "Avanzado",
                    "key_points": []
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