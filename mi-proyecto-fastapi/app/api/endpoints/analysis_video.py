import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.movement_analysis_video import MovementType, MovementVideoAnalysisResponse
from app.services.video_analysis_service import analyze_video, get_biomechanical_profile

router = APIRouter()

@router.post("/analyze-video", response_model=MovementVideoAnalysisResponse)
async def analyze_movement_video(
    file: UploadFile = File(...),
    movement_type: MovementType = Form(...)
):
    """
    Analiza un video de surfing usando MediaPipe y evalúa la biomecánica 
    contra perfiles específicos para cada tipo de movimiento.
    """
    filename = None
    
    try:
        # Validar el archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó un archivo")
        
        # Validar extensión de video
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato de archivo no soportado. Formatos permitidos: {', '.join(allowed_extensions)}"
            )

        # Validar tamaño de archivo (máximo 100MB)
        file.file.seek(0, 2)  # Ir al final del archivo
        file_size = file.file.tell()
        file.file.seek(0)  # Volver al inicio
        
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo es demasiado grande. Tamaño máximo: {max_size // (1024*1024)}MB"
            )

        # Guardar archivo temporal
        file_id = str(uuid.uuid4())
        filename = f"videos/{file_id}_{file.filename}"
        os.makedirs("videos", exist_ok=True)
        
        try:
            with open(filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

        # Analizar el video usando el servicio de MediaPipe con análisis biomecánico detallado
        try:
            print(f"Iniciando análisis biomecánico de video: {filename} - Movimiento: {movement_type.value}")
            video_analysis = analyze_video(filename, movement_type)
            print(f"Análisis completado exitosamente")
            
            # Verificar que se recibieron datos del análisis
            if not video_analysis:
                raise HTTPException(status_code=500, detail="El análisis del video no retornó datos")
            
            # Log de métricas principales
            print(f"Frames procesados: {video_analysis.get('frames_analyzed', 0)}/{video_analysis.get('total_frames', 0)}")
            print(f"Puntaje de calidad: {video_analysis.get('quality_score', 0)}")
            
        except FileNotFoundError as e:
            if filename and os.path.exists(filename):
                os.remove(filename)
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {str(e)}")
        except ValueError as e:
            if filename and os.path.exists(filename):
                os.remove(filename)
            raise HTTPException(status_code=400, detail=f"Error en el formato del video: {str(e)}")
        except Exception as e:
            if filename and os.path.exists(filename):
                os.remove(filename)
            raise HTTPException(status_code=500, detail=f"Error en el análisis del video: {str(e)}")

        # Obtener perfil biomecánico detallado con valores específicos
        try:
            biomechanical_feedback = get_biomechanical_profile(movement_type)
            if not biomechanical_feedback:
                biomechanical_feedback = {
                    "General": "Perfil biomecánico no disponible para este movimiento"
                }
        except Exception as e:
            print(f"Error obteniendo perfil biomecánico: {str(e)}")
            biomechanical_feedback = {
                "General": "Error al cargar perfil biomecánico específico"
            }

        # Extraer datos del análisis con valores por defecto seguros
        detailed_analysis = video_analysis.get("detailed_analysis", {})
        measurements = detailed_analysis.get("measurements", {})
        deviations = detailed_analysis.get("deviations", {})
        recommendations = detailed_analysis.get("recommendations", [])
        phase_analysis = detailed_analysis.get("phase_analysis", [])

        # Formatear issues encontrados con mayor detalle
        issues_formatted = []
        raw_issues = video_analysis.get("issues_found", [])
        
        for issue in raw_issues:
            if isinstance(issue, str) and "dentro de parámetros aceptables" not in issue:
                issues_formatted.append({
                    "type": "biomechanical_deviation",
                    "description": issue,
                    "severity": determine_issue_severity(issue, deviations)
                })

        # Calcular métricas de procesamiento
        frames_processed = video_analysis.get("frames_analyzed", 0)
        total_frames = video_analysis.get("total_frames", 1)
        processing_rate = calculate_processing_rate(frames_processed, total_frames)

        # Resultado del análisis estructurado
        analysis_result = {
            "status": "success",
            "message": f"Análisis biomecánico completado para {movement_type.value}",
            "video_info": {
                "frames_processed": frames_processed,
                "total_frames": total_frames,
                "video_duration": round(video_analysis.get("video_duration", 0), 2),
                "fps": round(video_analysis.get("fps", 0), 1),
                "processing_rate": processing_rate
            },
            "biomechanical_analysis": {
                "movement_detected": video_analysis.get("movement_detected", movement_type.value),
                "quality_score": video_analysis.get("quality_score", 0),
                "measurements": format_measurements(measurements),
                "deviations": format_deviations(deviations),
                "phase_analysis": phase_analysis[:10] if phase_analysis else [],  # Limitar a 10 fases
                "pose_analysis_completed": True if frames_processed > 0 else False
            },
            "issues_detected": issues_formatted,
            "recommendations": recommendations if isinstance(recommendations, list) else [],
            "technical_details": {
                "algorithm": "MediaPipe Pose + Biomechanical Analysis",
                "confidence_threshold": 0.7,
                "smoothing_applied": True,
                "analysis_precision": "High"
            }
        }

        # Limpiar archivo temporal después del análisis exitoso
        try:
            if filename and os.path.exists(filename):
                os.remove(filename)
        except Exception as cleanup_error:
            print(f"Advertencia: No se pudo eliminar archivo temporal {filename}: {cleanup_error}")

        return MovementVideoAnalysisResponse(
            movement_type=movement_type,
            analysis_result=analysis_result,
            biomechanical_feedback=biomechanical_feedback,
            video_url=None  # No guardamos el archivo permanentemente
        )

    except HTTPException:
        raise
    except Exception as e:
        # Limpiar archivo en caso de error general
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


def determine_issue_severity(issue_description: str, deviations: dict) -> str:
    """
    Determina la severidad de un problema biomecánico basado en las desviaciones.
    """
    if not isinstance(issue_description, str):
        return "media"
        
    issue_lower = issue_description.lower()
    
    # Buscar desviaciones numéricas en la descripción
    severity_keywords = {
        "alta": ["muy", "excesivamente", "demasiado", "crítico", "grave"],
        "media": ["moderadamente", "ligeramente elevado", "notable"],
        "baja": ["levemente", "un poco", "mínimo"]
    }
    
    for severity, keywords in severity_keywords.items():
        if any(keyword in issue_lower for keyword in keywords):
            return severity
    
    # Evaluar basado en desviaciones numéricas si están disponibles
    if isinstance(deviations, dict):
        for deviation_data in deviations.values():
            if isinstance(deviation_data, dict) and 'deviation' in deviation_data:
                try:
                    deviation_value = float(deviation_data['deviation'])
                    if deviation_value > 20:
                        return "alta"
                    elif deviation_value > 10:
                        return "media"
                    elif deviation_value > 5:
                        return "baja"
                except (ValueError, TypeError):
                    continue
    
    return "media"  # Por defecto


def calculate_processing_rate(frames_processed: int, total_frames: int) -> float:
    """
    Calcula la tasa de procesamiento exitoso.
    """
    if total_frames == 0:
        return 0.0
    try:
        rate = (frames_processed / total_frames) * 100
        return round(rate, 1)
    except (ZeroDivisionError, TypeError):
        return 0.0


def format_measurements(measurements: dict) -> dict:
    """
    Formatea las mediciones para una presentación más clara.
    """
    if not isinstance(measurements, dict) or not measurements:
        return {}
    
    formatted = {}
    
    for key, value in measurements.items():
        try:
            if key == 'center_of_mass' and isinstance(value, dict):
                formatted[key] = {
                    "x": round(float(value.get('x', 0)), 3),
                    "y": round(float(value.get('y', 0)), 3),
                    "description": "Centro de masa corporal"
                }
            elif isinstance(value, (int, float)):
                # Formatear ángulos y mediciones
                if any(term in key for term in ['angle', 'flexion', 'inclination']):
                    formatted[key] = {
                        "value": round(float(value), 1),
                        "unit": "grados",
                        "description": get_measurement_description(key)
                    }
                elif 'ratio' in key:
                    formatted[key] = {
                        "value": round(float(value), 2),
                        "unit": "ratio",
                        "description": get_measurement_description(key)
                    }
                else:
                    formatted[key] = {
                        "value": round(float(value), 2),
                        "unit": "unidades",
                        "description": get_measurement_description(key)
                    }
            elif isinstance(value, dict):
                # Manejar diccionarios anidados
                formatted[key] = value
            else:
                # Convertir otros tipos a string
                formatted[key] = {
                    "value": str(value),
                    "unit": "unidades",
                    "description": get_measurement_description(key)
                }
        except (ValueError, TypeError) as e:
            print(f"Error formateando medición {key}: {e}")
            formatted[key] = {
                "value": "Error",
                "unit": "unidades",
                "description": get_measurement_description(key)
            }
    
    return formatted


def format_deviations(deviations: dict) -> dict:
    """
    Formatea las desviaciones para una presentación más clara.
    """
    if not isinstance(deviations, dict) or not deviations:
        return {}
    
    formatted = {}
    
    for key, deviation_data in deviations.items():
        try:
            if isinstance(deviation_data, dict):
                current_val = deviation_data.get('current', 0)
                deviation_val = deviation_data.get('deviation', 0)
                
                formatted[key] = {
                    "current_value": round(float(current_val), 1),
                    "target_range": deviation_data.get('target_range'),
                    "target_max": deviation_data.get('target_max'),
                    "deviation_amount": round(float(deviation_val), 1),
                    "description": get_measurement_description(key),
                    "severity": determine_deviation_severity(float(deviation_val))
                }
            else:
                # Manejar valores simples
                formatted[key] = {
                    "current_value": float(deviation_data) if isinstance(deviation_data, (int, float)) else 0,
                    "target_range": "No definido",
                    "target_max": None,
                    "deviation_amount": 0,
                    "description": get_measurement_description(key),
                    "severity": "baja"
                }
        except (ValueError, TypeError) as e:
            print(f"Error formateando desviación {key}: {e}")
            formatted[key] = {
                "current_value": 0,
                "target_range": "Error",
                "target_max": None,
                "deviation_amount": 0,
                "description": get_measurement_description(key),
                "severity": "baja"
            }
    
    return formatted


def determine_deviation_severity(deviation_value: float) -> str:
    """
    Determina la severidad de una desviación numérica.
    """
    try:
        deviation_val = float(deviation_value)
        if deviation_val > 25:
            return "alta"
        elif deviation_val > 15:
            return "media"
        elif deviation_val > 5:
            return "baja"
        else:
            return "mínima"
    except (ValueError, TypeError):
        return "media"


def get_measurement_description(measurement_key: str) -> str:
    """
    Proporciona descripción legible para cada tipo de medición.
    """
    descriptions = {
        "head_inclination": "Inclinación de la cabeza",
        "shoulder_width": "Apertura de hombros",
        "left_elbow_flexion": "Flexión del codo izquierdo",
        "right_elbow_flexion": "Flexión del codo derecho",
        "left_knee_flexion": "Flexión de rodilla izquierda",
        "right_knee_flexion": "Flexión de rodilla derecha",
        "body_inclination": "Inclinación corporal",
        "spine_curvature": "Curvatura de la columna",
        "center_of_mass": "Centro de masa corporal",
        "foot_separation_ratio": "Separación de pies (ratio)",
        "foot_separation": "Separación de pies",
        "hip_rotation": "Rotación de cadera",
        "knee_angle": "Ángulo de rodilla",
        "ankle_angle": "Ángulo de tobillo",
        "torso_angle": "Ángulo del torso"
    }
    
    return descriptions.get(measurement_key, measurement_key.replace('_', ' ').title())


@router.get("/movement-types")
async def get_available_movement_types():
    """
    Retorna los tipos de movimientos disponibles para análisis.
    """
    return {
        "available_movements": [
            {
                "type": MovementType.TAKE_OFF,
                "name": "Take Off",
                "description": "Análisis de la técnica de despegue y puesta en pie"
            },
            {
                "type": MovementType.BOTTOM_TURN,
                "name": "Bottom Turn",
                "description": "Análisis de la curva de fondo"
            }
        ],
        "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"],
        "max_file_size": "100MB",
        "analysis_features": [
            "Detección de pose con MediaPipe",
            "Análisis biomecánico detallado",
            "Evaluación contra perfiles ideales",
            "Recomendaciones específicas",
            "Puntaje de calidad técnica"
        ]
    }


@router.get("/biomechanical-profile/{movement_type}")
async def get_movement_biomechanical_profile(movement_type: MovementType):
    """
    Retorna el perfil biomecánico ideal para un tipo de movimiento específico.
    """
    try:
        profile = get_biomechanical_profile(movement_type)
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Perfil biomecánico no encontrado para {movement_type.value}"
            )
        return {
            "movement_type": movement_type,
            "biomechanical_profile": profile,
            "description": f"Perfil biomecánico ideal para {movement_type.value}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error al obtener perfil biomecánico: {str(e)}"
        )