from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.movement_analysis_video import MovementType, MovementVideoAnalysisResponse
from app.services.video_analysis_service import analyze_video  # Importar el servicio
import shutil
import os
import uuid

router = APIRouter()

@router.post("/analyze-video", response_model=MovementVideoAnalysisResponse)
async def analyze_movement_video(
    file: UploadFile = File(...),
    movement_type: MovementType = Form(...)
):
    try:
        # Validar el archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó un archivo")
        
        # Validar extensión de video
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato de archivo no soportado. Formatos permitidos: {', '.join(allowed_extensions)}"
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

        # Analizar el video usando el servicio de MediaPipe
        try:
            print(f"Iniciando análisis de video: {filename}")
            video_analysis = analyze_video(filename, movement_type)
            print(f"Análisis completado: {video_analysis}")
        except Exception as e:
            # Limpiar archivo si falla el análisis
            if os.path.exists(filename):
                os.remove(filename)
            raise HTTPException(status_code=500, detail=f"Error en el análisis del video: {str(e)}")

        # Obtener perfil biomecánico
        biomechanical_feedback = get_biomechanical_profile(movement_type)

        # Combinar resultados del análisis real con el perfil biomecánico
        analysis_result = {
            "status": "success",
            "message": f"Video analizado correctamente para movimiento {movement_type.value}",
            "frames_processed": video_analysis.get("frames_analyzed", 0),
            "total_frames": video_analysis.get("total_frames", 0),
            "video_duration": video_analysis.get("video_duration", 0),
            "fps": video_analysis.get("fps", 0),
            "movement_detected": video_analysis.get("movement_detected", movement_type.value),
            "issues_detected": video_analysis.get("issues_found", []),
            "pose_analysis_completed": True
        }

        # TODO: Guardar en base de datos si es necesario
        # save_to_database(user_id, movement_type, filename, analysis_result, biomechanical_feedback)

        return MovementVideoAnalysisResponse(
            movement_type=movement_type,
            analysis_result=analysis_result,
            biomechanical_feedback=biomechanical_feedback,
            video_url=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        # Limpiar archivo en caso de error general
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


def get_biomechanical_profile(movement_type: MovementType) -> dict:
    """
    Extrae los puntos biomecánicos por tipo de movimiento.
    """
    profiles = {
        MovementType.TAKE_OFF: {
            "Cabeza/Cuello": "Mirada al frente, inclinación 20–30°",
            "Hombros": "Abiertos 80°–100°, escápulas retraídas",
            "Rodillas": "Semiflexión 90°–110°",
            "Cadera": "Posición neutra, activación del core",
            "Pies": "Separación ancho de hombros, distribución de peso equilibrada"
        },
        MovementType.BOTTOM_TURN: {
            "Cabeza/Cuello": "Giro 40°–50° hacia la curva",
            "Rodillas": "Delantera 100°–110°; Trasera 120°–130°",
            "Cadera": "Rotación hacia el interior de la curva",
            "Hombros": "Alineación con la dirección del giro",
            "Brazos": "Extensión para mantener equilibrio"
        },
        # Agregar más movimientos según sea necesario
    }
    
    return profiles.get(movement_type, {
        "General": "Perfil biomecánico no disponible para este movimiento"
    })