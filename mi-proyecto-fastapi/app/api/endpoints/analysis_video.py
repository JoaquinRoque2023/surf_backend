import os
import uuid
import shutil
import aiofiles
import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect,Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import Request
from app.schemas.movement_analysis_video import MovementType, MovementVideoAnalysisResponse
from app.services.video_analysis_service import analyze_video, get_biomechanical_profile
from app.services.realtime_analysis_service import RealtimeAnalyzer
from pathlib import Path
import cv2
import base64
from typing import Dict, List, Optional

router = APIRouter()

# Almacén temporal para videos procesados
processed_videos: Dict[str, Dict] = {}

# Gestor de conexiones WebSocket para streaming en tiempo real
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.analyzers: Dict[str, RealtimeAnalyzer] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.analyzers[client_id] = RealtimeAnalyzer()

    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.remove(websocket)
        if client_id in self.analyzers:
            del self.analyzers[client_id]

    async def send_analysis_data(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))

manager = ConnectionManager()

@router.get("/video/analysis-overlay/{analysis_id}")
async def get_video_with_analysis_overlay(analysis_id: str):
    """
    Genera un video con overlay del análisis biomecánico (poses, mediciones, etc.)
    para un video ya analizado.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    video_info = processed_videos[analysis_id]
    video_path = video_info["filename"]
    poses_data = video_info.get("poses_data", [])
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
    
    # Generar video con overlay
    async def generate_video_with_overlay():
        cap = cv2.VideoCapture(video_path)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Obtener número de frame actual
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                
                # Aplicar overlay del análisis
                if frame_number < len(poses_data):
                    frame_with_overlay = apply_analysis_overlay(
                        frame, 
                        poses_data[frame_number],
                        video_info["movement_type"]
                    )
                else:
                    frame_with_overlay = frame
                
                # Codificar frame a JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_overlay, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Control de velocidad (opcional)
                await asyncio.sleep(0.033)  # ~30 FPS
                
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_video_with_overlay(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/video/play/{analysis_id}")
async def play_video(analysis_id: str):
    """
    Reproduce un video analizado por su ID con soporte para streaming.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video no encontrado")
    
    video_info = processed_videos[analysis_id]
    video_path = video_info["filename"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
    
    # Detectar el tipo de archivo para el media_type correcto
    file_extension = os.path.splitext(video_path)[1].lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.wmv': 'video/x-ms-wmv',
        '.m4v': 'video/mp4'
    }
    
    media_type = media_type_map.get(file_extension, 'video/mp4')
    
    return FileResponse(
        video_path,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={video_info['original_filename']}"
        }
    )

@router.get("/video/stream/{analysis_id}")
async def stream_video(analysis_id: str, request: Request):
    """
    Streaming de video con soporte para range requests (mejor para videos grandes).
    """
    if analysis_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video no encontrado")
    
    video_info = processed_videos[analysis_id]
    video_path = video_info["filename"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
    
    # Obtener información del archivo
    file_size = os.path.getsize(video_path)
    file_extension = os.path.splitext(video_path)[1].lower()
    
    media_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.wmv': 'video/x-ms-wmv',
        '.m4v': 'video/mp4'
    }
    
    media_type = media_type_map.get(file_extension, 'video/mp4')
    
    # Manejar range requests para streaming
    range_header = request.headers.get('range')
    
    if range_header:
        # Parsear el header Range
        try:
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Validar rangos
            if start >= file_size or end >= file_size:
                raise HTTPException(status_code=416, detail="Range not satisfiable")
            
            chunk_size = end - start + 1
            
            async def generate_chunk():
                async with aiofiles.open(video_path, 'rb') as file:
                    await file.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        chunk = await file.read(min(8192, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(chunk_size),
                'Content-Type': media_type,
            }
            
            return StreamingResponse(
                generate_chunk(),
                status_code=206,
                headers=headers
            )
        
        except (ValueError, IndexError):
            # Si hay error parseando el range, servir el archivo completo
            pass
    
    # Servir archivo completo
    async def generate_full_file():
        async with aiofiles.open(video_path, 'rb') as file:
            while chunk := await file.read(8192):
                yield chunk
    
    headers = {
        'Content-Length': str(file_size),
        'Accept-Ranges': 'bytes',
        'Content-Type': media_type,
    }
    
    return StreamingResponse(
        generate_full_file(),
        headers=headers
    )



@router.websocket("/ws/realtime-analysis/{movement_type}")
async def websocket_realtime_analysis(websocket: WebSocket, movement_type: str):
    """
    WebSocket para análisis en tiempo real de video streaming.
    Recibe frames del cliente y devuelve análisis biomecánico en tiempo real.
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    try:
        movement_enum = MovementType(movement_type)
        analyzer = manager.analyzers[client_id]
        
        while True:
            # Recibir frame del cliente (base64 encoded)
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            if 'frame' in frame_data:
                # Decodificar frame de base64
                frame_bytes = base64.b64decode(frame_data['frame'])
                
                # Convertir a imagen OpenCV
                import numpy as np
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Realizar análisis en tiempo real
                    analysis_result = analyzer.analyze_frame(frame, movement_enum)
                    
                    # Enviar resultado al cliente
                    response_data = {
                        "timestamp": asyncio.get_event_loop().time(),
                        "movement_type": movement_type,
                        "poses": analysis_result.get("poses", []),
                        "measurements": analysis_result.get("measurements", {}),
                        "quality_score": analysis_result.get("quality_score", 0),
                        "issues": analysis_result.get("issues", []),
                        "frame_analysis": analysis_result.get("frame_analysis", {})
                    }
                    
                    await manager.send_analysis_data(websocket, response_data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        manager.disconnect(websocket, client_id)



@router.get("/video/info/{analysis_id}")
async def get_video_info(analysis_id: str):
    """
    Obtiene información detallada del video.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video no encontrado")
    
    video_info = processed_videos[analysis_id]
    video_path = video_info["filename"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
    
    # Obtener información del archivo
    file_size = os.path.getsize(video_path)
    file_stats = os.stat(video_path)
    
    return {
        "analysis_id": analysis_id,
        "original_filename": video_info["original_filename"],
        "file_path": video_path,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "created_at": video_info["created_at"],
        "movement_type": video_info["movement_type"],
        "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "video_analysis": {
            "duration": video_info["analysis_data"].get("video_info", {}).get("video_duration", 0),
            "fps": video_info["analysis_data"].get("video_info", {}).get("fps", 0),
            "frames_processed": video_info["analysis_data"].get("video_info", {}).get("frames_processed", 0),
            "quality_score": video_info["analysis_data"].get("biomechanical_analysis", {}).get("quality_score", 0)
        }
    }


@router.delete("/video/{analysis_id}")
async def delete_video_analysis(analysis_id: str):
    """
    Elimina un video analizado y sus datos asociados.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    video_info = processed_videos[analysis_id]
    
    # Eliminar archivo de video
    try:
        if os.path.exists(video_info["filename"]):
            os.remove(video_info["filename"])
    except Exception as e:
        print(f"Error eliminando archivo de video: {e}")
    
    # Eliminar de memoria
    del processed_videos[analysis_id]
    
    return {"message": f"Análisis {analysis_id} eliminado exitosamente"}


def convert_to_mp4(input_path: str, output_path: str) -> str:
    """
    Convierte un video a formato MP4 para compatibilidad web.
    """
    try:
        import subprocess
        
        # Usar ffmpeg para conversión
        cmd = [
            'ffmpeg', '-i', input_path, 
            '-c:v', 'libx264', 
            '-c:a', 'aac', 
            '-movflags', '+faststart',  # Optimizar para streaming web
            '-y',  # Sobrescribir archivo existente
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return output_path
        else:
            print(f"Error en conversión ffmpeg: {result.stderr}")
            # Si falla la conversión, copiar archivo original
            shutil.copy2(input_path, output_path)
            return output_path
            
    except Exception as e:
        print(f"Error en conversión de video: {e}")
        # Fallback: copiar archivo original
        shutil.copy2(input_path, output_path)
        return output_path


def draw_pose_on_frame(frame, pose_data):
    """
    Dibuja las poses detectadas sobre el frame del video.
    """
    if not pose_data or 'landmarks' not in pose_data:
        return frame
    
    import cv2
    
    landmarks = pose_data['landmarks']
    height, width = frame.shape[:2]
    
    # Conexiones de pose de MediaPipe
    connections = [
        # Cuerpo superior
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Brazos
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Cadera
        # Piernas
        (23, 25), (24, 26), (25, 27), (26, 28),  # Muslos y pantorrillas
        (27, 29), (28, 30), (29, 31), (30, 32),  # Pies
    ]
    
    # Dibujar conexiones
    for start_idx, end_idx in connections:
        if (start_idx < len(landmarks) and end_idx < len(landmarks) and
            landmarks[start_idx] and landmarks[end_idx]):
            
            start_point = (
                int(landmarks[start_idx]['x'] * width),
                int(landmarks[start_idx]['y'] * height)
            )
            end_point = (
                int(landmarks[end_idx]['x'] * width),
                int(landmarks[end_idx]['y'] * height)
            )
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Dibujar puntos de landmarks
    for i, landmark in enumerate(landmarks):
        if landmark and landmark.get('visibility', 0) > 0.5:
            x = int(landmark['x'] * width)
            y = int(landmark['y'] * height)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    
    return frame


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

        # Generar ID único para este análisis
        analysis_id = str(uuid.uuid4())
        
        # Guardar archivo temporal con el analysis_id
        file_id = str(uuid.uuid4())
        filename = f"videos/{analysis_id}_{file.filename}"
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
            print(f"Análisis completado exitosamente con ID: {analysis_id}")
            
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

        # Extraer datos de poses si están disponibles
        poses_data = video_analysis.get("poses_data", [])
        if not isinstance(poses_data, list):
            poses_data = []

        # Generar ID y almacenar en processed_videos
        processed_videos[analysis_id] = {
            "filename": filename,  # Mantener el archivo para consultas posteriores
            "original_filename": file.filename,
            "movement_type": movement_type.value,
            "created_at": datetime.now().isoformat(),
            "analysis_data": analysis_result,
            "biomechanical_feedback": biomechanical_feedback,
            "poses_data": poses_data,  # Agregar datos de poses si están disponibles
            "file_size": file_size,
            "status": "completed"
        }

        print(f"Análisis almacenado en memoria con ID: {analysis_id}")

        # Modificar el return para incluir el analysis_id
        return MovementVideoAnalysisResponse(
            analysis_id=analysis_id,  # ← AGREGAR ESTE CAMPO
            movement_type=movement_type,
            analysis_result=analysis_result,
            biomechanical_feedback=biomechanical_feedback,
            video_url=None
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


# Nuevo endpoint para consultar análisis por ID
@router.get("/analysis/{analysis_id}", response_model=dict)
async def get_analysis_by_id(analysis_id: str):
    """
    Obtiene un análisis específico por su ID.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(
            status_code=404, 
            detail=f"Análisis con ID {analysis_id} no encontrado"
        )
    
    analysis_data = processed_videos[analysis_id]
    
    return {
        "analysis_id": analysis_id,
        "movement_type": analysis_data["movement_type"],
        "original_filename": analysis_data["original_filename"],
        "created_at": analysis_data["created_at"],
        "status": analysis_data["status"],
        "analysis_result": analysis_data["analysis_data"],
        "biomechanical_feedback": analysis_data["biomechanical_feedback"],
        "file_info": {
            "size": analysis_data["file_size"],
            "has_poses_data": len(analysis_data["poses_data"]) > 0
        }
    }


# Nuevo endpoint para listar todos los análisis
@router.get("/analysis", response_model=dict)
async def list_analyses(limit: int = 10, offset: int = 0):
    """
    Lista todos los análisis almacenados con paginación.
    """
    analyses_list = []
    
    # Convertir el diccionario a lista ordenada por fecha de creación
    sorted_analyses = sorted(
        processed_videos.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    # Aplicar paginación
    paginated_analyses = sorted_analyses[offset:offset + limit]
    
    for analysis_id, data in paginated_analyses:
        analyses_list.append({
            "analysis_id": analysis_id,
            "movement_type": data["movement_type"],
            "original_filename": data["original_filename"],
            "created_at": data["created_at"],
            "status": data["status"],
            "quality_score": data["analysis_data"].get("biomechanical_analysis", {}).get("quality_score", 0)
        })
    
    return {
        "analyses": analyses_list,
        "total": len(processed_videos),
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < len(processed_videos)
    }


# Nuevo endpoint para eliminar un análisis
@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Elimina un análisis específico y su archivo asociado.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(
            status_code=404,
            detail=f"Análisis con ID {analysis_id} no encontrado"
        )
    
    analysis_data = processed_videos[analysis_id]
    filename = analysis_data.get("filename")
    
    # Eliminar archivo si existe
    if filename and os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Archivo eliminado: {filename}")
        except Exception as e:
            print(f"Error al eliminar archivo {filename}: {e}")
    
    # Eliminar de la memoria
    del processed_videos[analysis_id]
    
    return {
        "message": f"Análisis {analysis_id} eliminado exitosamente",
        "analysis_id": analysis_id
    }


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
            },
            {
                "type": "top_turn",
                "name": "Top Turn",
                "description": "Análisis del giro en la parte superior de la ola"
            },
            {
                "type": "cutback",
                "name": "Cutback",
                "description": "Análisis del giro de retorno hacia la espuma"
            },
            {
                "type": "floater",
                "name": "Floater",
                "description": "Análisis de la maniobra sobre la cresta de la ola"
            },
            {
                "type": "reentry",
                "name": "Reentry",
                "description": "Análisis de la reentrada en la ola"
            },
            {
                "type": "snap",
                "name": "Snap",
                "description": "Análisis del giro rápido y explosivo"
            },
              {
                "type": "carving",
                "name": "Carving",
                "description": "Análisis del giro amplio y fluido"
            },
            {
                "type": "roundhouse_cutback",
                "name": "Roundhouse Cutback",
                "description": "Análisis del cutback circular"
            },
            {
                "type": "foam_climb",
                "name": "Foam Climb",
                "description": "Análisis de la subida por la espuma"
            },
            {
                "type": "closeout_reentry",
                "name": "Closeout Reentry",
                "description": "Análisis de la reentrada en cierre de ola"
            },
            {
                "type": "tail_slide",
                "name": "Tail Slide",
                "description": "Análisis del deslizamiento de la cola"
            },
            {
                "type": "agarrar_velocidad",
                "name": "Agarrar Velocidad",
                "description": "Análisis de la generación de velocidad (riel a riel)"
            },
            {
                "type": "velocidad_horizontal",
                "name": "Velocidad Horizontal",
                "description": "Análisis de la velocidad y desplazamiento horizontal"
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
    
    
@router.get("/video/download/{analysis_id}")
async def download_video(analysis_id: str):
    """
    Descarga el video original analizado.
    """
    if analysis_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video no encontrado")
    
    video_info = processed_videos[analysis_id]
    video_path = video_info["filename"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
    
    # Detectar el tipo de archivo para el media_type correcto
    file_extension = os.path.splitext(video_path)[1].lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.wmv': 'video/x-ms-wmv',
        '.m4v': 'video/mp4'
    }
    
    media_type = media_type_map.get(file_extension, 'application/octet-stream')
    
    return FileResponse(
        video_path,
        media_type=media_type,
        filename=video_info["original_filename"],
        headers={
            "Content-Disposition": f"attachment; filename={video_info['original_filename']}"
        }
    )