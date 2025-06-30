import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple,Any
from dataclasses import dataclass
from app.schemas.movement_analysis_video import MovementType
from app.utils.biomechanics import BiomechanicsCalculator, BodyKeypoint



mp_pose = mp.solutions.pose

@dataclass
class BiomechanicalProfile:
    """Perfil biomecánico con valores numéricos específicos"""
    head_inclination_range: Tuple[float, float]  # (min, max) grados
    shoulder_width_range: Tuple[float, float]    # (min, max) grados de apertura
    elbow_flexion_range: Tuple[float, float]     # (min, max) grados
    knee_flexion_range: Tuple[float, float]      # (min, max) grados
    hip_rotation_range: Tuple[float, float]      # (min, max) grados
    spine_curvature_max: float                   # máximo grados de curvatura
    center_of_mass_tolerance: float              # tolerancia en posición
    foot_separation_range: Tuple[float, float]   # (min, max) ratio ancho hombros


class SurfBiomechanicsAnalyzer:
    """Analizador biomecánico específico para surf"""
    
    def __init__(self):
        self.calculator = BiomechanicsCalculator()
        self.profiles = self._initialize_biomechanical_profiles()
    
    def _initialize_biomechanical_profiles(self) -> Dict[MovementType, BiomechanicalProfile]:
        """Inicializa los perfiles biomecánicos basados en la imagen"""
        return {
            MovementType.TAKE_OFF: BiomechanicalProfile(
                head_inclination_range=(20, 30),      # Inclinación 20-30°
                shoulder_width_range=(80, 100),        # Abiertos 80-100°
                elbow_flexion_range=(50, 90),          # Flexión 50-90° (pegados al torso)
                knee_flexion_range=(90, 110),          # Semiflexión 90-110°
                hip_rotation_range=(-5, 5),            # Posición neutra ±5°
                spine_curvature_max=15,                # Pecho elevado, sin curvar
                center_of_mass_tolerance=0.1,          # Centrado
                foot_separation_range=(0.8, 1.2)      # Ancho de hombros
            ),
            MovementType.BOTTOM_TURN: BiomechanicalProfile(
                head_inclination_range=(40, 50),      # Giro 40-50° hacia curva
                shoulder_width_range=(10, 15),        # Alineación con giro
                elbow_flexion_range=(40, 50),          # Derecho empujón, izq. estabiliza
                knee_flexion_range=(100, 130),        # Delantera 100-110°, trasera 120-130°
                hip_rotation_range=(25, 35),           # Rotación interna 25-35°
                spine_curvature_max=30,                # Torsión 25-30°
                center_of_mass_tolerance=0.15,        # Desplazado 10-15cm interior
                foot_separation_range=(0.3, 0.4)      # Pie trasero presiona riel
            ),
             MovementType.TOP_TURN: BiomechanicalProfile(
            head_inclination_range=(60, 75),
            shoulder_width_range=(15, 20),
            elbow_flexion_range=(55, 65),
            knee_flexion_range=(85, 95),
            hip_rotation_range=(25, 35),
            spine_curvature_max=50,
            center_of_mass_tolerance=0.15,
            foot_separation_range=(0.4, 0.6)
        ),
        MovementType.CUTBACK: BiomechanicalProfile(
            head_inclination_range=(60, 80),
            shoulder_width_range=(45, 55),
            elbow_flexion_range=(60, 70),
            knee_flexion_range=(75, 110),
            hip_rotation_range=(15, 25),
            spine_curvature_max=50,
            center_of_mass_tolerance=0.2,
            foot_separation_range=(0.5, 0.7)
        ),
        MovementType.FLOATER: BiomechanicalProfile(
            head_inclination_range=(0, 10),
            shoulder_width_range=(0, 5),
            elbow_flexion_range=(40, 50),
            knee_flexion_range=(120, 150),
            hip_rotation_range=(0, 10),
            spine_curvature_max=10,
            center_of_mass_tolerance=0.2,
            foot_separation_range=(0.8, 1.0)
        ),
         MovementType.REENTRY: BiomechanicalProfile(
            head_inclination_range=(45, 60),
            shoulder_width_range=(15, 20),
            elbow_flexion_range=(55, 65),
            knee_flexion_range=(90, 160),
            hip_rotation_range=(90, 160),
            spine_curvature_max=30,
            center_of_mass_tolerance=0.1,
            foot_separation_range=(0.7, 1.0)
        ),
        MovementType.SNAP: BiomechanicalProfile(
            head_inclination_range=(40, 50),
            shoulder_width_range=(50, 60),
            elbow_flexion_range=(65, 75),
            knee_flexion_range=(100, 120),
            hip_rotation_range=(15, 30),
            spine_curvature_max=20,
            center_of_mass_tolerance=0.2,
            foot_separation_range=(0.5, 0.7)
        ),
        MovementType.CARVING: BiomechanicalProfile(
            head_inclination_range=(30, 45),
            shoulder_width_range=(20, 30),
            elbow_flexion_range=(25, 35),
            knee_flexion_range=(90, 100),
            hip_rotation_range=(25, 35),
            spine_curvature_max=25,
            center_of_mass_tolerance=0.1,
            foot_separation_range=(0.7, 1.0)
        ),
        MovementType.ROUNDHOUSE_CUTBACK: BiomechanicalProfile(
            head_inclination_range=(70, 90),
            shoulder_width_range=(45, 55),
            elbow_flexion_range=(75, 85),
            knee_flexion_range=(80, 150),
            hip_rotation_range=(80, 160),
            spine_curvature_max=50,
            center_of_mass_tolerance=0.1,
            foot_separation_range=(0.6, 0.9)
        ),
        MovementType.FOAM_CLIMB: BiomechanicalProfile(
            head_inclination_range=(0, 0),
            shoulder_width_range=(0, 5),
            elbow_flexion_range=(40, 50),
            knee_flexion_range=(80, 150),
            hip_rotation_range=(90, 110),
            spine_curvature_max=10,
            center_of_mass_tolerance=0.1,
            foot_separation_range=(0.8, 1.0)
        ),
        MovementType.CLOSEOUT_REENTRY: BiomechanicalProfile(
            head_inclination_range=(30, 40),
            shoulder_width_range=(50, 60),
            elbow_flexion_range=(55, 65),
            knee_flexion_range=(80, 90),
            hip_rotation_range=(90, 110),
            spine_curvature_max=15,
            center_of_mass_tolerance=0.1,
            foot_separation_range=(0.7, 1.0)
        ),
        MovementType.TAIL_SLIDE: BiomechanicalProfile(
            head_inclination_range=(30, 40),
            shoulder_width_range=(30, 40),
            elbow_flexion_range=(50, 60),
            knee_flexion_range=(80, 130),
            hip_rotation_range=(65, 75),
            spine_curvature_max=25,
            center_of_mass_tolerance=0.25,
            foot_separation_range=(0.5, 0.7)
        ),
         MovementType.AGARRAR_VELOCIDAD: BiomechanicalProfile(
            head_inclination_range=(0, 5),
            shoulder_width_range=(10, 15),
            elbow_flexion_range=(15, 25),
            knee_flexion_range=(95, 140),
            hip_rotation_range=(20, 30),
            spine_curvature_max=10,
            center_of_mass_tolerance=0.05,
            foot_separation_range=(0.7, 1.0)
        ),
        MovementType.VELOCIDAD_HORIZONTAL: BiomechanicalProfile(
            head_inclination_range=(0, 0),
            shoulder_width_range=(0, 5),
            elbow_flexion_range=(0, 10),
            knee_flexion_range=(110, 130),
            hip_rotation_range=(90, 110),
            spine_curvature_max=10,
            center_of_mass_tolerance=0.15,
            foot_separation_range=(0.8, 1.0)
        ),
        }


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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback si fps es 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,  # Mayor confianza
        min_tracking_confidence=0.7
    )

    frame_count = 0
    keypoints_log = []
    processed_frames = 0
    analyzer = SurfBiomechanicsAnalyzer()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada 3 frames para mayor precisión
            if frame_count % 3 == 0:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        landmarks = extract_landmarks_as_bodyKeypoints(results.pose_landmarks)
                        if landmarks and len(landmarks) >= 33:  # MediaPipe devuelve 33 landmarks mínimo
                            # Validar calidad de keypoints si existe el método
                            if hasattr(analyzer.calculator, 'validate_keypoints_quality'):
                                if analyzer.calculator.validate_keypoints_quality(landmarks, min_confidence=0.7):
                                    keypoints_log.append({
                                        'frame_number': frame_count,
                                        'timestamp': frame_count / fps,
                                        'landmarks': landmarks
                                    })
                                    processed_frames += 1
                            else:
                                # Si no existe el método de validación, agregar directamente
                                keypoints_log.append({
                                    'frame_number': frame_count,
                                    'timestamp': frame_count / fps,
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

    # Suavizar keypoints si el método existe
    try:
        if hasattr(analyzer.calculator, 'smooth_keypoints') and len(keypoints_log) > 1:
            smoothed_keypoints = analyzer.calculator.smooth_keypoints(
                [frame['landmarks'] for frame in keypoints_log], 
                window_size=min(5, len(keypoints_log))
            )
            
            # Actualizar keypoints_log con datos suavizados
            for i, smoothed_frame in enumerate(smoothed_keypoints):
                if i < len(keypoints_log):
                    keypoints_log[i]['landmarks'] = smoothed_frame
    except Exception as e:
        print(f"Error en suavizado de keypoints: {str(e)}")
        # Continuar sin suavizado

    # Evaluar biomecánica con análisis detallado
    biomechanical_issues, detailed_analysis = evaluate_keypoints_detailed(
        keypoints_log, movement_type, analyzer
    )

    return {
        "frames_analyzed": processed_frames,
        "total_frames": total_frames,
        "movement_detected": movement_type.value,
        "issues_found": biomechanical_issues,
        "detailed_analysis": detailed_analysis,
        "video_duration": total_frames / fps,
        "fps": fps,
        "quality_score": calculate_overall_quality_score(detailed_analysis)
    }


def extract_landmarks_as_bodyKeypoints(pose_landmarks) -> Optional[List[BodyKeypoint]]:
    """
    Convierte landmarks de MediaPipe a objetos BodyKeypoint.
    """
    try:
        keypoints = []
        for landmark in pose_landmarks.landmark:
            keypoint = BodyKeypoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                confidence=landmark.visibility
            )
            keypoints.append(keypoint)
        return keypoints
    except Exception as e:
        print(f"Error extrayendo landmarks: {str(e)}")
        return None


def evaluate_keypoints_detailed(
    keypoints_sequence: List[Dict], 
    movement_type: MovementType, 
    analyzer: SurfBiomechanicsAnalyzer
) -> Tuple[List[str], Dict]:
    """
    Evalúa los keypoints del video con análisis biomecánico detallado.
    """
    issues = []
    detailed_analysis = {
        "measurements": {},
        "deviations": {},
        "recommendations": [],
        "phase_analysis": []
    }

    if not keypoints_sequence:
        return ["No se detectaron posturas válidas en el video"], detailed_analysis

    try:
        profile = analyzer.profiles[movement_type]
        measurements = []
        
        # Analizar cada frame
        for i, frame_data in enumerate(keypoints_sequence):
            landmarks = frame_data.get('landmarks', [])
            if not landmarks or len(landmarks) < 33:
                continue
                
            frame_analysis = analyze_frame_biomechanics(landmarks, profile, analyzer.calculator)
            measurements.append(frame_analysis)
            
            # Simplificar detección de fase (eliminar dependencia de método inexistente)
            phase = f"frame_{i}"
            detailed_analysis["phase_analysis"].append({
                "frame": frame_data['frame_number'],
                "phase": phase,
                "timestamp": frame_data['timestamp']
            })

        # Calcular promedios y desviaciones
        if measurements:
            detailed_analysis["measurements"] = calculate_average_measurements(measurements)
            issues, deviations = evaluate_against_profile(detailed_analysis["measurements"], profile, movement_type)
            detailed_analysis["deviations"] = deviations
            detailed_analysis["recommendations"] = generate_recommendations(deviations, movement_type)

    except Exception as e:
        issues.append(f"Error en el análisis biomecánico: {str(e)}")

    if not issues:
        issues.append("Análisis completado. Técnica dentro de parámetros aceptables.")

    return issues, detailed_analysis

def analyze_frame_biomechanics(
    landmarks: List[BodyKeypoint], 
    profile: BiomechanicalProfile, 
    calculator: BiomechanicsCalculator
) -> Dict:
    """
    Analiza la biomecánica de un frame individual con todas las métricas necesarias.
    """
    analysis = {}
    
    try:
        # Verificar que tenemos suficientes landmarks
        if len(landmarks) < 33:
            return analysis
        
        # 1. Inclinación de cabeza
        analysis['head_inclination'] = calculator.calculate_head_inclination(landmarks)
        
        # 2. Apertura de hombros
        analysis['shoulder_width'] = calculator.calculate_shoulder_alignment(landmarks)
        
        # 3. Flexión de codos
        analysis['left_elbow_flexion'] = calculator.calculate_arm_extension('left', landmarks)
        analysis['right_elbow_flexion'] = calculator.calculate_arm_extension('right', landmarks)
        
        # 4. Flexión de rodillas
        analysis['left_knee_flexion'] = calculator.calculate_knee_flexion('left', landmarks)
        analysis['right_knee_flexion'] = calculator.calculate_knee_flexion('right', landmarks)
        
        # 5. Rotación de caderas
        analysis['hip_rotation'] = calculator.calculate_hip_rotation(landmarks)
        
        # 6. Inclinación corporal
        analysis['body_inclination'] = calculator.calculate_body_inclination(landmarks)
        
        # 7. Curvatura de columna
        analysis['spine_curvature'] = calculator.calculate_spine_curvature(landmarks)
        
        # 8. Centro de masa
        analysis['center_of_mass'] = calculator.get_body_center(landmarks)
        
        # 9. Separación de pies
        analysis['foot_separation_ratio'] = calculator.calculate_foot_separation_ratio(landmarks)

        # 10. Análisis del torso respecto a la base (nuevo)
        left_foot = landmarks[31]
        right_foot = landmarks[32]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        if all([left_foot, right_foot, left_shoulder, right_shoulder]):
            base_x_min = min(left_foot.x, right_foot.x)
            base_x_max = max(left_foot.x, right_foot.x)
            torso_x = (left_shoulder.x + right_shoulder.x) / 2
            base_width = abs(base_x_max - base_x_min)
            margin = base_width * 0.05
            analysis['torso_out_of_base'] = not (base_x_min - margin <= torso_x <= base_x_max + margin)
            analysis['torso_base_margin'] = margin
            analysis['torso_x'] = torso_x
            analysis['base_x_min'] = base_x_min
            analysis['base_x_max'] = base_x_max

            # Inclinación de la tabla (base) y del torso
            tabla_angle = np.degrees(np.arctan2(right_foot.y - left_foot.y, right_foot.x - left_foot.x))
            torso_angle = np.degrees(np.arctan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x))
            angle_diff = abs(tabla_angle - torso_angle)
            analysis['torso_table_angle_diff'] = angle_diff
            analysis['tabla_angle'] = tabla_angle
            analysis['torso_angle'] = torso_angle
            analysis['torso_parallel_to_table'] = angle_diff <= 5.0  # margen de 5 grados
        
    except Exception as e:
        print(f"Error en análisis de frame: {str(e)}")
    
    return analysis

def calculate_average_measurements(measurements: List[Dict]) -> Dict:
    """
    Calcula los promedios de las mediciones de todos los frames.
    """
    if not measurements:
        return {}
    
    averages = {}
    
    # Obtener todas las claves únicas
    all_keys = set()
    for m in measurements:
        all_keys.update(m.keys() if m else [])
    
    for key in all_keys:
        if key == 'center_of_mass':
            # Manejar objeto BodyKeypoint especialmente
            x_values = []
            y_values = []
            
            for m in measurements:
                if key in m and m[key] and hasattr(m[key], 'x') and hasattr(m[key], 'y'):
                    x_values.append(m[key].x)
                    y_values.append(m[key].y)
            
            if x_values and y_values:
                averages[key] = {
                    'x': np.mean(x_values),
                    'y': np.mean(y_values)
                }
        else:
            values = []
            for m in measurements:
                if key in m and isinstance(m[key], (int, float)) and not np.isnan(m[key]):
                    values.append(m[key])
            
            if values:
                averages[key] = np.mean(values)
    
    return averages

def evaluate_against_profile(
    measurements: Dict, 
    profile: BiomechanicalProfile, 
    movement_type: MovementType
) -> Tuple[List[str], Dict]:
    """
    Evalúa las mediciones contra el perfil biomecánico ideal con TODOS los rangos.
    """
    issues = []
    deviations = {}
    
    # 1. Evaluar inclinación de cabeza
    if 'head_inclination' in measurements:
        head_inc = measurements['head_inclination']
        min_head, max_head = profile.head_inclination_range
        if not (min_head <= head_inc <= max_head):
            deviation = min(abs(head_inc - min_head), abs(head_inc - max_head))
            deviations['head_inclination'] = {
                'current': head_inc,
                'target_range': profile.head_inclination_range,
                'deviation': deviation
            }
            if head_inc < min_head:
                issues.append(f"Inclinación de cabeza insuficiente: {head_inc:.1f}° (objetivo: {min_head}-{max_head}°)")
            else:
                issues.append(f"Cabeza muy inclinada: {head_inc:.1f}° (objetivo: {min_head}-{max_head}°)")
    
    # 2. Evaluar apertura de hombros
    if 'shoulder_width' in measurements:
        shoulder_w = measurements['shoulder_width']
        min_shoulder, max_shoulder = profile.shoulder_width_range
        if not (min_shoulder <= shoulder_w <= max_shoulder):
            deviation = min(abs(shoulder_w - min_shoulder), abs(shoulder_w - max_shoulder))
            deviations['shoulder_width'] = {
                'current': shoulder_w,
                'target_range': profile.shoulder_width_range,
                'deviation': deviation
            }
            if shoulder_w < min_shoulder:
                issues.append(f"Hombros muy cerrados: {shoulder_w:.1f}° (objetivo: {min_shoulder}-{max_shoulder}°)")
            else:
                issues.append(f"Hombros muy abiertos: {shoulder_w:.1f}° (objetivo: {min_shoulder}-{max_shoulder}°)")
    
    # 3. Evaluar flexión de codos
    for side in ['left', 'right']:
        key = f'{side}_elbow_flexion'
        if key in measurements:
            elbow_flex = measurements[key]
            min_elbow, max_elbow = profile.elbow_flexion_range
            if not (min_elbow <= elbow_flex <= max_elbow):
                deviation = min(abs(elbow_flex - min_elbow), abs(elbow_flex - max_elbow))
                deviations[key] = {
                    'current': elbow_flex,
                    'target_range': profile.elbow_flexion_range,
                    'deviation': deviation
                }
                if elbow_flex < min_elbow:
                    issues.append(f"Codo {side} muy flexionado: {elbow_flex:.1f}° (objetivo: {min_elbow}-{max_elbow}°)")
                else:
                    issues.append(f"Codo {side} muy extendido: {elbow_flex:.1f}° (objetivo: {min_elbow}-{max_elbow}°)")
    
    # 4. Evaluar flexión de rodillas
    for side in ['left', 'right']:
        key = f'{side}_knee_flexion'
        if key in measurements:
            knee_flex = measurements[key]
            min_knee, max_knee = profile.knee_flexion_range
            if not (min_knee <= knee_flex <= max_knee):
                deviation = min(abs(knee_flex - min_knee), abs(knee_flex - max_knee))
                deviations[key] = {
                    'current': knee_flex,
                    'target_range': profile.knee_flexion_range,
                    'deviation': deviation
                }
                if knee_flex < min_knee:
                    issues.append(f"Rodilla {side} muy flexionada: {knee_flex:.1f}° (objetivo: {min_knee}-{max_knee}°)")
                else:
                    issues.append(f"Rodilla {side} muy extendida: {knee_flex:.1f}° (objetivo: {min_knee}-{max_knee}°)")
    
    # 5. Evaluar rotación de caderas
    if 'hip_rotation' in measurements:
        hip_rot = measurements['hip_rotation']
        min_hip, max_hip = profile.hip_rotation_range
        if not (min_hip <= hip_rot <= max_hip):
            deviation = min(abs(hip_rot - min_hip), abs(hip_rot - max_hip))
            deviations['hip_rotation'] = {
                'current': hip_rot,
                'target_range': profile.hip_rotation_range,
                'deviation': deviation
            }
            if hip_rot < min_hip:
                issues.append(f"Rotación de cadera insuficiente: {hip_rot:.1f}° (objetivo: {min_hip}-{max_hip}°)")
            else:
                issues.append(f"Exceso de rotación de cadera: {hip_rot:.1f}° (objetivo: {min_hip}-{max_hip}°)")
    
    # 6. Evaluar curvatura de columna
    if 'spine_curvature' in measurements:
        spine_curv = abs(measurements['spine_curvature'])
        if spine_curv > profile.spine_curvature_max:
            deviations['spine_curvature'] = {
                'current': spine_curv,
                'target_max': profile.spine_curvature_max,
                'deviation': spine_curv - profile.spine_curvature_max
            }
            issues.append(f"Columna muy curvada: {spine_curv:.1f}° (máximo: {profile.spine_curvature_max}°)")
    
    # 7. Evaluar separación de pies
    if 'foot_separation_ratio' in measurements:
        foot_ratio = measurements['foot_separation_ratio']
        min_foot, max_foot = profile.foot_separation_range
        if not (min_foot <= foot_ratio <= max_foot):
            deviation = min(abs(foot_ratio - min_foot), abs(foot_ratio - max_foot))
            deviations['foot_separation'] = {
                'current': foot_ratio,
                'target_range': profile.foot_separation_range,
                'deviation': deviation
            }
            if foot_ratio < min_foot:
                issues.append(f"Pies muy juntos: ratio {foot_ratio:.2f} (objetivo: {min_foot:.2f}-{max_foot:.2f})")
            else:
                issues.append(f"Pies muy separados: ratio {foot_ratio:.2f} (objetivo: {min_foot:.2f}-{max_foot:.2f})")
    
    # 8. Evaluar centro de masa (si es BodyKeypoint, convertir)
    if 'center_of_mass' in measurements:
        center = measurements['center_of_mass']
        if hasattr(center, 'x') and hasattr(center, 'y'):
            # Para Take Off: centro debe estar entre 0.4-0.6 (centrado)
            # Para Bottom Turn: centro desplazado hacia interior
            if movement_type == MovementType.TAKE_OFF:
                if not (0.4 <= center.x <= 0.6):
                    deviations['center_of_mass'] = {
                        'current': center.x,
                        'target_range': (0.4, 0.6),
                        'deviation': min(abs(center.x - 0.4), abs(center.x - 0.6))
                    }
                    issues.append(f"Centro de masa desplazado: {center.x:.2f} (objetivo: centrado 0.4-0.6)")
            elif movement_type == MovementType.BOTTOM_TURN:
                # Para bottom turn, el centro debe estar desplazado
                if 0.4 <= center.x <= 0.6:  # Demasiado centrado
                    deviations['center_of_mass'] = {
                        'current': center.x,
                        'target_range': (0.2, 0.4),  # Desplazado hacia interior
                        'deviation': min(abs(center.x - 0.2), abs(center.x - 0.4))
                    }
                    issues.append(f"Centro de masa muy centrado para giro: {center.x:.2f} (objetivo: desplazado 0.2-0.4)")
    
    # 9. Evaluar si el torso está fuera de la base
    if 'torso_out_of_base' in measurements:
        if measurements['torso_out_of_base']:
            deviations['torso_out_of_base'] = {
                'current': measurements.get('torso_x', 0),
                'base_x_min': measurements.get('base_x_min', 0),
                'base_x_max': measurements.get('base_x_max', 0),
                'margin': measurements.get('torso_base_margin', 0),
                'deviation': abs(measurements.get('torso_x', 0) - ((measurements.get('base_x_min', 0) + measurements.get('base_x_max', 0)) / 2))
            }
            issues.append("El torso se sale de la base de los pies (alineación lateral incorrecta).")

    # 10. Evaluar si el torso está paralelo a la tabla
    if 'torso_parallel_to_table' in measurements:
        if not measurements['torso_parallel_to_table']:
            deviations['torso_table_alignment'] = {
                'current': measurements.get('torso_angle', 0),
                'tabla_angle': measurements.get('tabla_angle', 0),
                'deviation': measurements.get('torso_table_angle_diff', 0)
            }
            issues.append("El torso no está alineado (paralelo) con la inclinación de la tabla.")
   
    return issues, deviations

def generate_recommendations(deviations: Dict, movement_type: MovementType) -> List[str]:
    """
    Genera recomendaciones específicas basadas en las desviaciones detectadas.
    """
    recommendations = []
    
    # Recomendaciones para inclinación de cabeza
    if 'head_inclination' in deviations:
        dev = deviations['head_inclination']
        current = dev['current']
        target_min, target_max = dev['target_range']
        
        if current < target_min:
            recommendations.append(
                f"🔸 POSTURA DE CABEZA: Eleva más la mirada. Tu cabeza está inclinada {current:.1f}° "
                f"pero necesita estar entre {target_min}-{target_max}°. Mantén la vista hacia adelante."
            )
        else:
            recommendations.append(
                f"🔸 POSTURA DE CABEZA: Reduce la inclinación de la cabeza. Está en {current:.1f}° "
                f"cuando debería estar entre {target_min}-{target_max}°. Relaja el cuello."
            )
    
    # Recomendaciones para hombros
    if 'shoulder_width' in deviations:
        dev = deviations['shoulder_width']
        current = dev['current']
        target_min, target_max = dev['target_range']
        
        if current < target_min:
            recommendations.append(
                f"🔸 APERTURA DE HOMBROS: Abre más los hombros para mejor equilibrio. "
                f"Actualmente {current:.1f}°, objetivo: {target_min}-{target_max}°."
            )
        else:
            recommendations.append(
                f"🔸 APERTURA DE HOMBROS: Relaja los hombros, están muy tensos. "
                f"Actualmente {current:.1f}°, objetivo: {target_min}-{target_max}°."
            )
    
    # Recomendaciones para codos
    elbow_issues = [k for k in deviations.keys() if 'elbow_flexion' in k]
    if elbow_issues:
        for side_key in elbow_issues:
            dev = deviations[side_key]
            current = dev['current']
            target_min, target_max = dev['target_range']
            side = 'izquierdo' if 'left' in side_key else 'derecho'
            
            if current < target_min:
                recommendations.append(
                    f"🔸 FLEXIÓN DE CODOS: Extiende más el codo {side}. "
                    f"Está en {current:.1f}°, necesita {target_min}-{target_max}° para mejor control."
                )
            else:
                recommendations.append(
                    f"🔸 FLEXIÓN DE CODOS: Flexiona ligeramente el codo {side}. "
                    f"Está muy rígido en {current:.1f}°, objetivo: {target_min}-{target_max}°."
                )
    
    # Recomendaciones para rodillas
    knee_issues = [k for k in deviations.keys() if 'knee_flexion' in k]
    if knee_issues:
        for side_key in knee_issues:
            dev = deviations[side_key]
            current = dev['current']
            target_min, target_max = dev['target_range']
            side = 'izquierda' if 'left' in side_key else 'derecha'
            
            if current < target_min:
                recommendations.append(
                    f"🔸 FLEXIÓN DE RODILLAS: Extiende más la rodilla {side}. "
                    f"Está muy flexionada en {current:.1f}°, objetivo: {target_min}-{target_max}°."
                )
            else:
                recommendations.append(
                    f"🔸 FLEXIÓN DE RODILLAS: Flexiona más la rodilla {side} para mejor absorción. "
                    f"Está en {current:.1f}°, necesita {target_min}-{target_max}°."
                )
    
    # Recomendaciones para rotación de caderas
    if 'hip_rotation' in deviations:
        dev = deviations['hip_rotation']
        current = dev['current']
        target_min, target_max = dev['target_range']
        
        if movement_type == MovementType.BOTTOM_TURN:
            if current < target_min:
                recommendations.append(
                    f"🔸 ROTACIÓN DE CADERAS: Rota más las caderas hacia el interior del giro. "
                    f"Actualmente {current:.1f}°, necesitas {target_min}-{target_max}° para un giro más fluido."
                )
            else:
                recommendations.append(
                    f"🔸 ROTACIÓN DE CADERAS: Controla la rotación excesiva de caderas. "
                    f"Está en {current:.1f}°, objetivo: {target_min}-{target_max}°."
                )
        else:
            recommendations.append(
                f"🔸 ROTACIÓN DE CADERAS: Mantén las caderas más alineadas. "
                f"Rotación actual: {current:.1f}°, objetivo: {target_min}-{target_max}°."
            )
    
    # Recomendaciones para curvatura de columna
    if 'spine_curvature' in deviations:
        dev = deviations['spine_curvature']
        current = dev['current']
        target_max = dev['target_max']
        
        recommendations.append(
            f"🔸 POSTURA DE ESPALDA: Endereza más la espalda para mejor control. "
            f"Curvatura actual: {current:.1f}°, máximo recomendado: {target_max}°. "
            f"Activa el core y mantén el pecho abierto."
        )
    
    # Recomendaciones para separación de pies
    if 'foot_separation' in deviations:
        dev = deviations['foot_separation']
        current = dev['current']
        target_min, target_max = dev['target_range']
        
        if current < target_min:
            recommendations.append(
                f"🔸 STANCE: Separa más los pies para mayor estabilidad. "
                f"Ratio actual: {current:.2f}, objetivo: {target_min:.2f}-{target_max:.2f}. "
                f"Una base más amplia mejorará tu equilibrio."
            )
        else:
            recommendations.append(
                f"🔸 STANCE: Acerca ligeramente los pies. "
                f"Ratio actual: {current:.2f}, objetivo: {target_min:.2f}-{target_max:.2f}. "
                f"Una base muy amplia reduce la maniobrabilidad."
            )
    
    # Recomendaciones para centro de masa
    if 'center_of_mass' in deviations:
        dev = deviations['center_of_mass']
        current = dev['current']
        
        if movement_type == MovementType.TAKE_OFF:
            recommendations.append(
                f"🔸 CENTRO DE MASA: Centrate mejor sobre la tabla. "
                f"Tu peso está desplazado ({current:.2f}). Distribuye el peso uniformemente "
                f"entre ambos pies para un take-off más estable."
            )
        elif movement_type == MovementType.BOTTOM_TURN:
            recommendations.append(
                f"🔸 CENTRO DE MASA: Inclinate más hacia el interior del giro. "
                f"Tu peso está muy centrado ({current:.2f}). Desplaza el peso hacia "
                f"el rail interior para generar más drive en el giro."
            )
    
    # Recomendaciones generales según el tipo de movimiento
    if movement_type == MovementType.TAKE_OFF:
        recommendations.append(
            "💡 CONSEJO GENERAL TAKE-OFF: Mantén una postura centrada y equilibrada. "
            "El take-off requiere estabilidad y control, no movimientos bruscos."
        )
    elif movement_type == MovementType.BOTTOM_TURN:
        recommendations.append(
            "💡 CONSEJO GENERAL BOTTOM TURN: Carga más peso en el rail interior y "
            "usa la flexión de rodillas para generar drive. El giro viene de las caderas."
        )
    elif movement_type == MovementType.CUTBACK:
        recommendations.append(
            "💡 CONSEJO GENERAL CUTBACK: Utiliza la rotación de hombros y caderas "
            "para generar el cambio de dirección. Mantén las rodillas flexibles."
        )
    
    # Si no hay recomendaciones específicas, dar feedback positivo
    if not recommendations:
        recommendations.append(
            "✅ ¡Excelente técnica! Tu postura está dentro de los rangos ideales. "
            "Sigue manteniendo esta forma y enfócate en la fluidez del movimiento."
        )
    # Recomendación para torso fuera de la base
    if 'torso_out_of_base' in deviations:
        recommendations.append(
            "🔸 ALINEACIÓN DEL TORSO: Mantén el torso dentro de la base formada por los pies. Evita inclinarte lateralmente fuera de la tabla para mejorar el equilibrio y control."
        )

    # Recomendación para torso no paralelo a la tabla
    if 'torso_table_alignment' in deviations:
        recommendations.append(
            "🔸 PARALELISMO TORSO-TABLA: Ajusta la inclinación del torso para que siga el ángulo de la tabla. El cuerpo debe acompañar la inclinación de la tabla para mayor estabilidad."
        )    
    
    return recommendations
def calculate_biomechanical_score(deviations: Dict) -> float:
    """
    Calcula un score general basado en las desviaciones (0-100).
    """
    if not deviations:
        return 100.0
    
    total_penalty = 0.0
    max_penalty_per_metric = 15.0  # Máximo 15 puntos de penalización por métrica
    
    for metric, dev_data in deviations.items():
        if 'deviation' in dev_data:
            deviation = dev_data['deviation']
            
            # Normalizar la desviación según el tipo de métrica
            if 'angle' in metric or 'flexion' in metric or 'inclination' in metric or 'rotation' in metric:
                # Para ángulos, penalizar más por desviaciones grandes
                normalized_penalty = min(deviation / 10.0, 1.0) * max_penalty_per_metric
            elif 'ratio' in metric or 'separation' in metric:
                # Para ratios, penalizar por desviaciones relativas
                normalized_penalty = min(deviation / 0.5, 1.0) * max_penalty_per_metric
            elif 'center_of_mass' in metric:
                # Para centro de masa, penalizar desplazamientos
                normalized_penalty = min(deviation / 0.2, 1.0) * max_penalty_per_metric
            else:
                # Penalización general
                normalized_penalty = min(deviation / 5.0, 1.0) * max_penalty_per_metric
            
            total_penalty += normalized_penalty
    
    # Calcular score final (100 - penalizaciones)
    final_score = max(0.0, 100.0 - total_penalty)
    return round(final_score, 1)


def get_biomechanical_profile(movement_type: MovementType) -> dict:
    """
    Extrae los puntos biomecánicos por tipo de movimiento con valores específicos.
    """
    analyzer = SurfBiomechanicsAnalyzer()
    profile = analyzer.profiles[movement_type]

    detailed_profiles = {
        MovementType.TAKE_OFF: {
            "Cabeza/CCuello": f"Mirada al frente, inclinación {profile.head_inclination_range[0]}–{profile.head_inclination_range[1]}° respecto al horizonte.",
            "Hombros": f"Abiertos {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°, escápulas retraídas y bajas.",
            "Codos": f"Flexión {profile.elbow_flexion_range[0]}°–{profile.elbow_flexion_range[1]}°, pegados al torso.",
            "Columna Torácica": f"Extendida, pecho elevado, sin curvar.",
            "Zona Lumbar/Caderas": "Elevación simultánea con hombros, alineación hombro-cadera.",
            "Rodillas": f"Semiflexión {profile.knee_flexion_range[0]}°–{profile.knee_flexion_range[1]}°.",
            "Pies": f"Delantero 40°–50°, trasero 15°–25°, separados {profile.foot_separation_range[0]:.1f}-{profile.foot_separation_range[1]:.1f}x ancho de hombros.",
            "Centro de Masa": "Dentro de la base entre ambos pies.",
            "Brazos": "Derecho: empujón y estabilización 50°–60°; Izquierdo: estabiliza 25°–35° lateral."
        },
        MovementType.BOTTOM_TURN: {
            "Cabeza/Cuello": f"Giro {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}° hacia la curva.",
            "Hombros": f"Delantero baja 10°–15°, trasero eleva 10°–15°.",
            "Columna Torácica": f"Torsión {profile.spine_curvature_max}° en el plano horizontal.",
            "Caderas": f"Rotación interna {profile.hip_rotation_range[0]}°–{profile.hip_rotation_range[1]}°.",
            "Rodillas": f"Delantera 100°–110°; Trasera 120°–130°.",
            "Pies": "Trasero presiona riel interno; delantero estabiliza con 30%–40% peso.",
            "Centro de Masa": f"Bajo, desplazado {profile.center_of_mass_tolerance*100:.0f}cm hacia interior del giro.",
            "Brazos": "Derecho: 40°–50° al giro; Izquierdo: 20°–30° abajo como contrapeso."
        },
        MovementType.TOP_TURN: {
            "Cabeza/Cuello": f"Rotación {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}° hacia la bajada.",
            "Hombros": f"Delantero baja 15°–20°, trasero eleva 15°–20°.",
            "Columna Torácica": f"Torsión sincronizada 40°–50°.",
            "Caderas": f"Giro 25°–35°, cadera trasera eleva 10°–15°.",
            "Rodillas": f"Flexión 85°–95° para cargar; extensión rápida al salir.",
            "Pies": "Trasero carga 60%–70% de peso sobre riel externo.",
            "Centro de Masa": "Transferencia 10%–15% hacia pie trasero.",
            "Brazos": "Derecho: alzado 55°–65° hacia la ola; Izquierdo: abierto 25°–35° atrás."
        },
        MovementType.CUTBACK: {
            "Cabeza/Cuello": f"Giro anticipado {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}° hacia la espuma.",
            "Hombros": f"Delantero gira 45°–55°; trasero abre 30°–40°.",
            "Columna Torácica": f"Torsión prolongada 40°–50°.",
            "Caderas": f"Retracción 20°–25° trasera; avance 15°–20° delantera.",
            "Rodillas": f"Delantera 75°–85°; trasera 100°–110°.",
            "Pies": "Pivote en trasero, delantero acompaña arco con 20% peso.",
            "Centro de Masa": "10%–20% desplazado hacia trasero.",
            "Brazos": "Derecho: envolvente 60°–70°; Izquierdo: desciende 20°–30°."
        },
        MovementType.FLOATER: {
            "Cabeza/Cuello": f"Neutro {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}° al frente.",
            "Hombros": f"Nivelados {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": "Alineada con tabla, sin torsión.",
            "Caderas": f"Mínima rotación <10°.",
            "Rodillas": f"Semiflexión {profile.knee_flexion_range[0]}°–{profile.knee_flexion_range[1]}°.",
            "Pies": "Distribución 50/50, microajustes de <5°.",
            "Centro de Masa": "10–20 cm sobre la tabla.",
            "Brazos": "Derecho e Izquierdo: semi-elevados 40°–50° para estabilizar."
        },
        MovementType.REENTRY: {
            "Cabeza/Cuello": f"Giro rápido {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}° antes del giro.",
            "Hombros": f"Delantero baja 15°–20°, trasero eleva 15°–20°.",
            "Columna Torácica": f"Extensión 10°–15° previa y torsión 30°.",
            "Caderas": f"Extensión 140°–160° al subir; flexión 90°–110° al aterrizar.",
            "Rodillas": f"Extensión completa 160°; compresión 90°–100°.",
            "Pies": "Aterrizaje 50%–50% presión.",
            "Centro de Masa": "Vertical alineado con tabla.",
            "Brazos": "Derecho: desciende 55°–65°; Izquierdo: mantiene 25°–35°."
        },
        MovementType.SNAP: {
            "Cabeza/Cuello": f"Explosivo giro {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}°.",
            "Hombros": f"Rotación rápida {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": "Giro y liberación en <0.3s.",
            "Caderas": f"Contrarrotación 15°–20° inicial, luego 30°.",
            "Rodillas": f"Flexión reactiva {profile.knee_flexion_range[0]}°–{profile.knee_flexion_range[1]}°.",
            "Pies": "Pivote trasero, carga 60%–70% peso.",
            "Centro de Masa": "Dinámico hacia trasero.",
            "Brazos": "Derecho: corta aire 65°–75°; Izquierdo: retrae 20°–30°."
        },
        MovementType.CARVING: {
            "Cabeza/Cuello": f"Giro leve {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}°.",
            "Hombros": f"Flexión {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": f"Torsión continua 15°–25°.",
            "Caderas": f"Rotación {profile.hip_rotation_range[0]}°–{profile.hip_rotation_range[1]}°.",
            "Rodillas": f"Flexión {profile.knee_flexion_range[0]}°–{profile.knee_flexion_range[1]}° adaptativa.",
            "Pies": "60%–40% distribución peso.",
            "Centro de Masa": "Bajo y alineado.",
            "Brazos": "Derecho: avanza 25°–35°; Izquierdo: equilibra 15°–25°."
        },
        MovementType.ROUNDHOUSE_CUTBACK: {
            "Cabeza/Cuello": f"Giro amplio {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}°.",
            "Hombros": f"Abiertos {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": f"Torsión sostenida 40°–50°.",
            "Caderas": f"Compresión 80°–90°; extensión 150°–160°.",
            "Rodillas": f"80° compresión; 140°–150° extensión.",
            "Pies": "Pivote y liberación sincronizada.",
            "Centro de Masa": "Bajo 5–10 cm.",
            "Brazos": "Derecho: eleva 75°–85°; Izquierdo: baja 25°–35°."
        },
        MovementType.FOAM_CLIMB: {
            "Cabeza/Cuello": f"Neutro {profile.head_inclination_range[0]}°.",
            "Hombros": "Nivelados.",
            "Columna Torácica": "Estable.",
            "Caderas": f"Flexión {profile.hip_rotation_range[0]}°–{profile.hip_rotation_range[1]}° al ascenso.",
            "Rodillas": f"Compresión 80°–90°; extensión 140°–150°.",
            "Pies": "Dorsiflexión 10°; empuje punta.",
            "Centro de Masa": "Adelantado 5%–10%.",
            "Brazos": "Derecho: extiende 40°–50°; Izquierdo: lateral 25°–35°."
        },
        MovementType.CLOSEOUT_REENTRY: {
            "Cabeza/Cuello": f"Anticipación {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}°.",
            "Hombros": f"Giro intenso {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": f"Extensión 15°; compresión 80°–90°.",
            "Caderas": f"Flexión {profile.hip_rotation_range[0]}°–{profile.hip_rotation_range[1]}° al aterrizar.",
            "Rodillas": f"Compresión 80°–90°.",
            "Pies": "Presión equitativa.",
            "Centro de Masa": "Bajo.",
            "Brazos": "Derecho: impulsa 55°–65°; Izquierdo: estabiliza 25°–35°."
        },
        MovementType.TAIL_SLIDE: {
            "Cabeza/Cuello": f"Giro {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}°.",
            "Hombros": f"Torsión {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": f"Flexión lateral 15°–25°.",
            "Caderas": f"Pie trasero carga 65%–75% peso.",
            "Rodillas": f"Trasera 110°–130°; Delantera 80°–100°.",
            "Pies": "Tail presiona 35°–45° de riel.",
            "Centro de Masa": "Desplazado 15%–25% atrás.",
            "Brazos": "Derecho: inicia 50°–60°; Izquierdo: baja 30°–40°."
        },
         MovementType.AGARRAR_VELOCIDAD: {
            "Cabeza/Cuello": f"Fija, {profile.head_inclination_range[0]}°–{profile.head_inclination_range[1]}°.",
            "Hombros": f"Oscilan {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": "Vertical oscila 10°.",
            "Caderas": f"Flexión-extensión {profile.hip_rotation_range[0]}°–{profile.hip_rotation_range[1]}°.",
            "Rodillas": f"95°–105° compresión y 130°–140° extensión.",
            "Pies": "Talón-punta 60%-40% transferencia.",
            "Centro de Masa": "Oscilación mínima 5 cm.",
            "Brazos": "Sube y baja 15°–25°."
        },
        MovementType.VELOCIDAD_HORIZONTAL: {
            "Cabeza/Cuello": f"Neutro {profile.head_inclination_range[0]}°.",
            "Hombros": f"Paralelos, {profile.shoulder_width_range[0]}°–{profile.shoulder_width_range[1]}°.",
            "Columna Torácica": "Alineada.",
            "Caderas": f"Flexión {profile.hip_rotation_range[0]}°–{profile.hip_rotation_range[1]}°.",
            "Rodillas": f"Semiflexión {profile.knee_flexion_range[0]}°–{profile.knee_flexion_range[1]}°.",
            "Pies": "Distribución estable 50/50.",
            "Centro de Masa": "Bajo 10–15 cm.",
            "Brazos": "Relajados junto al torso, <10° movimiento."
        }
    }

    return detailed_profiles.get(movement_type, {
        "General": "Perfil biomecánico no disponible para este movimiento"
    })
def generate_analysis_summary(
    measurements: Dict, 
    deviations: Dict, 
    movement_type: MovementType
) -> Dict:
    """
    Genera un resumen completo del análisis biomecánico.
    """
    score = calculate_biomechanical_score(deviations)  # Usar función renombrada
    
    # Clasificar nivel de técnica
    if score >= 90:
        technique_level = "Excelente"
        level_color = "green"
    elif score >= 75:
        technique_level = "Buena"
        level_color = "blue"
    elif score >= 60:
        technique_level = "Regular"
        level_color = "orange"
    else:
        technique_level = "Necesita Mejora"
        level_color = "red"
    
    # Identificar áreas críticas
    critical_areas = []
    if deviations:
        sorted_deviations = sorted(
            deviations.items(), 
            key=lambda x: x[1].get('deviation', 0), 
            reverse=True
        )
        
        for metric, _ in sorted_deviations[:3]:
            if 'head_inclination' in metric:
                critical_areas.append("Postura de cabeza")
            elif 'shoulder' in metric:
                critical_areas.append("Apertura de hombros")
            elif 'elbow' in metric:
                critical_areas.append("Flexión de codos")
            elif 'knee' in metric:
                critical_areas.append("Flexión de rodillas")
            elif 'hip' in metric:
                critical_areas.append("Rotación de caderas")
            elif 'spine' in metric:
                critical_areas.append("Curvatura de espalda")
            elif 'foot' in metric:
                critical_areas.append("Separación de pies")
            elif 'center_of_mass' in metric:
                critical_areas.append("Centro de masa")
    
    summary = {
        'overall_score': score,
        'technique_level': technique_level,
        'level_color': level_color,
        'movement_type': movement_type.value,
        'critical_areas': critical_areas,
        'total_deviations': len(deviations),
        'measurements_analyzed': len(measurements),
        'analysis_timestamp': time.time()
    }
    
    return summary


# 8. EJEMPLO DE USO COMPLETO



# En video_analysis_service.py, en la función principal de análisis:
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    frame_count = 0
    keypoints_log = []
    processed_frames = 0
    analyzer = SurfBiomechanicsAnalyzer()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada 3 frames
            if frame_count % 3 == 0:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        landmarks = extract_landmarks_as_bodyKeypoints(results.pose_landmarks)
                        if landmarks and len(landmarks) >= 33:
                            # Validación simplificada
                            keypoints_log.append({
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps,
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

    if not keypoints_log:
        raise ValueError("No se pudieron extraer poses del video. Verifica que el video contenga personas visibles.")

    # Evaluar biomecánica
    biomechanical_issues, detailed_analysis = evaluate_keypoints_detailed(
        keypoints_log, movement_type, analyzer
    )

    return {
        "frames_analyzed": processed_frames,
        "total_frames": total_frames,
        "movement_detected": movement_type.value,
        "issues_found": biomechanical_issues,
        "detailed_analysis": detailed_analysis,
        "video_duration": total_frames / fps,
        "fps": fps,
        "quality_score": calculate_overall_quality_score(detailed_analysis)
    }


def calculate_overall_quality_score(detailed_analysis: Dict) -> float:
    """
    Calcula un puntaje de calidad simplificado basado en el análisis detallado.
    """
    try:
        if not detailed_analysis or 'deviations' not in detailed_analysis:
            return 50.0  # Puntaje neutral
        
        deviations = detailed_analysis.get('deviations', {})
        return calculate_biomechanical_score(deviations)
        
    except Exception as e:
        print(f"Error calculando quality score: {str(e)}")
        return 50.0

def evaluate_body_alignment(landmarks: List[Dict]) -> float:
    """Evalúa la alineación corporal basada en landmarks"""
    try:
        if not landmarks or len(landmarks) < 33:
            return 0.0
        
        # Convertir landmarks a numpy array para facilitar cálculos
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Evaluar alineación vertical (hombros, caderas)
        shoulder_alignment = evaluate_shoulder_alignment(points)
        hip_alignment = evaluate_hip_alignment(points)
        spine_alignment = evaluate_spine_alignment(points)
        
        # Promedio de alineaciones
        alignment_score = (shoulder_alignment + hip_alignment + spine_alignment) / 3.0
        
        return max(0.0, min(1.0, alignment_score))
        
    except Exception as e:
        print(f"Error evaluando alineación: {str(e)}")
        return 0.0

def evaluate_shoulder_alignment(points: np.ndarray) -> float:
    """Evalúa la alineación de hombros"""
    try:
        # MediaPipe pose landmarks indices
        left_shoulder = points[11]  # LEFT_SHOULDER
        right_shoulder = points[12]  # RIGHT_SHOULDER
        
        # Calcular diferencia en altura
        height_diff = abs(left_shoulder[1] - right_shoulder[1])
        
        # Normalizar (menor diferencia = mejor alineación)
        alignment_score = max(0.0, 1.0 - (height_diff * 10))
        
        return alignment_score
        
    except Exception:
        return 0.5

def evaluate_hip_alignment(points: np.ndarray) -> float:
    """Evalúa la alineación de caderas"""
    try:
        left_hip = points[23]  # LEFT_HIP
        right_hip = points[24]  # RIGHT_HIP
        
        height_diff = abs(left_hip[1] - right_hip[1])
        alignment_score = max(0.0, 1.0 - (height_diff * 10))
        
        return alignment_score
        
    except Exception:
        return 0.5

def evaluate_spine_alignment(points: np.ndarray) -> float:
    """Evalúa la alineación de la columna"""
    try:
        # Puntos de referencia para la columna
        nose = points[0]
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_hip = points[23]
        right_hip = points[24]
        
        # Centro de hombros y caderas
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Vector de la columna
        spine_vector = shoulder_center - hip_center
        
        # Evaluar verticalidad (componente Y debería ser dominante)
        verticality = abs(spine_vector[1]) / (np.linalg.norm(spine_vector) + 1e-6)
        
        return max(0.0, min(1.0, verticality))
        
    except Exception:
        return 0.5

def calculate_power_score(frames_data: List[Dict]) -> float:
    """Calcula el puntaje de potencia basado en velocidad y amplitud"""
    try:
        if len(frames_data) < 2:
            return 0.0
        
        velocities = []
        amplitudes = []
        
        for i in range(1, len(frames_data)):
            prev_frame = frames_data[i-1]
            curr_frame = frames_data[i]
            
            if ('pose_landmarks' in prev_frame and 'pose_landmarks' in curr_frame and
                prev_frame['pose_landmarks'] and curr_frame['pose_landmarks']):
                
                # Calcular velocidad del centro de masa
                velocity = calculate_velocity_between_frames(prev_frame, curr_frame)
                velocities.append(velocity)
                
                # Calcular amplitud del movimiento
                amplitude = calculate_movement_amplitude(curr_frame['pose_landmarks'])
                amplitudes.append(amplitude)
        
        if not velocities or not amplitudes:
            return 0.0
        
        # Normalizar y combinar velocidad y amplitud
        velocity_score = min(1.0, np.mean(velocities) * 2)  # Factor de escala
        amplitude_score = min(1.0, np.mean(amplitudes))
        
        power_score = (velocity_score + amplitude_score) / 2.0
        
        return max(0.0, min(1.0, power_score))
        
    except Exception as e:
        print(f"Error calculando power score: {str(e)}")
        return 0.0

def calculate_velocity_between_frames(prev_frame: Dict, curr_frame: Dict) -> float:
    """Calcula la velocidad entre dos frames"""
    try:
        prev_landmarks = prev_frame['pose_landmarks']
        curr_landmarks = curr_frame['pose_landmarks']
        
        if len(prev_landmarks) != len(curr_landmarks):
            return 0.0
        
        # Calcular centro de masa
        prev_center = np.mean([[lm['x'], lm['y']] for lm in prev_landmarks], axis=0)
        curr_center = np.mean([[lm['x'], lm['y']] for lm in curr_landmarks], axis=0)
        
        # Calcular velocidad
        displacement = np.linalg.norm(curr_center - prev_center)
        
        return displacement
        
    except Exception:
        return 0.0

def calculate_movement_amplitude(landmarks: List[Dict]) -> float:
    """Calcula la amplitud del movimiento"""
    try:
        if not landmarks:
            return 0.0
        
        # Calcular rango de movimiento en X e Y
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Amplitud total
        amplitude = np.sqrt(x_range**2 + y_range**2)
        
        return amplitude
        
    except Exception:
        return 0.0

def calculate_timing_score(frames_data: List[Dict], movement_type: str) -> float:
    """Calcula el puntaje de timing basado en el tipo de movimiento"""
    try:
        total_frames = len(frames_data)
        
        if total_frames == 0:
            return 0.0
        
        # Rangos ideales por tipo de movimiento (en frames aproximados)
        ideal_ranges = {
            'take_off': (15, 45),  # 0.5 - 1.5 segundos a 30fps
            'landing': (10, 30),   # 0.3 - 1.0 segundos a 30fps
            'jump': (20, 60),      # 0.7 - 2.0 segundos a 30fps
            'run': (30, 90)        # 1.0 - 3.0 segundos a 30fps
        }
        
        ideal_min, ideal_max = ideal_ranges.get(movement_type, (20, 60))
        
        # Calcular puntaje basado en qué tan cerca está del rango ideal
        if ideal_min <= total_frames <= ideal_max:
            timing_score = 1.0
        elif total_frames < ideal_min:
            timing_score = total_frames / ideal_min
        else:  # total_frames > ideal_max
            timing_score = ideal_max / total_frames
        
        return max(0.0, min(1.0, timing_score))
        
    except Exception:
        return 0.0

def generate_recommendations(scores: Dict[str, float]) -> List[str]:
    """Genera recomendaciones basadas en los puntajes"""
    recommendations = []
    
    try:
        # Recomendaciones basadas en técnica
        if scores['technique_score'] < 0.6:
            recommendations.append("Mejorar la alineación corporal y postura durante el movimiento")
        
        # Recomendaciones basadas en consistencia
        if scores['consistency_score'] < 0.7:
            recommendations.append("Trabajar en la consistencia del movimiento - practicar repeticiones")
        
        # Recomendaciones basadas en potencia
        if scores['power_score'] < 0.6:
            recommendations.append("Incrementar la velocidad y amplitud del movimiento")
        
        # Recomendaciones basadas en timing
        if scores['timing_score'] < 0.7:
            recommendations.append("Ajustar el timing del movimiento - puede ser muy rápido o muy lento")
        
        # Recomendaciones generales
        if scores['overall_score'] > 0.8:
            recommendations.append("¡Excelente técnica! Mantener el nivel y refinar detalles")
        elif scores['overall_score'] > 0.6:
            recommendations.append("Buena técnica base, enfocar en las áreas de mejora identificadas")
        else:
            recommendations.append("Requiere trabajo fundamental en técnica y forma")
        
        return recommendations if recommendations else ["Continuar practicando para mejorar"]
        
    except Exception:
        return ["Error generando recomendaciones"]

def get_overall_rating(overall_score: float) -> str:
    """Convierte el puntaje numérico en una calificación textual"""
    if overall_score >= 0.9:
        return "Excelente"
    elif overall_score >= 0.8:
        return "Muy Bueno"
    elif overall_score >= 0.7:
        return "Bueno"
    elif overall_score >= 0.6:
        return "Regular"
    elif overall_score >= 0.4:
        return "Necesita Mejora"
    else:
        return "Insuficiente"

def identify_weaknesses(scores: Dict[str, float]) -> List[str]:
    """Identifica las principales debilidades"""
    weaknesses = []
    
    score_names = {
        'technique_score': 'Técnica',
        'consistency_score': 'Consistencia',
        'power_score': 'Potencia',
        'timing_score': 'Timing'
    }
    
    # Identificar puntajes por debajo del 60%
    for key, value in scores.items():
        if key in score_names and value < 0.6:
            weaknesses.append(score_names[key])
    
    return weaknesses