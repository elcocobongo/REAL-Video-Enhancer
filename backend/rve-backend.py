# run_ddp_backend.py (Lanzador para Kaggle)

import os
import sys
import torch
import torch.multiprocessing as mp
import argparse
import cv2

# ### NECESITA ADAPTACIÓN ###
# Asegúrate de que esta ruta sea correcta para tu fork clonado en Kaggle.
# Si clonaste tu fork en /kaggle/working/REAL-Video-Enhancer, entonces:
REAL_VIDEO_ENHANCER_PATH = "REAL-Video-Enhancer"

# Añadir el directorio del backend al PATH para que Python pueda encontrarlo
sys.path.insert(0, os.path.join(REAL_VIDEO_ENHANCER_PATH, "backend"))

# Importar la función de procesamiento DDP desde el backend modificado
# Asegúrate de que el nombre del archivo sea rve-backend.py en tu fork
from rve_backend import render_video_segment_ddp # Importa la función que procesa cada segmento

# --- Función principal para lanzar los procesos DDP ---
def main_ddp_launcher():
    parser = argparse.ArgumentParser(description="DDP Launcher for REAL Video Enhancer Backend")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video file path (e.g., /kaggle/input/your-dataset/video.mp4)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output video file path (e.g., /kaggle/working/output.mp4)")
    parser.add_argument("--interpolate_factor", type=int, default=2, help="Interpolation factor (e.g., 2 for 2x, 4 for 4x)")
    parser.add_argument("--overlap_frames", type=int, default=10, help="Number of overlap frames at segment boundaries")
    
    # ### NECESITA ADAPTACIÓN ###
    # Añade aquí cualquier otro argumento que necesites pasar a rve-backend.py
    # (como --interpolate_model, --upscale_model, etc.)
    parser.add_argument("--interpolate_model", type=str, default="rife4.6.pkl", help="Interpolation model name or path")

    launcher_args = parser.parse_args()

    # Configurar variables de entorno para DDP
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" # Puedes usar cualquier puerto libre

    world_size = torch.cuda.device_count() # Detecta el número de GPUs disponibles
    if world_size < 2:
        print("Se necesitan al menos 2 GPUs para el procesamiento DDP. Abortando.")
        print("Asegúrate de que tu notebook de Kaggle esté configurado para usar 2xT4 o similar.")
        return

    print(f"Iniciando procesamiento DDP con {world_size} GPUs...")
    
    # Obtener el número total de fotogramas del video de entrada
    cap = cv2.VideoCapture(launcher_args.input)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video de entrada: {launcher_args.input}")
        return
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video de entrada: {launcher_args.input}")
    print(f"Total de fotogramas: {total_video_frames}")

    # Preparar los argumentos que se pasarán a cada proceso DDP
    # mp.spawn solo puede pasar argumentos que sean serializables.
    # Pasamos un diccionario de argumentos para reconstruirlos en el proceso hijo.
    args_for_processes = {
        "input": launcher_args.input,
        "output": launcher_args.output,
        "interpolate_factor": launcher_args.interpolate_factor,
        "overlap_frames": launcher_args.overlap_frames,
        "master_addr": os.environ["MASTER_ADDR"],
        "master_port": os.environ["MASTER_PORT"],
        "interpolate_model": launcher_args.interpolate_model, # Pasa el modelo de interpolación
        # ### NECESITA ADAPTACIÓN ###
        # Añade aquí todos los demás argumentos que rve-backend.py necesite
        # (backend, upscale_model, video_encoder_preset, etc.)
        "backend": "pytorch",
        "video_encoder_preset": "hevc_nvenc",
        "audio_encoder_preset": "aac",
        "subtitle_encoder_preset": "copy_subtitle",
    }

    # Lanzar los procesos DDP
    # La función `render_video_segment_ddp` se ejecutará en cada proceso.
    mp.spawn(render_video_segment_ddp, 
             args=(world_size, args_for_processes, total_video_frames),
             nprocs=world_size, 
             join=True)

    print("Proceso DDP finalizado.")

if __name__ == '__main__':
    main_ddp_launcher()
