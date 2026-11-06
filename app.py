from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil, os, uuid, librosa, numpy as np, json
import demucs_onnx

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("separated", exist_ok=True)
os.makedirs("results", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/separated", StaticFiles(directory="separated"), name="separated")

@app.post("/upload/")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    safe_name = file.filename.replace(" ", "_")
    file_path = f"uploads/{file_id}_{safe_name}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_audio, file_path, file_id)
    return JSONResponse({"status": "processing", "file_id": file_id})

def process_audio(file_path, file_id):
    try:
        # 1. Análise
        y, sr = librosa.load(file_path, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tom = notas[np.argmax(np.sum(chroma, axis=1))]
        
        # 2. Separação com demucs-onnx
        nome_base = os.path.splitext(os.path.basename(file_path))[0]
        pasta_stems = f"separated/{nome_base}"
        os.makedirs(pasta_stems, exist_ok=True)
        
        demucs_onnx.separate(file_path, output_dir=pasta_stems)
        
        # 3. Salvar resultado
        resultado = {
            "bpm": round(bpm, 2),
            "tom": tom,
            "stems": {
                "vocals": f"/{pasta_stems}/vocals.wav",
                "drums": f"/{pasta_stems}/drums.wav",
                "bass": f"/{pasta_stems}/bass.wav",
                "other": f"/{pasta_stems}/other.wav"
            }
        }
        
        with open(f"results/{file_id}.json", "w") as f:
            json.dump(resultado, f)
            
    except Exception as e:
        print(f"Erro no processamento: {e}")

@app.get("/results/{file_id}")
def get_results(file_id: str):
    caminho = f"results/{file_id}.json"
    if os.path.exists(caminho):
        with open(caminho, "r") as f:
            return json.load(f)
    return JSONResponse({"status": "processing"})

app.mount("/", StaticFiles(directory="static", html=True), name="static_root")