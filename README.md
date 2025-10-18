# DS3D-FGAttNet

Modelo dual-stream 3D para **detección automática de violencia** en vídeos de cámaras estacionarias, que integra dos flujos de entrada —RGB y flujo óptico— procesados por ramas convolucionales tridimensionales paralelas.
La arquitectura implementa mecanismos de atención CBAM3D (espacial y de canal) y SEBlock (squeeze-and-excitation) para resaltar características discriminativas, mientras que la fusión guiada por flujo (FGF) permite una integración adaptativa entre ambas corrientes de información mediante compuertas multiplicativas.
El modelo aprovecha convoluciones separables 3D, normalización ligera, optimizando el equilibrio entre precisión y eficiencia computacional en entornos de videovigilancia fija.

---

## 🚀 Requisitos

- Python ≥ 3.10
- PyTorch 2.x con CUDA (recomendado)
- Instalar dependencias:

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📚 Dataset (RWF-2000)

1. Descargar desde Kaggle: https://www.kaggle.com/datasets/vulamnguyen/rwf2000
2. Configurar la ruta de origen del dataset con los videos y la ruta destino donde se va a guardar los videos en .npy luego, convertir a `.npy` (RGB + Flow) con:

```bash
python video_to_numpy.py
```

Estructura esperada:

```
data/npy/train/{Fight,NonFight}
data/npy/val/{Fight,NonFight}
```

---

## ⚙️ Configuración

Editar el archivo YAML correspondiente:

- Propuesto: `src/config/config_best_v2.yaml`
- Baseline: `src/config/config_best.yaml`

Ejemplo mínimo:

```yaml
general:
  seed: 42
  device: "cuda"
dataset:
  root: "data/npy"
output:
  root: "outputs"
```

---

## ▶️ Entrenamiento

- Modelo propuesto:

```bash
python -m src.main --mode train_v2
```

- Baseline (CNN3D):

```bash
python -m src.main --mode train
```

Atajos directos (opcionales):

```bash
python src/training/training_cbam_se_fgf.py
python src/training/train_by_npy.py
```

---

### 📊 Uso con Weights & Biases (W&B) (Opcional)

Para registrar métricas y artefactos en W&B:

1. Iniciar sesión con tu API Key (solo la primera vez):

   ```bash
   wandb login TU_API_KEY
   ```

   2. En el archivo de configuración (`src/config/config_best_v2.yaml` o `config_best.yaml`), activar logging:

      ```yaml
      logging:
      use_wandb: true
      general:
      project_name: "ViolenceDualStreamNet"
      experiment_name: "experimento_prueba"
      ```

## 🔎 Búsqueda de hiperparámetros (Opcional)

```bash
python src/tools/optimize_hparams_v2.py
python src/tools/optimize_hparams.py
```

---

## 🧩 Estructura y scripts relevantes

### Control y orquestación

- `src/main.py`: CLI principal. Ejecuta modos `train` (baseline), `train_v2` (modelo con CBAM/SE/FGF), `evaluate`, `predict` y `fine_tune` si existen los scripts correspondientes.

### Preparación de datos

- `src/video_to_numpy.py`: Convierte vídeos crudos a `.npy` con **RGB + flujo óptico (Farneback)**; salida esperada `(T, 224, 224, 5)`.
- `src/datasets/rwf_npy_dataset.py`: Dataset para clips `.npy` (RWF‑2000). Transpone a `(5, D, H, W)`, normaliza (RGB a [-1,1], flow recortado) y valida forma/valores.
- `src/datasets/tensor_transform.py`: Transformaciones a nivel de tensor configurables por YAML (resize, crops basados en movimiento, jitter temporal, flip, cutout, noise de flow, normalización “standard” o “statistical”).
- `src/datasets/tensor_validator.py`: Validador de tensores (forma exacta, NaN/Inf, rango [-1,1], varianza mínima y etiqueta válida).

### Entrenamiento

- `src/training/train_by_npy.py`: Entrenamiento del **baseline** (CNN3D). Carga config, dataset, entrena y guarda curvas/artefactos.
- `src/training/training_cbam_se_fgf.py`: Entrenamiento del **modelo propuesto** (CBAM3D + SEBlock + FGF).
- `src/training/trainer.py`: Bucle de entrenamiento con **AMP**, **class weights** opcionales, **early stopping**, **checkpointing**, **pruning** programable, **scheduler**, métricas completas y gancho para **Optuna**. Exporta histórico y predicciones finales.
- `src/training/callback.py`: `EarlyStopping` (min/max con `delta`) y `ModelCheckpoint` (guarda mejor modelo, por‑época y artefactos en W&B; adjunta YAML de config).
- `src/training/pruning.py`: Pruning estructurado de `Conv3d` (L2 por canal de salida), estático o con **scheduler** (warmup + rampa de esparsidad).

### Métricas, logging y utilidades

- `src/training/metrics/metric_utils.py`: Cálculo robusto de **F1**, precisión, recall, **accuracy**, **AUC**, **specificity** y **sensitivity**.
- `src/training/utils/log_utils.py`: Guarda métricas/CM en JSON por época.
- `src/training/utils/model_utils.py`: Limpieza de máscaras de pruning y export de **modelo “slimmed”** listo para despliegue.
- `src/training/utils/optuna_utils.py`: Reporte de métrica primaria a **Optuna** y pruning de trials.
- `src/utils/checkpoints.py`: Carga/guarda **last_checkpoint** con estados de **modelo, optimizador, scaler, scheduler**, RNG de **PyTorch/CUDA/NumPy**, `early_stopping` y `WANDB_RUN_ID` (permite reanudar).
- `src/utils/config_loader.py`: Carga YAML y lo expone como `SimpleNamespace` (atributos anidados).
- `src/utils/compute_mean.py`: Cálculo de **mean/std** por canal (RGB+Flow) sobre la partición.
- `src/utils/load_clip.py`: Carga un clip individual desde `.npy`, aplica normalización y devuelve `(1, 5, 32, 224, 224)` listo para inferencia/visualización.
- `src/utils/seed.py`: Semillas deterministas en **Python/NumPy/PyTorch**.

---

### Inferencia y despliegue

Los notebooks incluidos permiten realizar **inferencia y análisis visual** sobre clips de prueba (RWF‑2000).
Ambos cargan configuraciones YAML predefinidas para reproducir el entorno de entrenamiento.

- `notebooks/inference_violence_dualstreamnet_cbam_se_fgf.ipynb`:
  Ejecuta inferencia con el **modelo propuesto DS3D‑FGAttNet** (Dual‑Stream 3D CNN con CBAM3D, SEBlock y fusión FGF).
  Incluye visualización de frames, heatmaps de atención, y métricas por clip.
- `notebooks/inferencia_violence_detection_3dcnn.ipynb`:
  Ejecuta inferencia con el **modelo alterno CNN3D baseline**, bajo idénticas condiciones de preprocesado y normalización.
  Facilita la comparación directa de desempeño y latencia.

Configuraciones de soporte:

- `config_inference_proposed.yaml` → hiperparámetros, rutas del modelo propuesto.
- `config_inference_altern.yaml` → parámetros equivalentes para el modelo alterno.

Ambos pueden ejecutarse en modo CPU o GPU y exportan las predicciones junto con métricas resumidas.

## 📈 Resultados

Los artefactos se guardan en `outputs/`:

- `metrics/` → métricas y curvas
- `model/` → checkpoints y exportados
- `predictions/` → predicciones por clip
