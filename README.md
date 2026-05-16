# ChiGNN — Modelado Generativo de Cadenas Laterales de Proteínas mediante Difusión Torsional

> **Trabajo Final de Máster** · Máster Universitario en Bioinformática y Bioestadística (UOC)  
> Autora: **Noelia Fernández Espiñeira** · Tutor: **Brian Jiménez García**  
> Área: Diseño de Fármacos y Biología Estructural · Junio 2026

---

## Resumen

**ChiGNN** es un modelo de inteligencia artificial generativa basado en **difusión torsional sobre el espacio circular S¹** para la recuperación de conformaciones de cadenas laterales de proteínas a partir de un *backbone* fijo. La clave técnica del modelo es el uso de la **distribución de Von Mises** como ruido del proceso de difusión directa, lo que permite respetar la naturaleza periódica y circular de los ángulos diedros χ₁–χ₄, a diferencia de los enfoques gaussianos estándar.

El modelo fue entrenado sobre **597 proteínas de alta resolución** (<2,0 Å) extraídas de PDB-REDO y logró una **tasa de recuperación de rotámeros del 53,9%** (tolerancia ±40°) con un MAE circular de χ₁ de **56,41°**, superando los baselines aleatorio y modal en χ₂–χ₄. El análisis de calibración de confianza arroja un coeficiente de Spearman **ρ = 0,299** (p < 0,001), evidenciando una señal real de autodiagnóstico del modelo.

---

## Tabla de contenidos

- [Motivación](#motivación)
- [Arquitectura del modelo](#arquitectura-del-modelo)
- [Pipeline del proyecto](#pipeline-del-proyecto)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y requisitos](#instalación-y-requisitos)
- [Uso](#uso)
- [Dataset](#dataset)
- [Comparativa con el estado del arte](#comparativa-con-el-estado-del-arte)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Citación](#citación)
- [Licencia](#licencia)

---

## Motivación

Aunque herramientas como AlphaFold2 predicen el *backbone* proteico con alta fidelidad (TM-score > 0,9), la posición de los ángulos diedros χ₁–χ₄ de las cadenas laterales sigue siendo imprecisa (~70–75% de recuperación en χ₁). Esta limitación es crítica porque **las cadenas laterales son las responsables directas de la función biológica**: definen los sitios activos enzimáticos, determinan la unión de ligandos y fármacos, y establecen los puentes de hidrógeno que estabilizan la estructura terciaria.

Los métodos clásicos (SCWRL4, Rosetta) modelan la conformación como un problema de optimización **determinista**, sin capturar la distribución de conformaciones posibles. ChiGNN aborda este problema mediante un enfoque **generativo probabilístico** con incertidumbre calibrada.

---

## Arquitectura del modelo

ChiGNN es una arquitectura *lite* inspirada en los principios de DiffPack, diseñada para ser ejecutable en entornos de computación en la nube (Google Colab con GPU T4).

```
Entrada: grafo de proteína G(V, E)
  ├── Nodos V: residuos amino ácidos (coordenadas Cα, tipo de residuo, ángulos φ/ψ)
  └── Aristas E: contactos espaciales (umbral de distancia Cα < 8 Å)

Proceso de difusión directa (forward):
  χ_t = χ_0 + ε,   ε ~ Von Mises(0, κ)

Arquitectura GNN (proceso inverso):
  ├── 4 × GCNConv con BatchNorm y conexiones residuales
  ├── Función de activación: ReLU
  └── Cabeza de salida: predicción del gradiente ∇ log p(χ_t)

Proceso de difusión inversa: DDIM adaptado al espacio circular S¹
Salida: ángulos diedros χ₁, χ₂, χ₃, χ₄ recuperados por residuo
```

**Parámetros del modelo:** 80.404 parámetros entrenables  
**Optimizador:** AdamW + CosineAnnealingLR  
**Entrenamiento:** 50 épocas · Mejor *checkpoint*: época 42 (val_loss = 0,0885)

---

## Pipeline del proyecto

El proyecto se organiza en tres fases ejecutadas desde un único notebook de Google Colab:

### Fase 1 — Cimentación y preparación de los datos
1. Consulta a la API REST de RCSB con filtros de calidad: resolución <2,0 Å, cristalografía de rayos X, cadenas proteicas
2. Descarga de estructuras re-refinadas desde **PDB-REDO** (`{pid}_final.pdb`)
3. Extracción de ángulos diedros χ₁–χ₄ mediante **Biopython** y coordenadas de *backbone*
4. Serialización del dataset en formato **PyTorch Geometric** (`.pt`) con máscara booleana `y_mask`
5. Partición **80/10/10** a nivel de proteína para evitar *data leakage*

### Fase 2 — Implementación del modelo de IA
1. Implementación del proceso forward de Von Mises con visualización en **py3Dmol**
2. Diseño y entrenamiento de la **GNN** (ChiGNN) para la difusión inversa
3. Recuperación de conformaciones y ajuste de hiperparámetros

### Fase 3 — Evaluación bioestadística y benchmarking
1. Cálculo del **MAE circular** y tasa de recuperación de rotámeros (±20° y ±40°)
2. Análisis de distribuciones rotaméricas con **diagramas de rosa** (*Rose Plots*)
3. Calibración de confianza: correlación de Spearman entre varianza de Von Mises y MAE real
4. Benchmarking frente a SCWRL4 y baselines aleatorio/modal

El notebook incluye un **panel de control** con flags booleanos para controlar qué fases se re-ejecutan o se cargan desde Google Drive, minimizando el tiempo de cómputo en sesiones posteriores.

---

## Resultados

| Métrica | ChiGNN | Baseline modal | Baseline aleatorio |
|---|---|---|---|
| MAE circular χ₁ | **56,41°** [IC 95%: 55,63°–57,15°] | ~60–65° | ~90° |
| Recuperación rotámeros χ₁ (±40°) | **53,9%** | ~47% | ~33% |
| Recuperación rotámeros χ₁ (±20°) | ~28% | — | — |
| Calibración (Spearman ρ) | **0,299** (p < 0,001) | — | — |

Los **diagramas de rosa** confirman que el modelo reproduce cualitativamente la trimodalidad rotamérica característica (g⁻ ≈ −60°, t ≈ 180°, g⁺ ≈ +60°).

> **Nota:** SCWRL4 alcanza ~83% de recuperación en χ₁. La brecha actual se atribuye principalmente al tamaño del dataset (597 proteínas vs. decenas de miles) y a la arquitectura no equivariante.

---

## Estructura del repositorio

```
torsional-diffusion-sidechain/
│
├── TFM_script.ipynb              # Notebook principal (Fases 1, 2 y 3)
├── README.md
│
├── data/
│   ├── dataset_sidechains_curated.csv   # Dataset legible (residuos, coords, ángulos χ)
│   └── dataset_torsional_diffusion.pt   # Grafos serializados para PyTorch Geometric
│
├── models/
│   └── chignn_best.pt            # Checkpoint del mejor modelo (época 42)
│
├── results/
│   ├── figures/                  # Rose plots, curvas de aprendizaje, calibración
│   └── metrics/                  # Tablas de MAE y recuperación de rotámeros
│
└── docs/
    └── TFM_Noelia_Fernandez.pdf  # Memoria completa del TFM
```

> Los archivos de datos (`.pt`, `.csv`) y el checkpoint del modelo se alojan en Google Drive por su tamaño. Ver sección [Dataset](#dataset) para instrucciones de descarga.

---

## Instalación y requisitos

El proyecto está diseñado para ejecutarse en **Google Colab con GPU** (T4 o superior). La instalación de dependencias se realiza automáticamente desde el Bloque 2 del notebook.

### Requisitos de sistema
- Python ≥ 3.9
- CUDA ≥ 11.x (GPU obligatoria; el entrenamiento no es viable en CPU)
- Google Colab (recomendado) o entorno local con GPU compatible

### Instalación manual (entorno local)

```bash
# Instalar dependencias base
pip install biopython py3Dmol rdkit

# PyTorch Geometric (ajustar versiones según tu entorno)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.',''))")

pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install torch-sparse  -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install torch-geometric
```

### Stack tecnológico

| Librería | Uso |
|---|---|
| PyTorch + CUDA | Backend de entrenamiento y difusión |
| PyTorch Geometric | Construcción y propagación de grafos GNN |
| Biopython | Parsing de estructuras PDB y cálculo de ángulos diedros |
| NumPy / Pandas | Procesamiento de datos y métricas |
| Matplotlib | Rose plots, curvas de aprendizaje, histogramas |
| py3Dmol | Visualización 3D interactiva de proteínas |
| RDKit | Química computacional auxiliar |

---

## Uso

### Opción A — Google Colab (recomendado)

1. Abre el notebook `TFM_script.ipynb` en Google Colab
2. Activa la GPU: `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`
3. Monta tu Google Drive cuando se solicite
4. **Primera ejecución:** en el Bloque 0, establece `FORCE_ALL = True` y ejecuta todas las celdas
5. **Ejecuciones posteriores:** mantén todos los flags en `False` para cargar desde Drive (~30 seg)

### Opción B — Inferencia sobre una estructura nueva

```python
from Bio.PDB import PDBParser
import torch
from model import ChiGNN, run_reverse_diffusion

# Cargar estructura
parser = PDBParser(QUIET=True)
structure = parser.get_structure("proteina", "mi_proteina.pdb")

# Cargar modelo
model = ChiGNN()
model.load_state_dict(torch.load("models/chignn_best.pt"))
# Repositorio: https://github.com/noeliafernandezesp/torsional-diffusion-sidechain
model.eval()

# Recuperar cadenas laterales
chi_pred = run_reverse_diffusion(model, structure, n_steps=100)
print(chi_pred)  # ángulos χ₁–χ₄ predichos por residuo
```

### Panel de control del pipeline

El Bloque 0 del notebook permite controlar qué fases se re-ejecutan:

```python
FORCE_ALL        = False  # True = regenera todo desde cero
FORCE_DOWNLOAD   = False  # Re-descarga PDBs de PDB-REDO
FORCE_EXTRACTION = False  # Re-extrae ángulos chi y coordenadas
FORCE_DATASET    = False  # Re-construye grafos y partición train/val/test
FORCE_TRAINING   = False  # Re-entrena la GNN
FORCE_INFERENCE  = False  # Re-genera predicciones sobre el test set
```

La lógica de cascada garantiza la coherencia: si se re-extrae el dataset, el entrenamiento y la inferencia se re-ejecutan automáticamente.

---

## Dataset

El dataset fue construido a partir de **PDB-REDO**, una base de datos de estructuras cristalográficas re-refinadas con REFMAC5, que ofrece mayor precisión en coordenadas atómicas y geometría de enlace respecto al PDB original.

**Criterios de selección:**
- Resolución < 2,0 Å (cristalografía de rayos X)
- Solo cadenas proteicas (sin ADN/ARN)
- Disponibilidad en PDB-REDO (`{pid}_final.pdb`)

**Estadísticas del dataset curado:**

| Parámetro | Valor |
|---|---|
| Proteínas totales | 597 |
| Residuos totales | 156.407 |
| Partición train / val / test | 80% / 10% / 10% (a nivel de proteína) |
| Distribución χ₁ | Trimodal confirmada (g⁻, t, g⁺) |
| Formato | `.csv` (legible) + `.pt` (PyTorch Geometric) |

La descarga y procesamiento de los archivos PDB se realiza automáticamente desde el notebook (Bloques 5–9). Los archivos procesados se almacenan en Google Drive bajo `MyDrive/TFM_SideChain/`.

---

## Comparativa con el estado del arte

| Método | Tipo | Recuperación χ₁ (±40°) | Incertidumbre |
|---|---|---|---|
| **ChiGNN** (este trabajo) | Difusión torsional (GNN) | 53,9% | ✅ Calibrada (Von Mises) |
| SCWRL4 | Librería de rotámeros | ~83% | ❌ No |
| Rosetta FastRelax | Campo de fuerza | ~85% | ❌ No |
| AlphaFold2 | Deep learning predictivo | ~70–75% | ❌ No |
| AlphaFold3 | Difusión all-atom | >90% | Parcial |
| DiffPack | Difusión torsional | ~85% | ✅ Sí |

La **ventaja diferencial de ChiGNN** frente a métodos determinísticos no reside en la precisión puntual, sino en la **calibración de incertidumbre**: la varianza circular de Von Mises de las muestras del proceso inverso se correlaciona significativamente con el error real (ρ = 0,299, p < 0,001), lo que permite identificar predicciones de baja confianza sin información experimental adicional.

---

## Limitaciones y trabajo futuro

### Limitaciones actuales
- **Dataset reducido:** 597 proteínas frente a las decenas de miles usadas por DiffPack o AlphaFold3
- **Arquitectura no equivariante:** GCNConv no es invariante a rotaciones y traslaciones del sistema de referencia; una arquitectura EGNN o SE(3)-equivariante mejoraría sustancialmente la precisión
- **Proceso de evaluación:** la ausencia de SCWRL4 instalado en el entorno de Colab impide la comparación directa sobre el mismo conjunto de test

### Líneas de trabajo futuro
1. **Escalado del dataset:** ampliar a ≥10.000 proteínas de PDB-REDO con balanceo por tipo de aminoácido
2. **Arquitecturas equivariantes:** integrar EGNN, GVP o ProteinMPNN como *backbone* del denoiser
3. **Difusión multi-ángulo:** modelar la distribución conjunta P(χ₁, χ₂, χ₃, χ₄) en lugar de tratarlos de forma independiente
4. **Validación externa:** benchmarking sobre CASP15 y el conjunto de test estándar de SCWRL4
5. **Refinamiento iterativo:** combinar ChiGNN con un campo de fuerza ligero como paso de post-procesamiento

---

## Citación

Si utilizas este trabajo en tu investigación, por favor cita:

```bibtex
@mastersthesis{fernandez2026chignn,
  author  = {Fernández Espiñeira, Noelia},
  title   = {Modelado Generativo de Cadenas Laterales de Proteínas
             mediante Difusión Torsional en Espacios No Euclidianos},
  school  = {Universitat Oberta de Catalunya (UOC)},
  year    = {2026},
  month   = {June},
  program = {Máster Universitario en Bioinformática y Bioestadística},
  area    = {Diseño de Fármacos y Biología Estructural},
  advisor = {Jiménez García, Brian},
  url     = {https://github.com/noeliafernandezesp/torsional-diffusion-sidechain}
}
```

---

## Licencia

© 2026 Noelia Fernández Espiñeira. Todos los derechos reservados.

El código fuente de este repositorio se distribuye bajo licencia **MIT** para uso académico y de investigación. Está prohibida la reproducción total o parcial de la memoria del TFM por cualquier medio sin autorización expresa de la autora.

---

*Proyecto desarrollado como Trabajo Final del Máster Universitario en Bioinformática y Bioestadística de la UOC, en el área de Diseño de Fármacos y Biología Estructural.*
