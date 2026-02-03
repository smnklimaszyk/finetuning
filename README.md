# Medical Diagnosis Model Finetuning

**Datenschutzkonformes Finetuning eines Small Language Models für medizinische Diagnoseunterstützung**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Projektziel

Dieses Projekt entwickelt ein spezialisiertes KI-Modell zur Unterstützung von Ärzten bei der Diagnosestellung. Basierend auf Arzt-Patienten-Dialogen schlägt das Modell passende **ICD-10 Diagnose-Codes** vor.

## Vergleich

Wir vergleichen drei Ansätze:

| Ansatz            | Beschreibung                      |
| ----------------- | --------------------------------- |
| **Baseline LLM**  | Großes Modell mit System-Prompt   |
| **Baseline SLM**  | Kleines Modell ohne Finetuning    |
| **Finetuned SLM** | Kleines Modell nach LoRA-Training |

## Quick Start

```bash
# 1. Environment einrichten
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Baseline-Evaluation
python main.py --experiment baseline

# 3. Vollständiger Durchlauf (mit Training)
python main.py --experiment full
```

## 📁 Projektstruktur

```
finetuning/
├── src/            # Python Packages
│   ├── config/     # Konfiguration
│   ├── data/       # Datenverarbeitung
│   ├── models/     # Modell-Wrapper
│   ├── training/   # Finetuning mit LoRA
│   ├── evaluation/ # Metriken & Visualisierung
│   └── utils/      # Hilfsfunktionen
├── tests/          # Unit Tests
├── data/           # Daten-Outputs
├── models/         # Modell-Checkpoints
├── main.py         # Haupt-Pipeline
├── GUIDE.md        # 📖 Ausführlicher Guide
└── README.md       # Diese Datei
```

## 📖 Dokumentation

**Für eine vollständige Erklärung aller Komponenten, Hyperparameter und Konzepte siehe [GUIDE.md](GUIDE.md).**

## 🔧 Technologien

- **PyTorch** - Deep Learning Framework
- **Transformers** - HuggingFace Transformers
- **PEFT/LoRA** - Parameter-Efficient Fine-Tuning
- **BitsAndBytes** - Quantisierung
- **MLflow** - Experiment Tracking

## Daten

Der [MedSynth-Datensatz](https://huggingface.co/datasets/Ahmad0067/MedSynth) enthält:

- Synthetische Arzt-Patienten-Dialoge
- ICD-10 Diagnose-Codes
- Ca. 10.000 Beispiele
