# 🏥 Medical Diagnosis Model Finetuning

## Datenschutzkonformes Finetuning eines Small Language Models für medizinische Diagnoseunterstützung

---

# 📖 VOLLSTÄNDIGER ENTWICKLER-GUIDE

Dieser Guide führt dich Schritt für Schritt durch die komplette Entwicklung und Nutzung dieses MLOps-Projekts.
Er ist für Entwickler geschrieben, die ein tiefes Verständnis für Machine Learning Engineering entwickeln möchten.

---

## 📋 Inhaltsverzeichnis

1. [Projektübersicht & Architektur](#1-projektübersicht--architektur)
2. [Theoretische Grundlagen](#2-theoretische-grundlagen)
3. [Projektstruktur Erklärt](#3-projektstruktur-erklärt)
4. [Konfigurationssystem](#4-konfigurationssystem)
5. [Datenverarbeitungs-Pipeline](#5-datenverarbeitungs-pipeline)
6. [Modell-Architektur](#6-modell-architektur)
7. [Training mit LoRA](#7-training-mit-lora)
8. [Evaluation & Metriken](#8-evaluation--metriken)
9. [Experiment-Workflow](#9-experiment-workflow)
10. [Best Practices & Troubleshooting](#10-best-practices--troubleshooting)

---

## 1. Projektübersicht & Architektur

### 1.1 Was macht dieses Projekt?

Dieses Projekt entwickelt ein **spezialisiertes KI-Modell** zur Unterstützung von Ärzten bei der Diagnosestellung.
Basierend auf Arzt-Patienten-Dialogen schlägt das Modell passende **ICD-10 Diagnose-Codes** vor.

**Der Workflow:**

```
Arzt-Patienten-Dialog → KI-Modell → ICD-10 Code Vorschlag
```

### 1.2 Experimentelles Design: Größe vs. Spezialisierung

**Zentrale Forschungsfrage:**
> Können spezialisierte kleine Modelle (3B) größere untrainierte Modelle (7-8B) schlagen?

Wir vergleichen zwei fundamentale Ansätze:

| Ansatz | Beschreibung | Modelle | Vorteil | Nachteil |
|--------|--------------|---------|---------|----------|
| **LLMs (Large, Untrained)** | Große Modelle (7-8B) ohne Finetuning | Llama 8B, Mistral 7B | Mehr Parameter = mehr Wissen | Langsamer, mehr Ressourcen |
| **SLMs (Small, Finetuned)** | Kleine Modelle (3B) mit LoRA Finetuning | Llama 3B, Qwen 3B | Schneller, spezialisiert, effizient | Trainingsaufwand |

**Was wir testen:**
- **Size Advantage**: Haben 7-8B Modelle genug Wissen für medizinische Diagnosen?
- **Specialization Advantage**: Kann Finetuning den Größennachteil kompensieren?
- **Deployment Trade-offs**: Performance vs. Effizienz

### 1.3 Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: LLM Evaluation (Zero-Shot)                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Llama 8B  →  Zero-Shot Predict  →  Evaluate        │      │
│  │  Mistral 7B → Zero-Shot Predict  →  Evaluate        │      │
│  └──────────────────────────────────────────────────────┘      │
│                            ↓                                    │
│         (Memory Cleanup & GPU Reset)                           │
│                            ↓                                    │
│  PHASE 2: SLM Finetuning + Evaluation                         │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Llama 3B  →  LoRA Train  →  Evaluate Finetuned    │      │
│  │  Qwen 3B   →  LoRA Train  →  Evaluate Finetuned    │      │
│  └──────────────────────────────────────────────────────┘      │
│                            ↓                                    │
│  RESULT: Compare Size vs. Specialization                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Experimentelles Design (NEU!)

### 2.1 Die zentrale Forschungsfrage

**Kann Spezialisierung Größe schlagen?**

In diesem Projekt testen wir eine fundamentale Hypothese im Machine Learning:

> Können kleine, spezialisierte Modelle (3B Parameter mit Finetuning)
> größere, generelle Modelle (7-8B Parameter ohne Finetuning)
> bei spezifischen Aufgaben übertreffen?

### 2.2 Warum ist das wichtig?

**Praktische Relevanz:**
- **Kosten:** Kleinere Modelle = weniger GPU-Kosten
- **Latenz:** 3B Modelle sind 2-3x schneller als 8B Modelle
- **Datenschutz:** Kleine Modelle können lokal laufen (kein Cloud-Upload)
- **Energie:** Weniger Parameter = weniger Stromverbrauch

**Wissenschaftliche Relevanz:**
- Versteht man die Trade-offs zwischen Größe und Spezialisierung?
- Wann lohnt sich Finetuning?
- Gibt es einen "Sweet Spot" zwischen Größe und Effizienz?

### 2.3 Experimenteller Aufbau

```
LLMs (Size Advantage)           SLMs (Specialization Advantage)
├─ Llama 8B (untrained)        ├─ Llama 3B (LoRA finetuned)
└─ Mistral 7B (untrained)      └─ Qwen 3B (LoRA finetuned)

        ↓                               ↓
   Zero-Shot                     Domain-Adapted
   Inference                     Inference
        ↓                               ↓
   ┌─────────────────────────────────────┐
   │     ICD-10 Code Classification      │
   │        (Medical Diagnosis)          │
   └─────────────────────────────────────┘
        ↓                               ↓
   Performance                     Performance
   Comparison                      Comparison
```

### 2.4 Erwartete Ergebnisse

**Mögliches Szenario 1: SLMs gewinnen**
- Finetuning kompensiert Größennachteil
- → Empfehlung: Nutze kleine finetuned Modelle in Produktion

**Mögliches Szenario 2: LLMs gewinnen**
- Allgemeines Wissen wichtiger als Spezialisierung
- → Empfehlung: Nutze größere Modelle auch ohne Finetuning

**Mögliches Szenario 3: Nuanciert**
- SLMs besser bei häufigen Diagnosen
- LLMs besser bei seltenen/komplexen Fällen
- → Empfehlung: Hybrid-System je nach Use Case

---

## 3. Theoretische Grundlagen

### 3.1 Was ist Finetuning?

**Finetuning** ist das Anpassen eines vortrainierten Modells auf eine spezifische Aufgabe.

```
Vortrainiertes Modell     Finetuning        Spezialisiertes Modell
(generelles Wissen)    →  (+ Domain-Daten) →  (+ Domänen-Wissen)
```

**Warum Finetuning statt Training von Grund auf?**

- Vortrainierte Modelle haben bereits Sprachverständnis gelernt
- Finetuning braucht viel weniger Daten (1000e vs. Milliarden)
- Schneller und günstiger

### 3.2 Was ist LoRA?

**LoRA (Low-Rank Adaptation)** ist eine effiziente Finetuning-Methode.

**Das Problem:** Normale Finetuning-Methoden ändern alle Parameter (Milliarden!).

**Die Lösung:** LoRA trainiert nur kleine "Adapter"-Matrizen:

```
Original-Matrix W:    [1000 x 1000] = 1.000.000 Parameter
LoRA-Matrizen A, B:   [1000 x 16] + [16 x 1000] = 32.000 Parameter
                                                   = 3.2% der Original-Größe!
```

**LoRA-Formel:**

```
W' = W + ΔW = W + A × B
```

Wobei:

- `W` = Originale Gewichte (eingefroren, nicht trainiert)
- `A` = Down-Projection (Input → niedrig-dimensionaler Raum)
- `B` = Up-Projection (niedrig-dimensionaler Raum → Output)
- `r` = Rank (typisch 8-64, kontrolliert Kapazität)

### 3.3 ICD-10 Klassifikation

**ICD-10** (International Classification of Diseases) ist das weltweite Standard-System für Diagnosen.

**Aufbau:**

```
J06.9
│││ │
│││ └── Weitere Spezifikation (.9 = nicht näher bezeichnet)
│││
││└──── Hauptgruppe innerhalb Kapitel (06 = Akute Infektionen obere Atemwege)
││
│└───── Kapitel-Buchstabe (J = Atmungssystem)
│
└────── Hierarchie-Ebene
```

**Beispiele:**

- `J06.9` = Akute Infektion der oberen Atemwege, nicht näher bezeichnet
- `I10` = Essentielle Hypertonie (Bluthochdruck)
- `G43.9` = Migräne, nicht näher bezeichnet

---

## 4. Projektstruktur Erklärt

### 4.1 Verzeichnisstruktur

```
finetuning/
├── src/                    # 📦 Python Packages (src-layout)
│   ├── config/            # 🔧 Konfiguration
│   │   ├── __init__.py
│   │   └── base_config.py # Alle Hyperparameter und Settings
│   │
│   ├── data/              # 📊 Datenverarbeitung
│   │   ├── __init__.py
│   │   ├── data_loader.py # Lädt Daten von HuggingFace
│   │   └── data_processor.py  # Tokenisierung und Formatierung
│   │
│   ├── models/            # 🤖 Modell-Wrapper
│   │   ├── __init__.py
│   │   ├── base_model.py  # Abstrakte Basisklasse
│   │   ├── llm_model.py   # Large Language Model Wrapper
│   │   └── slm_model.py   # Small Language Model Wrapper
│   │
│   ├── training/          # 🏋️ Training
│   │   ├── __init__.py
│   │   └── trainer.py     # Finetuning mit LoRA
│   │
│   ├── evaluation/        # 📈 Auswertung
│   │   ├── __init__.py
│   │   ├── metrics.py     # Metriken-Berechnung
│   │   └── visualization.py  # Plots und Reports
│   │
│   └── utils/             # 🛠️ Hilfsfunktionen
│       └── __init__.py    # Logging, Helpers
│
├── tests/                 # ✅ Unit Tests
│   ├── __init__.py
│   └── test_pipeline.py
│
├── data/                  # 📊 Daten-Outputs (raw, processed, cache)
├── models/                # 💾 Modell-Checkpoints und Finetuned Models
├── notebooks/             # 📓 Jupyter Notebooks
├── experiments/           # 🧪 Experiment Tracking (MLflow)
├── outputs/               # 📁 Outputs (logs, metrics, plots, reports)
│
├── main.py               # 🚀 Haupt-Pipeline
├── pyproject.toml        # 📦 Dependencies
├── .gitignore           # Git-Ignore
└── GUIDE.md             # 📖 Dieser Guide
```

### 3.2 Warum src-layout?

Diese Struktur folgt **MLOps Best Practices** und verwendet das **src-layout**:

1. **Saubere Trennung:** Code (src/) vs. Daten/Outputs (Projekt-Root)
2. **Build-Tool Kompatibilität:** setuptools, pip, uv funktionieren problemlos
3. **Keine versehentlichen Imports:** Nur installierte Packages sind importierbar
4. **Separation of Concerns:** Jedes Modul hat eine klare Verantwortung
5. **Testbarkeit:** Klare Interfaces ermöglichen Unit Tests
6. **Reproduzierbarkeit:** Experimente sind nachvollziehbar

---

## 4. Konfigurationssystem

### 4.1 Datei: `src/config/base_config.py`

Das Konfigurationssystem nutzt **Pydantic** für type-safe Konfigurationen.

**Warum Pydantic?**

- Automatische Typvalidierung
- Defaults und Overrides
- Serialisierung (JSON speichern/laden)
- IDE-Unterstützung (Autocomplete)

### 4.2 Wichtige Konfigurationsklassen

#### DataConfig

```python
class DataConfig(BaseModel):
    dataset_name: str = "Ahmad0067/MedSynth"  # HuggingFace Dataset
    train_ratio: float = 0.7   # 70% für Training
    val_ratio: float = 0.15    # 15% für Validation
    test_ratio: float = 0.15   # 15% für finale Tests
    max_sequence_length: int = 512  # Max Token-Länge
    batch_size: int = 8
```

**Erklärung der Split-Ratios:**

- **Training (70%):** Das Modell lernt von diesen Daten
- **Validation (15%):** Zum Tunen von Hyperparametern und Early Stopping
- **Test (15%):** Finale Evaluation - NIEMALS während Training nutzen!

#### TrainingConfig

```python
class TrainingConfig(BaseModel):
    # Wichtigste Hyperparameter
    num_epochs: int = 3              # Durchläufe durch Datensatz
    learning_rate: float = 2e-5      # Schrittgröße beim Lernen
    warmup_steps: int = 500          # Langsamer Start
    weight_decay: float = 0.01       # L2-Regularisierung

    # LoRA-Konfiguration
    use_lora: bool = True
    lora_r: int = 16                 # Rank (8-64 typisch)
    lora_alpha: int = 32             # Scaling-Faktor
    lora_dropout: float = 0.05       # Regularisierung
```

### 4.3 Hyperparameter-Erklärungen

| Parameter               | Typischer Wert | Bedeutung                          | Effekt wenn zu hoch            | Effekt wenn zu niedrig |
| ----------------------- | -------------- | ---------------------------------- | ------------------------------ | ---------------------- |
| `learning_rate`         | 1e-5 bis 5e-5  | Wie stark werden Weights angepasst | Instabiles Training, Divergenz | Zu langsames Lernen    |
| `num_epochs`            | 1-5            | Anzahl Durchläufe                  | Overfitting                    | Underfitting           |
| `warmup_steps`          | 500-2000       | Schritte zum Hochfahren der LR     | Zu langsamer Start             | Instabiler Anfang      |
| `weight_decay`          | 0.01-0.1       | L2-Regularisierung                 | Zu starke Regularisierung      | Overfitting            |
| `lora_r`                | 8-64           | LoRA Rank/Kapazität                | Mehr Memory, evtl. Overfitting | Zu wenig Kapazität     |
| `lora_alpha`            | 2\*r           | LoRA Scaling                       | Stärkere Adaptation            | Schwächere Adaptation  |
| `batch_size`            | 4-32           | Samples pro Schritt                | Memory-Fehler                  | Langsam, instabil      |
| `gradient_accumulation` | 1-8            | Simuliert größere Batches          | Langsamer                      | Weniger stabil         |

**Die "effektive Batch-Größe":**

```
Effektive Batch Size = batch_size × gradient_accumulation_steps × num_gpus

Beispiel: 4 × 4 × 1 = 16 effektive Batch Size
```

---

## 5. Datenverarbeitungs-Pipeline

### 5.1 Datei: `data/data_loader.py`

Diese Datei lädt den MedSynth-Datensatz von HuggingFace.

**MedSynth-Datensatz:**

- Synthetische Arzt-Patienten-Dialoge
- ICD-10 Diagnose-Codes
- Ca. 50.000 Beispiele

**Wichtige Methoden:**

```python
class MedSynthDataLoader:
    def load(self) -> Dataset:
        """Lädt Dataset von HuggingFace Hub."""

    def get_statistics(self) -> Dict:
        """Berechnet Statistiken (Längen, Verteilungen)."""

    def validate_dataset(self) -> bool:
        """Prüft ob Dataset erwartete Struktur hat."""
```

### 5.2 Datei: `data/data_processor.py`

Hier werden die Rohdaten in ein für das Modell verständliches Format gebracht.

**Der Verarbeitungsprozess:**

```
Roher Dialog           Formatierung           Tokenisierung
"Patient: ..." →  "[SYSTEM] Du bist..." → [101, 234, 567, ...]
                  "[USER] Dialog..."       (Token IDs)
                  "[ASSISTANT] J06.9"
```

**Wichtige Konzepte:**

#### Tokenisierung

Tokenisierung wandelt Text in Zahlen um:

```
"Ich habe Kopfschmerzen"
→ ["Ich", "habe", "Kopf", "##schmerzen"]  (Subword Tokenization)
→ [1234, 5678, 9012, 3456]                 (Token IDs)
```

#### Chat-Templates

Moderne Modelle erwarten spezielle Formatierungen:

```python
# Phi-3 Format
<|system|>
Du bist ein medizinisches Assistenzsystem.
<|end|>
<|user|>
Analysiere diesen Dialog...
<|end|>
<|assistant|>
J06.9
<|end|>
```

### 5.3 Train/Val/Test Split

```python
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Warum 3 Splits?

    Training (70%): Modell lernt von diesen Daten
    ↓
    Validation (15%): Prüft Fortschritt WÄHREND Training
                      - Hyperparameter-Tuning
                      - Early Stopping
    ↓
    Test (15%): Finale Evaluation NACH Training
                - Nur EINMAL nutzen!
                - Niemals für Entscheidungen während Training
    """
```

---

## 6. Modell-Architektur

### 6.1 Datei: `models/base_model.py`

Definiert die abstrakte Schnittstelle für alle Modelle.

**Design Pattern: Strategy Pattern**

```
                    ┌──────────────┐
                    │  BaseModel   │  (Abstract)
                    │  - predict() │
                    │  - load()    │
                    └──────────────┘
                          ↑
            ┌─────────────┴─────────────┐
            ↓                           ↓
    ┌──────────────┐           ┌──────────────┐
    │   LLMModel   │           │   SLMModel   │
    │   (Groß)     │           │   (Klein)    │
    └──────────────┘           └──────────────┘
```

**Vorteile:**

- Einheitliche Schnittstelle für alle Modelle
- Einfacher Austausch von Modellen
- Konsistente Evaluation

### 6.2 Datei: `models/llm_model.py`

Wrapper für Large Language Models.

**Wichtige Features:**

#### 4-bit Quantisierung

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,           # Aktiviert 4-bit Quantisierung
    bnb_4bit_compute_dtype=torch.float16,  # Rechentyp
    bnb_4bit_use_double_quant=True,  # Nested Quantization
    bnb_4bit_quant_type="nf4"    # Normal Float 4-bit
)
```

**Was ist Quantisierung?**

```
Original (float32):    32 bit pro Parameter  → 100% Memory
Half Precision (fp16): 16 bit pro Parameter  → 50% Memory
4-bit Quantisierung:   4 bit pro Parameter   → 12.5% Memory
```

**Trade-off:** Weniger Memory, minimal schlechtere Qualität.

#### Generation-Parameter

```python
outputs = model.generate(
    max_new_tokens=256,      # Max Ausgabelänge
    temperature=0.7,         # Kreativität (0=deterministisch, 1=kreativ)
    top_p=0.9,              # Nucleus Sampling
    top_k=50,               # Top-K Sampling
    repetition_penalty=1.1,  # Verhindert Wiederholungen
)
```

**Sampling-Strategien erklärt:**

| Strategie                      | Beschreibung                       | Wann nutzen?                      |
| ------------------------------ | ---------------------------------- | --------------------------------- |
| **Greedy** (`do_sample=False`) | Immer wahrscheinlichstes Token     | Deterministische Ergebnisse nötig |
| **Temperature**                | Flacht Probability-Verteilung ab   | Kreativere Ausgaben               |
| **Top-K**                      | Nur aus K besten Tokens wählen     | Verhindert "verrückte" Tokens     |
| **Top-p (Nucleus)**            | Aus kleinster Menge die p% abdeckt | Dynamischer als Top-K             |

---

## 7. Training mit LoRA

### 7.1 Datei: `training/trainer.py`

Implementiert das Finetuning mit Parameter-Efficient Fine-Tuning (PEFT).

### 7.2 LoRA im Detail

```python
lora_config = LoraConfig(
    r=16,                    # Rank - Kapazität der Adapter
    lora_alpha=32,           # Scaling-Faktor
    lora_dropout=0.05,       # Regularisierung
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Welche Layer
    task_type=TaskType.CAUSAL_LM,  # Aufgabentyp
)
```

**Welche Layer werden adaptiert?**

```
Transformer Block
├── Self-Attention
│   ├── q_proj (Query)      ← LoRA Adapter
│   ├── k_proj (Key)        ← LoRA Adapter
│   ├── v_proj (Value)      ← LoRA Adapter
│   └── o_proj (Output)     ← LoRA Adapter
├── MLP
│   ├── gate_proj
│   ├── up_proj
│   └── down_proj
```

**Parameter-Einsparung Beispiel (Phi-3 Mini, 3.8B Parameter):**

```
Ohne LoRA:  3,800,000,000 trainierbare Parameter
Mit LoRA:      10,000,000 trainierbare Parameter (0.26%!)
```

### 7.3 Training-Loop erklärt

```python
# Vereinfachter Training Loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 1. Forward Pass: Berechne Predictions
        outputs = model(batch)
        loss = outputs.loss

        # 2. Backward Pass: Berechne Gradienten
        loss.backward()

        # 3. Gradient Clipping: Verhindert explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 4. Optimizer Step: Update Weights
        optimizer.step()

        # 5. Learning Rate Scheduler
        scheduler.step()

        # 6. Zero Gradients für nächste Iteration
        optimizer.zero_grad()
```

### 7.4 Wichtige Training-Konzepte

#### Gradient Accumulation

```
Ohne Gradient Accumulation (batch_size=4):
    Batch 1 → Forward → Backward → Update
    Batch 2 → Forward → Backward → Update
    ...

Mit Gradient Accumulation (batch_size=4, accumulation=4):
    Batch 1 → Forward → Backward (sammle Gradienten)
    Batch 2 → Forward → Backward (sammle Gradienten)
    Batch 3 → Forward → Backward (sammle Gradienten)
    Batch 4 → Forward → Backward (sammle Gradienten) → Update

    Effekt: Wie batch_size=16, aber nur Memory für 4!
```

#### Early Stopping

```python
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,     # Stoppt nach 3 Evals ohne Verbesserung
    early_stopping_threshold=0.001 # Minimale Verbesserung
)
```

**Warum Early Stopping?**

- Verhindert Overfitting
- Spart Trainingszeit
- Wählt automatisch besten Checkpoint

---

## 8. Evaluation & Metriken

### 8.1 Datei: `evaluation/metrics.py`

Berechnet verschiedene Qualitäts-Metriken.

### 8.2 Metriken erklärt

#### Exact Match Accuracy

```
Prediction: "J06.9"  vs  Reference: "J06.9"  → Match! ✓
Prediction: "J06.1"  vs  Reference: "J06.9"  → Kein Match ✗
```

#### Prefix Match Accuracy

Berücksichtigt die ICD-10 Hierarchie:

```
3-Char Prefix:
  "J06.9" und "J06.1" → "J06" = "J06" → Match! ✓

1-Char Prefix (Hauptkategorie):
  "J06.9" und "J10.0" → "J" = "J" → Match! ✓
```

#### Precision, Recall, F1

```
                    Tatsächliche Klasse
                    Positiv    Negativ
Vorhergesagt  ┌──────────┬──────────┐
Positiv       │    TP    │    FP    │  ← Precision = TP/(TP+FP)
              ├──────────┼──────────┤
Negativ       │    FN    │    TN    │
              └──────────┴──────────┘
                    ↑
              Recall = TP/(TP+FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Wann welche Metrik?**

- **Precision wichtig:** Wenn falsche Positive teuer sind (z.B. unnötige Behandlung)
- **Recall wichtig:** Wenn falsche Negative teuer sind (z.B. übersehene Krankheit)
- **F1:** Balancierter Trade-off

### 8.3 Performance-Metriken

```python
metrics = {
    "latency_seconds": 0.05,         # Zeit pro Prediction
    "throughput_samples_per_sec": 20, # Predictions pro Sekunde
    "tokens_per_second": 100,         # Token-Generierungsrate
}
```

---

## 9. Experiment-Workflow

### 9.1 Schritt-für-Schritt Anleitung

#### Schritt 1: Environment einrichten

```bash
# Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Dependencies installieren
pip install -e ".[dev]"
```

#### Schritt 2: Memory-Optimierung aktivieren (WICHTIG!)

```bash
# Setze PyTorch Environment Variable für besseres Memory Management
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Für permanente Aktivierung in ~/.bashrc oder ~/.zshrc:
echo 'export PYTORCH_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
```

#### Schritt 3: Nur LLM Evaluation (ohne Training)

```bash
# Testet nur große Modelle (zero-shot)
# Nützlich für schnellen Test der LLM-Performance
python main.py --experiment baseline
```

#### Schritt 4: Vollständiger Durchlauf (LLMs + SLM Training)

```bash
# PHASE 1: Evaluiert LLMs (Llama 8B, Mistral 7B)
# PHASE 2: Trainiert und evaluiert SLMs (Llama 3B, Qwen 3B)
python main.py --experiment full
```

#### Schritt 5: Nur SLM Training (ohne LLM Evaluation)

```bash
python main.py --experiment training
```

#### Schritt 6: Ergebnisse analysieren

```bash
# Öffne generierten Report
open outputs/reports/evaluation_report.html

# Oder inspiziere results.json
cat outputs/reports/results.json
```

### 9.2 CLI-Optionen

```bash
python main.py --help

Options:
  --experiment {baseline,training,full}
  --skip-training          # Nutze existierendes Modell
  --model-llm MODEL_NAME   # Override LLM
  --model-slm MODEL_NAME   # Override SLM
  --config PATH            # Eigene Config-Datei
```

### 9.3 Modelle anpassen

In `config/base_config.py`:

```python
# === LLMs: Große Modelle (zero-shot) ===
llm_models = [
    ModelInstanceConfig(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        size="8B",
        description="Large reference model",
        load_in_4bit=True,
    ),
    ModelInstanceConfig(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        size="7B",
        description="Medium reference model",
        load_in_4bit=True,
    ),
]

# === SLMs: Kleine Modelle (werden finetuned) ===
slm_models = [
    ModelInstanceConfig(
        name="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        description="Compact model for finetuning",
        load_in_4bit=False,  # Full precision für Training
        dtype="bfloat16",
    ),
    ModelInstanceConfig(
        name="Qwen/Qwen2.5-3B-Instruct",
        size="3B",
        description="Alternative compact model",
        load_in_4bit=False,
        dtype="bfloat16",
    ),
]
```

**Für kleinere GPUs (<16GB VRAM):**
```python
# Nutze nur ein LLM und ein SLM
llm_models = [ModelInstanceConfig(name="mistralai/Mistral-7B-Instruct-v0.3", size="7B", ...)]
slm_models = [ModelInstanceConfig(name="Qwen/Qwen2.5-3B-Instruct", size="3B", ...)]
```

> **Hinweis:** Für `meta-llama` Modelle müssen Sie die Nutzungsbedingungen auf
> HuggingFace akzeptieren: https://huggingface.co/meta-llama

---

## 10. Best Practices & Troubleshooting

### 10.1 Memory-Probleme

**Symptom:** `CUDA out of memory`

**Neue verbesserte Memory-Management-Lösung:**

```bash
# 1. Setze PyTorch Environment Variable (WICHTIG!)
export PYTORCH_ALLOC_CONF=expandable_segments:True
python main.py --experiment full
```

**In der Config anpassen:**

```python
# 2. Kleinere Batch Size
training.per_device_train_batch_size = 16  # Statt 32

# 3. Mehr Gradient Accumulation
training.gradient_accumulation_steps = 2  # Statt 1

# 4. Aktiviere 4-bit Quantisierung für LLMs (bereits Standard)
llm_models = [
    ModelInstanceConfig(..., load_in_4bit=True)
]

# 5. Gradient Checkpointing bei Bedarf
training.gradient_checkpointing = True
```

**Die neue Pipeline cleaned automatisch:**
- Nach jedem LLM Evaluation: `aggressive_memory_cleanup()`
- Nach jedem SLM Training: `aggressive_memory_cleanup()`
- GPU Memory wird zwischen Modellen vollständig freigegeben

### 10.2 Training konvergiert nicht

**Symptom:** Loss sinkt nicht

**Lösungen:**

```python
# 1. Learning Rate anpassen
training.learning_rate = 1e-5  # Versuche kleinere Werte

# 2. Mehr Warmup
training.warmup_steps = 1000

# 3. Prüfe Daten
# Sind Labels korrekt? Ist Formatierung konsistent?
```

### 10.3 Overfitting

**Symptom:** Train-Loss sinkt, Val-Loss steigt

**Lösungen:**

```python
# 1. Mehr Regularisierung
training.weight_decay = 0.05
training.lora_dropout = 0.1

# 2. Weniger Kapazität
training.lora_r = 8

# 3. Früher stoppen
training.early_stopping_patience = 2
```

### 10.4 Reproduzierbarkeit

Für identische Ergebnisse bei jedem Lauf:

```python
# In config setzen:
experiment.seed = 42
experiment.deterministic = True

# Aber Achtung: deterministic=True macht Training ~10% langsamer!
```

---

## 📚 Weitere Ressourcen

### Papers

- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Transformer: https://arxiv.org/abs/1706.03762

### Dokumentation

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PEFT (LoRA): https://huggingface.co/docs/peft
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

### ICD-10

- WHO ICD-10: https://icd.who.int/browse10/2019/en
- DIMDI (Deutschland): https://www.bfarm.de/DE/Kodiersysteme/Klassifikationen/ICD/ICD-10-GM

---

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle Feature Branch: `git checkout -b feature/meine-feature`
3. Committe Änderungen: `git commit -m "Add: Meine neue Feature"`
4. Push zum Branch: `git push origin feature/meine-feature`
5. Erstelle Pull Request

---

## 📄 Lizenz

MIT License - siehe LICENSE Datei

---

_Dieser Guide wurde erstellt, um Machine Learning Engineering Best Practices zu vermitteln.
Bei Fragen oder Problemen bitte ein Issue erstellen._
