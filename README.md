# Cerberus: Agentic AI Voice Spoofing Detection System

This repository consolidates the baseline components into an executable pipeline. It implements the previously missing anti-spoofing classifier (Task 1) to enable end-to-end processing.

## 1. Project Status & Deliverables

| Task | Previous State | Current Implementation |
| :--- | :--- | :--- |
| **Task 0: Dataset Builder** | Prototyped in notebooks (`VOICE CLONER/Cloner.ipynb`, `tts.ipynb`). | Rebuilt: `src/task0_cloning/build_adversarial_dataset.py` constructs a 3-class adversarial dataset using raw audio files. |
| **Task 1: Classifier** | Missing (`classifier_tool.py` relied on a mock heuristic). | Built: 3-class MLP utilizing 224-dim adversarial features. |
| **Task 2: Agent Workflow** | State machine implemented. | Enhanced: Consumes Task 1 output, integrates Out-of-Distribution (OOD) checks, and evaluates robustness context. |
| **Task 3: Robustness** | Not built. | Built: `src/task3_robustness/robustness_eval.py` (executes noise SNR sweeps and out-of-sample generalization tests). |
| **Task 4: Interface** | Not built. | Built: FastAPI + WebSocket dashboard for live stream processing, probability rendering, and pipeline control. |

---

## 2. Technical Specifications & Problem Statement

**Core Objective:** Deploy a system capable of voice cloning, AI-generated speech detection, autonomous threat assessment, robustness evaluation under noise, and real-time decision explanation.

**Dependency & Architecture Constraints:**
* **Voice Cloning:** XTTS, Coqui TTS, FishSpeech
* **Anti-Spoofing Baselines:** ECAPA-TDNN, AASIST, RawNet2, WavLM
* **Feature Extraction:** MFCC, Spectrogram, SSL embeddings
* **Agentic Framework:** LangGraph
* **Audio Processing:** torchaudio, librosa, SpeechBrain, Hugging Face Transformers
* **Frontend:** Streamlit or Gradio
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, EER.

### Note on Task 1 Model Selection
Resource and temporal constraints preclude the training of end-to-end deep networks (AASIST/RawNet2/WavLM) over raw waveforms on ASVspoof-scale data. Task 1 currently relies on a lightweight **MLP over MFCC statistics**. This model trains efficiently on local CPUs utilizing the Task 0 cloner output. This is a baseline limitation; the architecture (`src/task1_classifier/model.py`) is modular and designed to be swapped for a heavy-weight model if GPU infrastructure and large-scale datasets become available.

---

## 3. Architecture Pipeline

```text
User Audio
   │
   ▼
Task 0: Adversarial Dataset Builder  ──generates──▶  data/real/, data/fake_synth/, data/fake_distort/
   │                                                         │
   │                                                         ▼
   │                                          Task 1: Train 3-Class Classifier
   │                                              (224-dim features -> MLP)
   │                                                         │
   │                                                         ▼  models/task1_classifier.pt
   ▼
Task 2: LangGraph Agent (src/task2_agent/graph.py)
   extract_features -> ood_check -> classify -> robustness_context
        -> [confident? skip : secondary_checks] -> threat_assessment -> explain -> JSON verdict
   │
   ▼
Task 3: Robustness Evaluation (noise SNR sweep, unseen-attack generalization test)
   │
   ▼
Task 4: Demo App (src/task4_demo/app.py) — upload audio, render verdict.
