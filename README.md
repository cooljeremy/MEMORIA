# MEMORIA: A Large Language Model, Instruction Data and Evaluation Benchmark for Intangible Cultural Heritage

## Overview

This project implements the complete MEMORIA framework as described in our ICLR 2026 paper, featuring:

- **ICHLLM**: ICH-focused Large Language Models (7B and 13B parameters)
- **CHIT Dataset**: Cultural Heritage Instruction Tuning dataset with 158K samples
- **ICHEB Benchmark**: Comprehensive evaluation framework for ICH-aware models
- **Cultural Metrics**: Novel Cultural Score (CS) and Cultural Fidelity (CF) metrics
- **LoRA Fine-tuning**: Efficient adaptation for 147 cultural traditions

##  Architecture

```
memoria/
├── data/                    # CHIT Dataset Construction
│   ├── chit_dataset.py     # Core dataset implementation
│   ├── collectors.py       # Data collection from 13 sources
│   ├── processors.py       # Cultural sensitivity processing
│   └── builders.py         # Task-specific instruction builders
├── models/                  # ICHLLM Model Implementation
│   ├── ichllm.py           # Core ICHLLM models (7B/13B)
│   ├── cultural_adapters.py # Cultural routing and adaptation
│   ├── lora_layers.py      # LoRA fine-tuning implementation
│   └── tokenizers.py       # Cultural tokenization
├── training/                # Training Infrastructure
│   ├── trainer.py          # LoRA fine-tuning trainer
│   └── cultural_loss.py    # Cultural-aware loss functions
└── evaluation/              # ICHEB Evaluation Framework
    ├── icheb.py            # Benchmark implementation
    ├── cultural_metrics.py # CS and CF metrics
    └── task_evaluators.py  # Task-specific evaluators
```

##  Key Features

### 1. CHIT Dataset (158K Samples)
- **13 Data Sources**: UNESCO, Heritage-NER, Craft Knowledge, Cultural Stories, etc.
- **6 Task Categories**: Heritage-NER, Culture-QA, Story-Gen, Translation, Classification, Dialogue
- **147 Cultural Traditions**: Global coverage across all UNESCO ICH domains
- **Sacred Boundary Protection**: Culturally sensitive content filtering

### 2. ICHLLM Models
- **Base Models**: LLaMA-7B and LLaMA-13B with cultural adaptations
- **Cultural Routing**: 147-culture adapter with domain-specific routing
- **LoRA Fine-tuning**: Rank-64, Alpha-16 configuration (~6% trainable parameters)
- **Sacred Boundary Filter**: Protects sensitive cultural content

### 3. ICHEB Benchmark
- **6 Evaluation Tasks**: Comprehensive ICH capability assessment
- **Cultural Metrics**: Novel CS (5-dimensional) and CF (4-dimensional) metrics
- **Multi-level Evaluation**: Task-specific, culture-specific, and overall performance
- **Sacred Level Filtering**: Ethical evaluation with sensitivity controls

### 4. Cultural Score Metrics
- **Structural Authenticity (SA)**: Conformance to culturally-specific narrative architectures
- **Motif Fidelity (MF)**: Semantic integrity of cultural motifs and archetypal entities
- **Linguistic Authenticity (LA)**: Prosodic and stylistic features of traditional narratives
- **Value Alignment (VA)**: Axiological coherence with indigenous epistemological frameworks
- **Transmission Appropriateness (TA)**: Efficacy as vehicles for intergenerational knowledge transfer

### 5. Cultural Fidelity Metrics (Translation Tasks)
- **Conceptual Preservation (CP)**: Semantic integrity of culture-bound concepts during translation
- **Metaphorical Mapping (MM)**: Transfer of figurative language and conceptual metaphor systems
- **Pragmatic Accuracy (PA)**: Preservation of illocutionary force and socio-pragmatic appropriateness
- **Sacred Knowledge Handling (SKH)**: Ethical treatment of epistemologically-restricted content


## Research Contributions

1. **Novel Cultural Metrics**: First comprehensive framework for evaluating cultural authenticity in AI
2. **Large-scale ICH Dataset**: Largest curated dataset for intangible cultural heritage
3. **Cultural-aware Architecture**: Systematic approach to cultural adaptation in LLMs
4. **Ethical AI Framework**: Comprehensive safeguards for cultural sensitivity
5. **Benchmark Suite**: Standardized evaluation for cultural AI systems
