# 🦙 TinyLlama Fine-Tuning with LoRA (PEFT)

This repository demonstrates how to fine-tune [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using Parameter-Efficient Fine-Tuning (PEFT) via LoRA, and then use the trained model for inference.

---

## 📁 Project Structure

├── train.py # Fine-tuning script
├── inference.py # Script to run inference from fine-tuned model
├── project_dataset.json # Dataset in JSON format
├── tinyllama_finetuned/ # Output directory for the fine-tuned model
└── README.md # This file


Dataset Format: project_dataset.json
🚀 How to Train: add data to project_dataset.json
 -> python train.py
 
🤖 How to Use the Fine-Tuned Model (Inference)
python inference.py

📌 Notes
Uses PEFT for lightweight fine-tuning

4-bit quantization greatly reduces memory usage

Works best on GPUs with > 12GB VRAM
