$ErrorActionPreference = "Stop"

Write-Host "1/6 Building AppleSupport pairs..."
python scripts/build_brand_dataset.py

Write-Host "2/6 Discovering intents..."
python scripts/discover_intents.py --k 12
Write-Host "Review data/processed/intent_clusters.csv and update configs/intent_taxonomy.yaml."
Read-Host "Press Enter after taxonomy review"

Write-Host "3/6 Creating training-label candidates..."
python scripts/prepare_training_labels.py --n 500
Write-Host "Label data/processed/train_labels.csv using docs/labeling_guide.md."
Read-Host "Press Enter after training labels are complete"

Write-Host "4/6 Training classifier and building retrieval index..."
python scripts/train_intent.py
python scripts/build_index.py

Write-Host "5/6 Creating held-out golden set..."
python scripts/prepare_golden.py --n 200
Write-Host "Label data/golden/golden_to_label.csv using docs/labeling_guide.md."
Read-Host "Press Enter after golden labels are complete"

Write-Host "6/6 Running evaluation..."
python scripts/evaluate.py
python scripts/failure_analysis.py

Write-Host "Done. Inspect reports/evaluation.json and reports/top_intent_failures.csv."
