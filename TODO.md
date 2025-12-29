# Project Status

## Completed

### Initial Setup (2025-12-29)
- [x] Monorepo structure: `backend/` (FastAPI), `frontend/` (React/TypeScript)
- [x] FastAPI backend with `/api/health` endpoint and static file serving
- [x] React frontend with TypeScript (Vite build)
- [x] Dockerfile for containerized deployment
- [x] GitHub Actions CI/CD workflow (`backend`, `frontend`, `deploy` jobs)
- [x] Artifact upload for logs in each job (90-day retention)
- [x] `data/runs.csv` schema for ground truth labels

### Azure Infrastructure (2025-12-29)
- [x] Provisioning script: `scripts/azure_setup.sh`
- [x] Auto-registration of Azure resource providers
- [x] Azure Container Registry (ACR) + App Service Plan + Web App
- [x] GitHub Actions deploy job using `azure/login@v2` with service principal
- [x] Container logging enabled
- [x] Documentation: `docs/DEPLOYMENT.md`

### Evaluation Scripts (2025-12-29)
- [x] `scripts/download_artifacts.py` - Download GHA artifacts via `gh` CLI
- [x] `scripts/normalize_logs.py` - Normalize logs for analysis
- [x] `scripts/run_logai.py` - Anomaly detection (placeholder implementation)
- [x] `scripts/compute_metrics.py` - Precision/recall/F1 computation
- [x] `scripts/label_runs.py` - Interactive labeling helper

---

# Remaining TODOs

## LogAI Integration

### Current Status
- Skeleton scripts created in `scripts/`
- Placeholder anomaly scoring implemented
- Real LogAI integration TODO

### Next Steps

1. **Install LogAI**
   ```bash
   pip install logai
   # or from source if needed:
   # git clone https://github.com/salesforce/logai && cd logai && pip install -e .
   ```

2. **Update `scripts/run_logai.py`**
   Replace placeholder with actual LogAI anomaly detection:
   - Choose detection algorithm (isolation forest, LSTM, etc.)
   - Configure vectorization/embedding for log lines
   - Tune threshold based on validation set

3. **Training approach options**
   - **Unsupervised**: Train on all logs, detect statistical outliers
   - **Semi-supervised**: Train only on success logs, detect deviation
   - **Supervised**: Use labeled failures for training (needs 40+ runs)

### Recommended LogAI Pipeline
```python
from logai.dataloader.openset_data_loader import OpenSetDataLoader
from logai.preprocess.preprocessor import Preprocessor
from logai.analysis.anomaly_detector import AnomalyDetector

# 1. Load logs
loader = OpenSetDataLoader(...)
logrecord = loader.load_data()

# 2. Preprocess
preprocessor = Preprocessor(...)
logrecord = preprocessor.clean_log(logrecord)

# 3. Detect anomalies
detector = AnomalyDetector(...)
results = detector.predict(logrecord)
```

---

## Failure Injection Scenarios

### Backend Test Failure
Create branch: `experiment/backend-fail`
```python
# backend/tests/test_api.py - add failing test
def test_intentional_failure():
    assert False, "Intentional failure for experiment"
```

### Frontend Build Failure
Create branch: `experiment/frontend-fail`
```typescript
// frontend/src/App.tsx - add type error
const x: number = "not a number";  // TS error
```

### Dependency Failure
Create branch: `experiment/dep-fail`
```
# backend/requirements.txt - add nonexistent package
nonexistent-package-xyz==99.99.99
```

### Deployment Failure
Create branch: `experiment/deploy-fail`
```dockerfile
# Dockerfile - invalid command
RUN exit 1
```

---

## Data Collection Workflow

1. **Run successful pipelines** (10+ runs)
   - Push small commits to main
   - Let CI complete successfully

2. **Inject failures** (15+ runs across classes)
   - Create experiment branches
   - Push and let CI fail
   - Label each run in `data/runs.csv`

3. **Download artifacts**
   ```bash
   python scripts/download_artifacts.py --repo <owner>/<repo>
   ```

4. **Normalize logs**
   ```bash
   python scripts/normalize_logs.py
   ```

5. **Run LogAI**
   ```bash
   python scripts/run_logai.py --threshold 0.5
   ```

6. **Compute metrics**
   ```bash
   python scripts/compute_metrics.py
   ```

---

## Elastic/Dynatrace (Shallow Integration)

### Elastic Observability
- Set up Elastic Cloud trial
- Configure Filebeat to ship CI logs
- Create Kibana dashboards
- Document: screenshots + qualitative observations

### Dynatrace Davis AI
- Set up Dynatrace trial
- Configure log ingestion
- Enable Davis AI anomaly detection
- Document: screenshots + qualitative observations

---

## Definition of Done Checklist

- [ ] CI pipeline works (backend + frontend + deploy)
- [ ] 40+ runs in `data/runs.csv`
- [ ] 15+ failures across classes
- [ ] LogAI produces scores reproducibly
- [ ] Metrics computed (precision/recall/F1)
- [ ] Elastic integration documented
- [ ] Dynatrace integration documented
