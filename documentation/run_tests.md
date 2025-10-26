## 🧪 How to Run the Tests (with UV)

### **1. Open a terminal in your project root:**

```
cd path/to/your/project-root
```


### **2. Activate the UV virtual environment:**

```
uv venv          # (Only needed once to create it)
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```


### **3. Install dev requirements for backend and frontend:**

```
uv pip install -r backend/requirements-dev.txt
uv pip install -r frontend/requirements-dev.txt
```


### **4. Run backend tests:**

```
cd backend
pytest tests/ -v
```


### **5. Run frontend tests:**

```
cd ../frontend
pytest tests/ -v
```


### **6. (Optional) Run tests with coverage:**

```
cd backend
pytest tests/ --cov=app --cov-report=html
# Open htmlcov/index.html to view coverage report
```


### **7. (Optional) Use the test runner script to run everything:**
Make the script executable
```
chmod +x run_tests.sh
```

```
./run_tests.sh
```


***

**Summary:**
Activate UV, install dev dependencies, then run `pytest` in each app directory (`backend`, `frontend`). For full test automation, use `run_tests.sh`.
