# Study Buddy Backend

FastAPI backend for the Study Buddy application.

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update .env file with your configurations

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation

Access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Running Tests

```bash
pytest app/tests
```
