flowchart TD
    A[Data Ingestion] --> B[Data Validation]
    B --> C[Feature Extraction]
    C --> D[Model Loading]
    D --> E[Model Training/Tuning]
    E --> F[Model Evaluation]
    F --> G[Model Registry]
    G --> H[Model Deployment]
    H --> I[Monitoring & Feedback]
    I --> A
    
    subgraph CI/CD Pipeline
    J[Unit Tests] --> K[Integration Tests]
    K --> L[Build Docker Image]
    L --> M[Push to Registry]
    M --> N[Deploy to Staging]
    N --> O[Deploy to Production]
    end