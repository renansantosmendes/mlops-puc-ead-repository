# MLOps - Fetal Health Prediction

Projeto de Machine Learning Operations (MLOps) desenvolvido para a PUC-Rio que implementa um pipeline completo de treinamento, registro e deploy de modelos para predição de saúde fetal.

## 📋 Visão Geral

Este projeto demonstra as melhores práticas de MLOps, incluindo:

- **Treinamento de modelos**: Usando Keras/TensorFlow
- **Rastreamento de experimentos**: MLflow para logging de modelos e métricas
- **Versionamento**: Git para controle de versão do código
- **Testes automatizados**: Pytest para garantir qualidade
- **CI/CD Pipeline**: GitHub Actions para automação
- **API REST**: FastAPI para servir predições
- **Containerização**: Docker para deploy

---

## 📁 Estrutura do Projeto

```
puc_ead/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml              # Pipeline CI/CD do GitHub Actions
│
├── train.py                          # Script principal de treinamento
├── test_train.py                     # Testes unitários do treinamento
├── main.py                           # API FastAPI para predições
├── locustfile.py                     # Load testing com Locust
│
├── train_notebook.ipynb              # Notebook de desenvolvimento
├── model_registry.ipynb              # Gerenciamento de registry do MLflow
├── automl_com_optuna.ipynb           # Experimentos com AutoML
│
├── requirements.txt                  # Dependências Python
├── environment.yml                   # Arquivo de ambiente Conda
├── Dockerfile                        # Configuração para containerização
├── .gitignore                        # Arquivo para ignorar arquivos no Git
│
├── mlruns/                           # Diretório de execuções do MLflow
├── .venv/ / venv/                    # Ambientes virtuais Python
│
└── README.md                         # Este arquivo
```

---

## 🛠️ Tecnologias Utilizadas

### Core ML/Data Science
- **NumPy** 2.1.3 - Computação numérica
- **Pandas** 2.2.3 - Manipulação de dados
- **Scikit-learn** 1.5.2 - Machine Learning
- **Matplotlib** 3.10.6 - Visualização

### Deep Learning
- **TensorFlow** 2.21.0 - Framework de Deep Learning
- **Keras** 3.13.2 - API de alto nível para redes neurais

### MLOps
- **MLflow** 3.10.1 - Rastreamento e registro de modelos
- **DVC** - Controle de versão de dados (opcional)

### Web Framework
- **FastAPI** 0.128.0 - API REST moderna
- **Pydantic** 2.12.5 - Validação de dados
- **Uvicorn** 0.40.0 - Servidor ASGI

### Testing & Quality
- **Pytest** 8.3.3 - Framework de testes
- **Locust** - Load testing

---

## 🚀 Instalação

### Opção 1: Usando Conda (Recomendado)

```bash
# Criar ambiente conda
conda env create -f environment.yml

# Ativar ambiente
conda activate mlops-env
```

### Opção 2: Usando pip e venv

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente
# No Windows:
.venv\Scripts\activate
# No macOS/Linux:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

---

## 📊 Estrutura dos Arquivos Principais

### `train.py`
Script principal para treinar o modelo de classificação de saúde fetal.

**Funções principais:**
- `reset_seeds()` - Garante reprodutibilidade com seeds
- `read_data()` - Lê dados de saúde fetal de repositório remoto
- `process_data()` - Normaliza e divide dados em train/test
- `create_model()` - Cria arquitetura neural network
- `config_mlflow()` - Configura conexão com MLflow
- `configure_mlflow_keras()` - Configura autolog do Keras
- `train_model()` - Treina modelo com MLflow tracking

**Uso:**
```bash
python train.py
```

### `test_train.py`
Testes unitários para validar as funções do `train.py`.

**Testes inclusos:**
- `test_read_data()` - Valida leitura de dados
- `test_create_model()` - Valida criação do modelo
- `test_train_model()` - Valida treinamento

**Executar testes:**
```bash
pytest test_train.py -v
```

### `main.py`
API REST para servir predições usando FastAPI.

**Endpoints:**
- `GET /` - Health check da API
- `POST /predict` - Realiza predição de saúde fetal

**Modelo de entrada:**
```json
{
  "accelerations": 0.0,
  "fetal_movement": 0.0,
  "uterine_contractions": 0.0,
  "severe_decelerations": 0.0
}
```

**Iniciar API:**
```bash
uvicorn main:app --reload
```

### `locustfile.py`
Load testing para simular múltiplos usuários fazendo requisições.

**Executar teste de carga:**
```bash
locust -f locustfile.py
```

---

## 🔄 Pipeline CI/CD (GitHub Actions)

O arquivo `.github/workflows/pipeline.yml` define a automação:

### Etapas do Pipeline:

1. **Test Train** ✅
   - Setup Python 3.12
   - Instala dependências
   - Executa testes com pytest

2. **Train Pipeline** 🤖
   - Treina modelo em ambiente ubuntu
   - Registra métricas e modelo no MLflow

3. **Build Image** 🐳
   - Constrói imagem Docker
   - Faz push para Docker Registry

**Triggers:**
- Push para qualquer branch
- Pull Request para `layers`, `branch_01`, `main`
- Disparo manual via `workflow_dispatch`
- Webhook customizado

---

## 🔐 Configuração de Credenciais

### MLflow (DagsHub)

Configure as variáveis de ambiente em `train.py` e `main.py`:

```python
os.environ['MLFLOW_TRACKING_USERNAME'] = 'seu_username'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'seu_token'
mlflow.set_tracking_uri('https://dagshub.com/seu_usuario/seu_repo.mlflow')
```

### Docker Hub (GitHub Actions)

Configure os secrets no repositório GitHub:
- `DOCKER_USER` - Username do Docker Hub
- `DOCKER_PASSWORD` - Token de acesso
- `DOCKER_IMAGE_NAME` - Nome da imagem

---

## 📦 Arquivos de Configuração

### `requirements.txt`
Lista de dependências Python com versões fixas. Usado por:
- `pip install`
- GitHub Actions
- Docker

### `environment.yml`
Configuração do Conda com canais e dependências. Cria ambiente chamado `mlops-env`.

### `Dockerfile`
Containeriza a aplicação FastAPI para deploy.

### `.gitignore`
Ignora arquivos desnecessários:
- `__pycache__/`, `*.pyc` - Bytecode Python
- `.venv/`, `venv/` - Ambientes virtuais
- `.pytest_cache/`, `.coverage/` - Testes
- `mlruns/` - Execuções do MLflow
- Dados e arquivos temporários

---

## 🧪 Executando Testes

### Rodar todos os testes
```bash
pytest test_train.py -v
```

### Rodar teste específico
```bash
pytest test_train.py::test_read_data -v
```

### Cobertura de testes
```bash
pytest test_train.py --cov=.
```

---

## 📈 Monitoramento com MLflow

### Acessar UI do MLflow (local)
```bash
mlflow ui
# Abre em http://localhost:5000
```

### Registrar modelo
O modelo é automaticamente registrado quando `is_train=True` em `train_model()`.

---

## 🐳 Docker

### Build da imagem
```bash
docker build -t seu_usuario/mlops-puc-ead:latest .
```

### Executar container
```bash
docker run -p 8000:8000 seu_usuario/mlops-puc-ead:latest
```

---

## 📝 Notebooks

### `train_notebook.ipynb`
Desenvolvimento interativo do pipeline de treinamento.

### `model_registry.ipynb`
Gerenciamento e versionamento de modelos no MLflow.

### `automl_com_optuna.ipynb`
Experiments com AutoML usando Optuna para otimização de hiperparâmetros.

---

## 🔄 Workflow de Desenvolvimento

1. **Desenvolver localmente**
   ```bash
   conda activate mlops-env
   python train.py
   ```

2. **Testar mudanças**
   ```bash
   pytest test_train.py -v
   ```

3. **Fazer commit e push**
   ```bash
   git add .
   git commit -m "descricao das mudancas"
   git push
   ```

4. **GitHub Actions automáticamente:**
   - Roda testes
   - Treina modelo
   - Constrói imagem Docker
   - Faz deploy

---

## 📊 Dataset

O projeto utiliza o dataset de **Fetal Health Status**:
- **Fonte**: Repositório GitHub (`renansantosmendes/lectures-cdas-2023`)
- **Arquivos**: `fetal_health_reduced.csv`
- **Features**: Acelerações, movimento fetal, contrações, desacelerações
- **Target**: Classe de saúde fetal (1, 2, 3)

---

## 🎯 Próximos Passos

- [ ] Implementar versionamento de dados com DVC
- [ ] Adicionar mais métricas de avaliação
- [ ] Configurar alertas automáticos
- [ ] Implementar A/B testing
- [ ] Adicionar documentação de API com Swagger
- [ ] Configurar monitoramento em produção

---

## 👥 Contribuindo

1. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
2. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
3. Push para a branch (`git push origin feature/AmazingFeature`)
4. Abra um Pull Request

---

## 📄 Licença

Este projeto é parte do curso de EAD da PUC-Rio.

---

## 📞 Contato

Desenvolvido por: **Renan Santos Mendes**  
Repositório: `mlops-puc-ead-repository`  
GitHub: [@renansantosmendes](https://github.com/renansantosmendes)

---

## 📚 Referências

- [MLflow Documentation](https://mlflow.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)

---

**Última atualização**: Março 2026
