# CV-Job Bipartite Graph Analysis: Link Prediction and Node Classification

A comprehensive graph-based analysis of the job recruitment process, modeling the relationship between applicant CVs and job postings as a bipartite graph. This project integrates machine learning, natural language processing, and network analysis to predict job matches and classify candidate experience levels.

## 📋 Project Overview

This project implements a multi-layered analysis framework that:

- **Constructs** a bipartite graph from real applicant and job data
- **Analyzes** structural properties and network patterns
- **Detects** communities within CV-Job networks and skill relationships
- **Predicts** missing CV-Job links using LLM embeddings and classical algorithms
- **Classifies** CVs by experience level using graph features and machine learning
- **Analyzes** behavioral patterns comparing stated interests with actual job viewing behavior

## 🎯 Key Features

### 1. **Graph Construction:**

- Bipartite graph with CV and Job nodes
- Edges based on skill overlap (weighted by matching skills)
- Real applicant viewing behavior integration
- Multi-source data enrichment

### 2. **Structural Analysis:**

- Graph metrics: nodes, edges, density, connectivity
- Centrality measures: degree, betweenness, closeness
- Component analysis and network visualization
- Key node identification

### 3. **Community Detection:**

- **Global communities**: Louvain algorithm on CV-Job graph
- **Internal communities**: Skill co-occurrence networks
- Modularity scoring and semantic interpretation

### 4. **Link Prediction:**

- **LLM-based**: Sentence transformers for semantic similarity
- **Classical methods**: Common Neighbors, Adamic-Adar Index
- Graph enrichment with predicted links
- Comparative validation

### 5. **Node Classification:**

- Experience level prediction (junior/intermediate/senior)
- Random Forest and Logistic Regression models
- Feature importance analysis
- Polyvalence vs. specialization profiling

### 6. **Behavioral Analysis:**

- Real applicant job viewing patterns
- Interest-behavior alignment analysis
- View duration weighting

## 📊 Data Sources

**Kaggle Dataset**: [Job Recommendation Datasets](https://www.kaggle.com/datasets/kandij/job-recommendation-datasets)

The project integrates **5 CSV datasets** from Kaggle:

| Dataset | Description | Records |
| ------- | ----------- | ------- |
| `Combined_Jobs_Final.csv` | Job postings with descriptions and requirements | Variable |
| `Experience.csv` | Applicant work history and experience records | Variable |
| `Job_Views.csv` | Real applicant job viewing behavior | Variable |
| `Positions_Of_Interest.csv` | Stated job preferences by applicants | Variable |
| `job_data.csv` | Additional job descriptions and metadata | Variable |

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

```bash
# Install required packages:
pip install pandas numpy networkx matplotlib seaborn scikit-learn sentence-transformers torch
```

### Data Setup

1. Download the dataset from Kaggle: [Job Recommendation Datasets](https://www.kaggle.com/datasets/kandij/job-recommendation-datasets)

1. **Option A - Local Setup**: Place all CSV files in a `data/` directory:

   ```text
   graph/
   ├── cv-job-graph-analysis.ipynb
   ├── data/
   │   ├── Combined_Jobs_Final.csv
   │   ├── Experience.csv
   │   ├── Job_Views.csv
   │   ├── Positions_Of_Interest.csv
   │   └── job_data.csv
   └── README.md
   ```

1. **Option B - Kaggle Notebook**: Upload the notebook directly to Kaggle and use the built-in dataset integration. Update file paths in the notebook from `/kaggle/input/datasets/kandij/job-recommendation-datasets/` to match your setup.

### Running the Analysis

Open the Jupyter notebook:

```bash
jupyter notebook cv-job-graph-analysis.ipynb
```

**Note**: Update the file paths in the notebook to match your data location (Kaggle or local).

Run cells sequentially from top to bottom. The notebook is organized in logical sections:

1. Data Loading & Preprocessing
2. CV Construction from Real Data
3. Graph Construction
4. Structural Analysis
5. Community Detection
6. Link Prediction
7. Node Classification
8. Visualizations
9. Comprehensive Summary
10. Advanced Profile Analysis

## 🔧 Technical Architecture

### Core Technologies

- **Graph Analysis**: NetworkX
- **Machine Learning**: scikit-learn (Random Forest, Logistic Regression)
- **NLP/Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

### Performance Optimizations

The notebook includes **optimized algorithms** for large-scale processing:

- ✅ **Precomputed skill sets**: Fast set intersection for overlap calculation
- ✅ **O(1) dictionary lookups**: Job ID mapping for viewing behavior
- ✅ **Batch edge addition**: `add_edges_from()` for better performance
- ✅ **Efficient iteration**: Direct DataFrame iteration instead of graph node searches

**Performance gains**: 10-100x faster execution on large datasets

## 📈 Key Results & Insights

### Graph Statistics

- **Nodes**: Real applicant CVs + Job postings
- **Edges**: Skill-based matches + viewing behavior
- **Communities**: Natural clustering of similar profiles and positions

### Link Prediction

- **LLM predictions**: High semantic similarity matches
- **Classical predictions**: Structural pattern-based matches
- **Validation**: Cross-method comparison for robustness

### Classification Performance

- **Random Forest**: High accuracy on experience level prediction
- **Feature Importance**: Combination of structural and semantic features
- **Confusion Matrix**: Detailed performance by class

### Profile Analysis

- **Polyvalent CVs**: Versatile candidates with diverse skill sets
- **Specialized CVs**: Domain experts with focused expertise
- **Career Patterns**: Experience level correlations

## 🎓 Academic Context

This project fulfills the requirements for the **M2 Graph Analysis Project**:

✅ Bipartite CV-Job graph construction  
✅ Structural analysis (degree, centrality, components)  
✅ Community detection (global + internal)  
✅ LLM-based link prediction with embeddings  
✅ Classical link prediction algorithms  
✅ Node feature engineering  
✅ Multi-method classification  
✅ Results analysis and recommendations  

## 📝 Deliverables

- ✅ **Jupyter Notebook**: Complete reproducible analysis
- ✅ **Visualizations**: Graph layouts, distributions, confusion matrices
- ✅ **Feature Engineering**: Multi-dimensional node representations
- ✅ **Model Comparison**: Multiple classification approaches
- ✅ **Comprehensive Summary**: Insights and recommendations

## 🔍 Use Cases

### Recruitment Systems

- Automated CV-Job matching
- Candidate recommendation engines
- Skill gap analysis

### Career Development

- Profile polyvalence assessment
- Career path identification
- Skill clustering insights

### Market Analysis

- Recruitment trend detection
- Skill demand patterns
- Community structure understanding

## ⚙️ Configuration

### Adjustable Parameters

```python
# CV construction
NUM_APPLICANTS = 1000  # Limit for performance

# Skill matching
SKILL_THRESHOLD = 3  # Minimum matching skills for edge

# Link prediction
SIMILARITY_THRESHOLD = 0.7  # LLM similarity cutoff
TOP_LLM_PREDICTIONS = 50   # Links to add from LLM

# Sampling
CV_SAMPLE_SIZE = 500   # For embeddings
JOB_SAMPLE_SIZE = 500  # For embeddings
```

## 📊 Output Examples

### Graph Metrics

```text
Number of nodes: 1500+
Number of edges: 5000+
Graph density: 0.00XX
Communities detected: 10+
```

### Classification Accuracy

```text
Random Forest: 0.8XX
Logistic Regression: 0.7XX
```

### Top Features

1. Degree centrality
2. Number of skills
3. Community membership
4. Embedding dimensions
5. Betweenness centrality

## 🛠️ Troubleshooting

### Common Issues

**1. Memory errors with large datasets:**

- Reduce `NUM_APPLICANTS` parameter
- Limit CV/Job sample sizes for embeddings
- Use batch processing for edge creation

**2. Slow execution:**

- Ensure optimized cells are used (with precomputed sets)
- Reduce centrality computation samples
- Skip visualization cells if not needed

**3. Missing dependencies:**

```bash
pip install --upgrade sentence-transformers
pip install torch torchvision
```

## 📚 References

### Algorithms

- **Community Detection**: Louvain algorithm (Blondel et al., 2008)
- **Embeddings**: Sentence-BERT (Reimers & Gurevych, 2019)
- **Link Prediction**: Adamic-Adar, Common Neighbors

### Libraries

- NetworkX: <https://networkx.org/>
- sentence-transformers: <https://www.sbert.net/>
- scikit-learn: <https://scikit-learn.org/>

## 👥 Author

Skander Adam Afi

## 📄 License

Academic project - For educational purposes

## 🙏 Acknowledgments

- Real applicant data sources
- Open-source community (NetworkX, scikit-learn, transformers)
- Academic supervisors and reviewers

---

**Last Updated**: February 11, 2026

For questions or issues, please refer to the comprehensive notebook documentation.
