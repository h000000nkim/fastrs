# Simars - FastText-based similarity analysis of answers & responses for human raters

Simars is a comprehensive toolkit for analyzing response similarity using FastText embeddings, specifically designed to support human raters in educational assessment and text analysis tasks.

## 🚀 Features

- **Text Preprocessing**: Comprehensive Korean and English text cleaning and tokenization
- **FastText Integration**: Training and fine-tuning of FastText models with Korean support
- **Dimensionality Reduction**: Support for UMAP, PCA, and t-SNE algorithms
- **Clustering**: HDBSCAN clustering for response grouping
- **Interactive Visualization**: Plotly-based interactive scatter plots with multiple visualization modes
- **Jamo Processing**: Advanced Korean text processing with Jamo decomposition

## 📦 Installation

```bash
pip install simars

# Install spaCy English model (required for text processing)
python -m spacy download en_core_web_sm
```

### Development Installation

```bash
git clone https://github.com/h000000nkim/simars.git
cd simars
pip install -e ".[dev]"
```

## 🔧 Dependencies

- **Core**: `gensim`, `numpy`, `pandas`, `scikit-learn`, `umap-learn`
- **NLP**: `jamo`, `pecab`, `spacy`
- **Visualization**: `plotly`
- **Clustering**: `hdbscan`
- **Development**: `pytest`, `ruff`, `mkdocs-material`

## 📖 Quick Start

### Basic Usage

```python
import simars
import numpy as np

# Sample data
answers = np.array([["허무"], ["흡수율"], ["부사어"]])
responses = np.array([
    ["허무", "공허", "무상", "허무감", "초월"],
    ["흡수율", "흡수", "반사율", "알베도"],
    ["부사어", "부사", "수식어", "부가어"]
])
informations = np.array([
    "문학 문제에 대한 정서적 태도",
    "과학 제재의 핵심 개념",
    "문법 성분 분석"
])

# Initialize Simars
analyzer = simars.Fastrs(
    answers=answers,
    responses=responses,
    informations=informations
)

# Preprocess text data
analyzer.preprocess()

# Train FastText model
model = analyzer.train(
    vector_size=100,
    window=5,
    min_count=1,
    epochs=10
)

# Reduce dimensionality
coordinates = analyzer.reduce(method="umap", n_neighbors=5)

# Perform clustering
analyzer.hdbscanize()

# Visualize results
figures = analyzer.visualize()
for fig in figures:
    fig.show()
```

### Advanced Usage with Custom Data Structure

```python
# Using dictionary format
data = {
    "item1": {
        "answer": ["정답1"],
        "response": ["정답1", "오답1", "오답2"],
        "information": "문항 설명"
    },
    "item2": {
        "answer": ["정답2"],
        "response": ["정답2", "유사답", "오답"],
        "information": "다른 문항 설명"
    }
}

analyzer = simars.Fastrs(data=data)
analyzer.preprocess()

# Fine-tune existing model
pretrained_model = simars.util.get_pretrained_model()
analyzer.finetune(model=pretrained_model, epochs=5)

# Advanced reduction with custom parameters
coordinates = analyzer.reduce(
    method="umap",
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine"
)
```

## 🛠️ Core Components

### Fastrs Class

Main class for response similarity analysis:

- `preprocess()`: Clean, tokenize, and prepare text data
- `train()`: Train new FastText model from scratch
- `finetune()`: Fine-tune existing FastText model
- `reduce()`: Reduce embeddings dimensionality
- `hdbscanize()`: Perform clustering analysis
- `visualize()`: Create interactive visualizations

### Item Class

Individual item processor for detailed analysis:

- `clean()`: Text cleaning with customizable options
- `tokenize()`: Korean/English tokenization
- `jamoize()`: Korean Jamo decomposition
- `formatize()`: Prepare data for FastText training

### Preprocessing Module

Advanced text preprocessing functions:

- `clean()`: Multi-option text cleaning
- `tokenize()`: Morphological analysis with PeCab
- `jamoize()`: Korean character decomposition
- `formatize()`: Data formatting for training

### Visualization Module

Interactive plotting with Plotly:

- `scatter()`: Unified scatter plot function
- Multiple plot types: simple, value count, labeled, combined
- Customizable themes and color schemes

## 📊 Visualization Types

### Simple Scatter Plot
Basic 2D visualization highlighting answers vs responses.

### Value Count Scatter Plot  
3D visualization showing response frequency in the z-axis.

### Labeled Scatter Plot
Color-coded visualization based on clustering results.

### Combined Scatter Plot
3D visualization combining clustering and frequency information.

## ⚙️ Configuration

simars uses JSON configuration files for customization:

- `color_schemes.json`: Color themes for visualizations
- `plot_config.json`: Plot layout and styling options
- `reduction_defaults.json`: Default parameters for dimensionality reduction
- `fasttext_defaults.json`: Default FastText training parameters

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest --cov=simars tests/
```

## 📚 Use Cases

- **Educational Assessment**: Analyze student response patterns
- **Content Analysis**: Group similar text responses
- **Quality Assurance**: Identify outlier responses for review
- **Research**: Study response similarity patterns in surveys

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Hoon Kim** - [h000000nkim@gmail.com](mailto:h000000nkim@gmail.com)

## 🔗 Links

- [GitHub Repository](https://github.com/h000000nkim/simars)
- [Documentation](https://h000000nkim.github.io/simars/)
- [PyPI Package](https://pypi.org/project/simars/)

## 📈 Roadmap

- [ ] Support for additional languages
- [ ] Web interface for easy usage
- [ ] Additional clustering algorithms
- [ ] Export functionality for results
- [ ] Integration with popular ML frameworks