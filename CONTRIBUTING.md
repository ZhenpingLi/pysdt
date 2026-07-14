# Contributing to MTMA Tool (pysdt)

First off, thank you for considering contributing to the Model-Based Telemetry Monitoring and Analysis (MTMA) tool! It's people like you that make open source such a great community.

We welcome contributions from everyone, whether you're a satellite engineer, a data scientist, a software developer, or someone interested in aerospace technology.

---

## How Can I Contribute?

### Reporting Bugs
If you find a bug, please help us by [opening an issue](https://github.com/zhenpingli/pysdt/issues). 
* Use a clear and descriptive title.
* Describe the exact steps to reproduce the problem.
* Explain which behavior you expected to see and what you actually saw.

### Suggesting Enhancements
Have an idea for a new algorithm or a better visualization?
* Open an enhancement request in the issues.
* Describe the goal of the feature and why it would be useful for telemetry monitoring.

### Pull Requests (Code & Docs)
We love pull requests! If you want to fix a bug or add a feature:
1. **Fork the repo** and create your branch from `main`.
2. **Setup your environment** (see below).
3. **Write your code**.
4. **Add docstrings**. We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html).
5. **Update documentation** if you're changing functionality.
6. **Submit the PR** with a clear description of your changes.

---

## Development Setup

To get started with development:

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pysdt.git
   cd pysdt
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary science libraries:
   ```bash
   pip install numpy scipy scikit-learn matplotlib ray
   ```

---

## Coding Standards

* **Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
* **Documentation**: All new functions and classes must have Google-style docstrings.
* **Modularity**: We use a pluggable architecture. If you're adding an algorithm, implement the `AlgorithmFactory`, `TrainingWorker`, and `DataTrend` interfaces.

---

## Our Values

* **Openness**: Anyone is welcome to contribute.
* **Scientific Integrity**: We prioritize mathematical accuracy and numerical stability.
* **Community**: Be respectful and helpful to other contributors.

## Contact

If you have questions or want to discuss a major architectural change, feel free to reach out to the project lead at [zpli1@yahoo.com](mailto:zpli1@yahoo.com).

Happy Coding!
