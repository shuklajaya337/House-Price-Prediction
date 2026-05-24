# Contributing to House Price Prediction

We are excited that you are interested in contributing to this project! Please review the guidelines below to ensure a smooth and productive workflow.

## 📁 Repository Structure

Our repository follows a structured layout:
- `data/`: Contains dataset files (`housing_new.csv`).
- `src/`: Core Python source code (`main.py` Streamlit app).
- `notebooks/`: Jupyter Notebook files for research and experimentation.
- `tests/`: Automated unit tests.
- `images/`: Documentation assets and screenshots.

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd House-Price-Prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8  # Development packages
   ```

4. **Run the Streamlit App locally:**
   ```bash
   streamlit run src/main.py
   ```

## 🧪 Testing and Quality

We use `pytest` for unit testing and `flake8` for linting.

- **To run tests:**
  ```bash
  pytest
  ```
- **To run the linter:**
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```

Ensure all tests pass and there are no linting errors before submitting a pull request.

## 🚀 Submitting Changes

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request (PR) against the `main` or `master` branch.
