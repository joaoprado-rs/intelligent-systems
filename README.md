# Intelligent Systems - UFABC

Study repository for **Sistemas Inteligentes (ESZI014-17)**, an undergraduate
course at the Federal University of ABC (UFABC), Brazil. It holds the lab
assignments I implemented in Python with pandas and scikit-learn.

## About the course

| | |
|---|---|
| **Code** | ESZI014-17 |
| **Workload (TPEI)** | 3-1-0-4: 3 h theory, 1 h practice, 0 h extension and 4 h individual study per week |
| **Recommended courses** | Structured Programming; Linear Algebra |

**Objective.** To introduce the tools usually associated with the notion of
intelligent systems and apply them to engineering problems.

**Syllabus.**

- Introduction to artificial neural networks
- The learning process
- Single-layer perceptrons
- Multilayer perceptrons
- Radial basis function networks
- Support vector machines
- Principal component analysis
- Self-organizing maps
- Genetic algorithms

**Basic bibliography.**

- BRAGA, A. P. *Redes Neurais Artificiais: Teoria e Aplicações*. 2nd ed. LTC, 2007.
- HAYKIN, S. *Redes Neurais: Princípios e Prática*. 2nd ed. Bookman, 1999.
- HU, Y. H.; HWANG, J. N. *Handbook of Neural Network Signal Processing*. CRC, 2001.

**Complementary bibliography.**

- BACK, T.; FOGEL, D. B.; MICHALEWICZ, Z. *Evolutionary Computation 1: Basic Algorithms and Operators*. Taylor & Francis, 2000.
- FAUSETT, L. V. *Fundamentals of Neural Networks: Architectures, Algorithms and Applications*. Prentice Hall, 1993.
- GOLDBERG, D. E. *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley, 1989.
- HAYKIN, S. *Neural Networks and Learning Machines*. Prentice Hall, 2008.
- SIVANANDAM, S. N.; DEEPA, S. N. *Introduction to Genetic Algorithms*. Springer, 2007.

Source: [UFABC undergraduate course catalog 2024–2025](https://prograd.ufabc.edu.br/pdf/catalogo_disciplinas_graduacao_2024_2025.pdf), p. 1644.

## Labs

The labs reproduce examples from Lantz, *Machine Learning with R*, and from
Igual & Seguí, *Introduction to Data Science*, translating the R code from the
books into Python and scikit-learn.

| Lab | Folder | Technique | Dataset | Based on |
|---|---|---|---|---|
| 1 | [`knn/`](knn) | k-nearest neighbors classification of breast tumors (benign vs. malignant) | Wisconsin Diagnostic Breast Cancer (`wdbc.data`) | Lantz, ch. 3 |
| 2 | [`decision-tree/`](decision-tree) | Decision tree for credit risk (loan default) | German credit data (`credit.csv`) | Lantz, ch. 5 |
| 3 | [`rlin-rlog/`](rlin-rlog) | Linear regression of medical expenses (`rlin.py`) and logistic regression (`rlog.py`, in progress) | Medical insurance charges (`insurance.csv`); Spanish La Liga 2012–13 matches (`EINF_SI_2026Q3_LAB_III_football_dataset.csv`) | Lantz, ch. 6; Igual & Seguí, ch. 6 |

### Lab 1 — k-nearest neighbors

- Min-max normalization of the 30 numeric features, with z-score
  standardization as an alternative.
- Stratified 80/20 train/test split and a `KNeighborsClassifier` with `k = 5`.
- Evaluation with a classification report, a confusion matrix and accuracy.

### Lab 2 — Decision tree

- One-hot encoding of the categorical columns with `pd.get_dummies`.
- `DecisionTreeClassifier` using entropy, with depth and split limits to
  reduce overfitting.
- Evaluation with a stratified 90/10 holdout and stratified 10-fold
  cross-validation.
- Plot of the top levels of the tree, saved to `arvore.png`.

### Lab 3 — Linear regression

- Dummy coding with `drop_first=True`, matching the reference levels that R's
  `lm()` uses.
- A baseline model, plus an improved model with age squared, an obesity flag
  (BMI ≥ 30) and an obesity × smoker interaction.
- Evaluation with R², MAE and RMSE, using a 90/10 holdout and 10-fold
  cross-validation.

## Running

The scripts need Python 3 with pandas, NumPy, scikit-learn (≥ 1.4, for
`root_mean_squared_error`), Matplotlib and seaborn:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Each script reads its dataset through a relative path, so run it from inside
its lab folder:

```bash
cd decision-tree
python main.py
```

The holdout and k-fold versions of an experiment are switched by commenting
and uncommenting the corresponding blocks. Code comments are in Portuguese.

## References

- LANTZ, Brett. *Machine Learning with R*. Birmingham: Packt Publishing, 2013.
- IGUAL, Laura; SEGUÍ, Santi. *Introduction to Data Science: A Python Approach
  to Concepts, Techniques and Applications*. Cham: Springer, 2017.
