# Sentiment Classification on Movie & Product Reviews 🎬🛍️

This project implements a sentiment classification pipeline on short reviews collected from **IMDb, Amazon, and Yelp**. The goal is to predict whether a review expresses a positive or negative sentiment using a **Bag-of-Words + Logistic Regression** approach.  


## 📌 Objectives
- Preprocess and clean raw review text  
- Extract features using **TF–IDF weighted Bag-of-Words**  
- Train and tune a **Logistic Regression classifier**  
- Evaluate model performance on validation and test sets  
- Analyze errors and provide insights for improvement  


## 📂 Repository Structure
```
ML-Sentiment-Classification/
├── data/ # x_train.csv, y_train.csv, x_test.csv, y_test
├── Movie_Review_Sentiment.ipynb # main notebook with pipeline
└── README.md
```


## ⚙️ Approach

### 1. Data Preprocessing  
We normalized reviews by lowercasing, removing punctuation, and lemmatizing words. Negations such as *not* and *never* were kept because they strongly affect sentiment polarity. This step reduced noise and ensured consistent text representation.  

### 2. Feature Engineering  
Features were extracted using a **TF–IDF weighted Bag-of-Words** model. TF–IDF was chosen instead of raw counts to emphasize informative words and reduce the influence of very frequent but unhelpful terms. We included both unigrams and bigrams to capture short phrases like *“not good”*, which are important sentiment cues.  

### 3. Model Training  
A **Logistic Regression** classifier was trained, as it performs well on sparse high-dimensional text data and is computationally efficient. Hyperparameters were tuned with cross-validation to balance underfitting and overfitting.  

### 4. Evaluation  
Accuracy, precision, recall, and F1-score were reported to give a balanced view of model performance. Error analysis revealed that mixed-sentiment phrasing and short emphatic reviews were the most challenging cases.  


## 📊 Results

### Validation (20% split)
- Accuracy: **77%**  
- Precision: **0.78**  
- Recall: **0.75**  
- F1-score: **0.77**

### Test set (600 reviews)
- Accuracy: **78.8%**  
- Precision: **82.6%**  
- Recall: **73.0%**  
- F1-score: **77.5%**

### Error Analysis
- **False Positives:** Mixed phrasing with contrast words (“only”, “but”).  
- **False Negatives:** Short emphatic praise with weak sentiment cues (“SUPERB”, “pleasantly surprised”).  
- Errors occurred across IMDb, Amazon, and Yelp, suggesting **general challenges with implicit sentiment**.  


## 🔎 Key Insights
- Preserving **negation words** is crucial (“not good” ≠ “good”).  
- TF–IDF with unigrams+bigrams balances performance and interpretability.  
- The classifier achieved **robust generalization** across domains, with similar CV, validation, and test performance.  


## ⚙️ Requirements
- Python 3.9+  
- Jupyter Notebook  

Install dependencies with:
```bash
pip install pandas numpy scikit-learn nltk
```

## 📜 License
This project is shared for educational purposes as part of a data science portfolio.
