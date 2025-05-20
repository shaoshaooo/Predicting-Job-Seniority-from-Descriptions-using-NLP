# Predicting Job Seniority from Descriptions using NLP

This project applies Natural Language Processing (NLP) and machine learning to classify job descriptions by their required seniority level — such as *Internship*, *Entry level*, or *Mid-Senior level* — using real-world job postings in the machine learning field.

---

## Project Overview

- **Dataset**: - <a href="https://github.com/shaoshaooo/Predicting-Job-Seniority-from-Descriptions-using-NLP/blob/main/1000_ml_jobs_us.csv">1000 ml jobs usa Dataset</a>
- **Goal**: Predict the seniority level of a job based on its description
- **Tech Stack**:
  - Python
  - pandas, scikit-learn
  - spaCy (for lemmatization and stopword filtering)
  - TF-IDF Vectorization
  - Random Forest Classifier
  - Matplotlib & Seaborn for visualization

---

## Workflow

1. **Text Preprocessing**:
   - Lowercasing, lemmatization, stopword removal using spaCy
2. **Feature Engineering**:
   - TF-IDF Vectorization of cleaned descriptions
3. **Modeling**:
   - Trained a Random Forest Classifier to predict seniority
4. **Evaluation**:
   - Classification report
   - Confusion matrix
   - Label distribution visualization
5. **Custom Input Testing**:
   - Predicts seniority level from any new job description

---

## Visualizations

| Confusion Matrix | Predicted Class Distribution | Prediction Test | Word cloud |
|------------------|------------------------------|-----------------|------------|
| ![confusion matrix](<img width="440" alt="ml job_confusion matrix" src="https://github.com/user-attachments/assets/1784ce47-9c4c-4c9f-8359-59397d73000a" />
) | ![barplot](<img width="413" alt="ml jobs_Barplot" src="https://github.com/user-attachments/assets/a4e38c42-7190-4882-ac0c-1ae281baeed6" />
) | ![test result](<img width="470" alt="ml jobs_prediction test" src="https://github.com/user-attachments/assets/43ea2fd1-1de3-41cf-820c-75137aac3125" />
) | ![word cloud](<img width="201" alt="ml jobs_wordcloud" src="https://github.com/user-attachments/assets/1ed9620a-4a26-45a2-a108-3305c8a04fbd" />
)



## Sample Prediction

```python
text = """
We are looking for a machine learning engineer to lead the development of scalable AI systems. 
Must have 5+ years of experience and proven leadership in team-based projects.
"""

predict_seniority(text)  ➝  "Mid-Senior level"
