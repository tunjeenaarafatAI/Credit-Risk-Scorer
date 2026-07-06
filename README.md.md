&#x20;                                                             **Credit Risk Scoring System**



*A simple tool to predict loan default risk using machine learning — with explanations for every prediction.*



##### **What it does?**

##### 

*Enter an applicant's financial details and get an instant risk score. The app also shows \*why\* that score was given, using SHAP to break down which factors helped or hurt the application.*



#### **Screenshots**



*\*\*Risk Prediction\*\**

*!\[](screenshot1.png)*



*\*\*What drove the prediction?\*\**

*!\[](screenshot2.png)*



*\*\*Key factors at a glance\*\**

*!\[](screenshot3.png)*



#### **How to run locally**?



*```bash*

*pip install streamlit pandas numpy scikit-learn shap matplotlib*

*streamlit run app.py*



#### **Techstack used:**

*Model: Random Forest (scikit-learn)*

*Explainability: SHAP*

*UI: Streamlit*

*Dataset: German Credit Data (UCI)*



#### **Notes:**

*AUC score on test set: \~0.80*

*Built as a learning project to explore explainable AI in finance*

