
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from mlxtend.frequent_patterns import apriori, association_rules
from io import BytesIO

st.set_page_config(page_title='BalanceBite Analytics', layout='wide')

# -------------- Load Data --------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

DATA_PATH = 'data/balancebite_survey_synthetic_1000.csv'
data = load_data(DATA_PATH)

# -------------- Sidebar Navigation --------------
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', 
                        ['Data Visualisation', 'Classification', 'Clustering', 
                         'Association Rules', 'Regression'])

# City filter reusable
city_filter = st.sidebar.multiselect('City filter', sorted(data['City'].unique()),
                                     default=list(data['City'].unique()))
df = data[data['City'].isin(city_filter)].copy()

# -------------- Helper Functions --------------
def show_kpis(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    col1.metric('Respondents', len(df))
    col2.metric('Avg Spend (₹)', f"{df['Avg_Spend_Order'].mean():.0f}")
    adoption_rate = (df['Likely_Try_30_Days'] == 'Yes').mean()*100
    col3.metric('Adoption Intent %', f"{adoption_rate:.1f}%")

# --------- Page: Data Visualisation ----------
if page == 'Data Visualisation':
    st.header('Descriptive Insights')
    show_kpis(df)

    # 10 plots
    st.markdown('### 1‑4. Key Relationships')
    plots = [
        ('Age vs Avg Spend', 'Age_Bracket', 'Avg_Spend_Order'),
        ('Income vs Avg Spend', 'Monthly_Income', 'Avg_Spend_Order'),
        ('Workouts vs Orders', 'Weekly_Workouts', 'Orders_Per_Week'),
        ('Price Band vs Avg Spend', 'Fair_Price_Bundle', 'Avg_Spend_Order')
    ]
    for title, x, y in plots:
        fig, ax = plt.subplots()
        if df[x].dtype == object:
            sns.boxplot(x=df[x], y=df[y], ax=ax)
        else:
            sns.scatterplot(x=df[x], y=df[y], ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    st.markdown('### 5‑7. Distributions')
    dist_cols = ['Monthly_Income', 'Avg_Spend_Order', 'Weekly_Workouts']
    for col in dist_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    st.markdown('### 8‑9. Category Counts')
    cat_cols = ['Drink_Preference', 'Dietary_Style']
    for col in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
        ax.set_title(f'Count of {col}')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    st.markdown('### 10. Correlation Heatmap')
    corr_cols = ['Monthly_Income', 'Avg_Spend_Order', 'Weekly_Workouts', 'Orders_Per_Week']
    fig, ax = plt.subplots()
    sns.heatmap(df[corr_cols].corr(), annot=True, ax=ax)
    st.pyplot(fig)

# --------- Page: Classification ----------
elif page == 'Classification':
    st.header('Adoption Prediction Models')
    target = 'Likely_Try_30_Days'
    df_model = df.copy()
    df_model[target] = df_model[target].map({'Yes':1, 'Maybe':0, 'No':0})
    X = pd.get_dummies(df_model.drop(columns=[target]), drop_first=True)
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(max_depth=6),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }

    results = []
    roc_info = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        results.append((name, acc, prec, rec, f1))

        proba = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_info[name] = (fpr, tpr, auc(fpr, tpr))

    st.subheader('Model Metrics')
    res_df = pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1'])
    st.dataframe(res_df.style.format('{:.2f}', subset=['Accuracy','Precision','Recall','F1']))

    sel_model = st.selectbox('Select model for Confusion Matrix', list(models.keys()))
    cm = confusion_matrix(y_test, models[sel_model].predict(X_test))
    st.write(pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1']))

    # ROC Curves
    fig, ax = plt.subplots()
    for name, (fpr, tpr, auc_val) in roc_info.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
    ax.plot([0,1],[0,1],'--', color='grey')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    st.pyplot(fig)

    # Upload new data for prediction
    st.markdown('---')
    st.subheader('Predict on New Respondents')
    up_file = st.file_uploader('Upload CSV (same structure, no target column)', type=['csv'])
    if up_file:
        new_df = pd.read_csv(up_file)
        new_X = pd.get_dummies(new_df, drop_first=True)
        new_X = new_X.reindex(columns=X.columns, fill_value=0)
        preds = models[sel_model].predict(scaler.transform(new_X))
        out_df = new_df.copy()
        out_df['Predicted_Adoption'] = preds
        st.write(out_df.head())
        st.download_button('Download Predictions',
                           out_df.to_csv(index=False).encode('utf-8'),
                           file_name='predictions.csv',
                           mime='text/csv')

# --------- Page: Clustering ----------
elif page == 'Clustering':
    st.header('K‑Means Customer Segmentation')
    k = st.sidebar.slider('Number of clusters (k)', 2, 10, 4)
    features = ['Monthly_Income','Avg_Spend_Order','Weekly_Workouts','Orders_Per_Week']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow curve
    sse = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_scaled)
        sse.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), sse, marker='o')
    ax.set_title('Elbow Curve')
    ax.set_xlabel('k')
    ax.set_ylabel('SSE')
    st.pyplot(fig)

    km_final = KMeans(n_clusters=k, random_state=42)
    labels = km_final.fit_predict(X_scaled)
    df_seg = df.copy()
    df_seg['Cluster'] = labels
    persona = df_seg.groupby('Cluster')[features].mean().round(1)
    st.subheader('Cluster Personas')
    st.dataframe(persona)

    st.download_button('Download Clustered Data',
                       df_seg.to_csv(index=False).encode('utf-8'),
                       file_name='clustered_data.csv',
                       mime='text/csv')

# --------- Page: Association Rules ----------
elif page == 'Association Rules':
    st.header('Apriori Association Rules')
    min_sup = st.sidebar.slider('Minimum support', 0.01, 0.2, 0.05, 0.01)
    min_conf = st.sidebar.slider('Minimum confidence', 0.1, 0.9, 0.3, 0.05)

    # Build basket of items
    def row_to_items(row):
        items = []
        for val in str(row['Favourite_Flavours']).split(','):
            v = val.strip()
            if v: items.append(f'Flavour_{v}')
        for val in str(row['Spirits_Enjoyed']).split(','):
            v = val.strip()
            if v and v!='None':
                items.append(f'Spirit_{v}')
        return items

    baskets = df.apply(row_to_items, axis=1).tolist()
    unique_items = sorted(set(itertools.chain.from_iterable(baskets)))
    onehot = pd.DataFrame(0, index=range(len(baskets)), columns=unique_items)
    for idx, items in enumerate(baskets):
        onehot.loc[idx, items] = 1

    freq = apriori(onehot, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
    if rules.empty:
        st.info('No rules meet the thresholds.')
    else:
        rules = rules.sort_values('confidence', ascending=False).head(10)
        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

# --------- Page: Regression ----------
else:
    st.header('Regression Models – Spend Prediction')
    target = 'Avg_Spend_Order'
    feature_cols = ['Monthly_Income','Orders_Per_Week','Weekly_Workouts']
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }

    rows = []
    for name, reg in models.items():
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        rows.append((name, round(r2,3)))

    st.subheader('R² Scores')
    st.table(pd.DataFrame(rows, columns=['Model','R2']))

    # Scatter visuals
    st.markdown('---')
    for col in feature_cols:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col], y=y, ax=ax)
        ax.set_title(f'{col} vs Avg Spend')
        st.pyplot(fig)
