import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Optional model imports
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ----------------- إعداد الصفحة -----------------
st.set_page_config(page_title="Surat Housing Data App - Complete Lifecycle", layout="wide")
st.title("Surat Housing Dataset – Data Cleaning, EDA & Simple Model Workflow")
st.write("تطبيق لعرض الداتا بعد الـ preprocessing مع أدوات تنظيف تفاعلية، عرض بيانات، رسومات، وإعداد سريع لنموذج (اختياري).")

# ----------------- تحميل الملف أو قراءة محليًا -----------------
uploaded_file = st.file_uploader("Upload a CSV file (أو اتركه ليحمّل الملف المحلي 'surat_uncleaned.csv')", type=["csv"]) 

data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File successfully loaded!")
    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
        st.stop()
else:
    # Try to load local fallback
    try:
        data = pd.read_csv("surat_uncleaned.csv")
        st.info("Loaded local file 'surat_uncleaned.csv'")
    except FileNotFoundError:
        st.warning("لم يتم رفع ملف ولا يوجد ملف محلي 'surat_uncleaned.csv'. الرجاء رفع ملف CSV للمتابعة.")
        st.stop()

# Work on a copy
df = data.copy()

# ----------------- Dataset preview controls -----------------
st.markdown("---")
st.header("Dataset preview")

if st.checkbox("Preview Dataset"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Head"):
            st.write(df.head())
    with col2:
        if st.button("Show Tail"):
            st.write(df.tail())

    number = st.slider("Select number of rows to preview (head)", min_value=1, max_value=min(100, len(df)), value=5)
    st.write(df.head(number))

if st.checkbox("Show all data"):
    st.dataframe(df)

if st.checkbox("Show Column Names"):
    st.write(list(df.columns))

if st.checkbox("Show Dimensions"):
    st.write({"rows": df.shape[0], "columns": df.shape[1]})

if st.checkbox("Show Summary (describe)"):
    st.write(df.describe(include='all').T)

st.markdown("---")

# ----------------- Cleaning helpers (from original script) -----------------
@st.cache_data
def extract_numeric_column(series):
    def extract_numeric(value):
        if isinstance(value, str):
            value = (
                value.replace("₹", "")
                     .replace(",", "")
                     .replace("per sqft", "")
                     .replace("sqft", "")
                     .replace("Lac", "")
                     .strip()
            )
            try:
                return float(value.split()[0])
            except:
                return np.nan
        return value
    return series.apply(extract_numeric)

# ----------------- Start Cleaning (interactive) -----------------
st.subheader("Start Cleaning / تنظيف تفاعلي")

# Convert columns that look numeric to numeric (interactive)
if st.checkbox("Convert columns that should be numeric (محاولة تحويل الأعمدة النصية إلى رقمية)"):
    object_cols = list(df.select_dtypes(include=['object']).columns)
    chosen = st.multiselect("Select columns to attempt to convert to numeric", object_cols)
    if st.button("Convert selected to numeric"):
        for c in chosen:
            # try smart extraction for common patterns
            if df[c].astype(str).str.contains(r'\d').any():
                df[c] = extract_numeric_column(df[c])
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        st.success("Conversion attempted. تحقق من القيم المفقودة أو الأنواع.")
        st.write(df[chosen].dtypes)

# Show unique values of categorical features
if st.checkbox("Show unique values of categorical (object) columns"):
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    if cat_cols:
        for col in cat_cols:
            st.write(f"**{col}**: {df[col].unique()[:50]}")
    else:
        st.info("No object/categorical columns detected.")

st.markdown("---")
# ----------------- Handle Missing Values -----------------
st.subheader("Handle Missing Values / معالجة القيم المفقودة")

if st.checkbox("Show Missing Values (count per column)"):
    st.write(df.isna().sum())

col_option = None
if st.checkbox("Select a column to treat missing values"):
    col_option = st.selectbox("Select Column to treat missing values", options=list(df.columns))
    strategy = st.selectbox("Select Missing values Strategy", ("Replace with Mean", "Replace with Median", "Replace with Mode", "Drop rows with NaN in this column", "Fill with custom value"))
    if st.button("Apply missing value strategy"):
        if strategy == "Drop rows with NaN in this column":
            before = len(df)
            df = df.dropna(subset=[col_option])
            st.success(f"Dropped rows with NaN in `{col_option}`. Rows before: {before}, after: {len(df)}")
        else:
            replaced_value = None
            if strategy == "Replace with Mean":
                if pd.api.types.is_numeric_dtype(df[col_option]):
                    replaced_value = df[col_option].mean()
                else:
                    st.error("Mean replacement requires a numeric column.")
            elif strategy == "Replace with Median":
                if pd.api.types.is_numeric_dtype(df[col_option]):
                    replaced_value = df[col_option].median()
                else:
                    st.error("Median replacement requires a numeric column.")
            elif strategy == "Replace with Mode":
                mode_series = df[col_option].mode()
                replaced_value = mode_series.iloc[0] if not mode_series.empty else np.nan
            else:  # custom value
                custom_val = st.text_input("Enter custom value to fill NaNs (string input)")
                if custom_val != "":
                    # try to cast to numeric if column numeric
                    if pd.api.types.is_numeric_dtype(df[col_option]):
                        try:
                            replaced_value = float(custom_val)
                        except:
                            st.error("Could not cast custom value to numeric for this column.")
                            replaced_value = None
                    else:
                        replaced_value = custom_val

            if replaced_value is not None:
                df[col_option] = df[col_option].fillna(replaced_value)
                st.success(f"Null values in `{col_option}` replaced with `{replaced_value}`")
                st.write(df[col_option].isna().sum())

st.markdown("---")

# Drop chosen columns
if st.checkbox("Drop columns"):
    columns_to_drop = st.multiselect("Select columns to drop", options=list(df.columns))
    if st.button("Drop selected columns"):
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            st.success(f"Dropped columns: {columns_to_drop}")
            st.dataframe(df.head())
        else:
            st.warning("No columns selected to drop.")

st.markdown("---")

# Drop rows with NaN across selected columns
if st.checkbox("Drop rows with NaN in selected columns"):
    columns_to_check = st.multiselect("Select columns to check for NaN values", options=list(df.columns))
    if st.button("Drop rows with NaN in these columns"):
        if columns_to_check:
            before = len(df)
            df = df.dropna(subset=columns_to_check)
            st.success(f"Dropped rows. Before: {before}, After: {len(df)}")
            st.dataframe(df.head())
        else:
            st.warning("Please select at least one column.")

st.markdown("---")
# ----------------- Handle Categorical Values -----------------
st.subheader("Handle Categorical Values / التعامل مع المتغيرات التصنيفية")

categorical_cols = list(df.select_dtypes(include=['object']).columns)
st.write(f"Categorical columns detected: {categorical_cols}")

# One-hot encoding for nominal columns
one_hot_enc = st.multiselect("Select nominal categorical columns to one-hot encode", options=categorical_cols)
if st.button("Apply one-hot encoding"):
    if one_hot_enc:
        for column in one_hot_enc:
            if df[column].dtype == 'object':
                df = pd.get_dummies(df, columns=[column], prefix=[column], drop_first=False)
        st.success("One-hot encoding applied.")
        st.write(df.head())
    else:
        st.warning("No columns selected for one-hot encoding.")

# Label encoding for ordinal columns (or single target)
label_enc_cols = st.multiselect("Select ordinal categorical columns to label-encode", options=categorical_cols)
if st.button("Apply label encoding"):
    if label_enc_cols:
        for column in label_enc_cols:
            if df[column].dtype == 'object':
                try:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column].astype(str))
                except Exception as e:
                    st.error(f"Could not encode column {column}: {e}")
        st.success("Label encoding applied.")
        st.write(df.head())
    else:
        st.warning("No columns selected for label encoding.")

st.markdown("---")

# ----------------- Basic automated cleaning from original script -----------------
if st.checkbox("Run quick automatic cleaning (drop duplicates, fill Unknown, extract numeric for common columns)"):
    before = df.shape[0]
    df = df.drop_duplicates()
    df = df.fillna("Unknown")
    # apply numeric extraction for common columns if present
    for c in ["square_feet", "price_per_sqft", "price"]:
        if c in df.columns:
            df[c] = extract_numeric_column(df[c])
    st.success(f"Auto-clean completed. Rows before: {before}, after dedup: {df.shape[0]}")
    st.write(df.head())

st.markdown("---")

# ----------------- Editable table (like original) -----------------
st.subheader("تعديل البيانات (Editable Table)")
st.write("تقدر تعدّل القيم في الجدول مباشرة أو تضيف صف جديد من خلال الجدول.")

edited_df = st.data_editor(df, num_rows="dynamic", key="editable_df")

if st.button("تأكيد التعديلات (لا يتم الحفظ في CSV الأصلي)"):
    st.success("تم تحديث البيانات داخل التطبيق (مش محفوظة على الملف الأصلي).")
    df = edited_df.copy()

st.markdown("---")

# ----------------- ملخصات إحصائية -----------------
st.subheader("ملخصات إحصائية للأعمدة الرقمية")
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(numeric_cols) > 0:
    st.write(df[numeric_cols].describe().T)
else:
    st.info("لا يوجد أعمدة رقمية لعرض ملخص إحصائي.")

# ----------------- Visualizations (مُعدّل: فقط الهارب المذكورة) -----------------
st.subheader("Visualizations / الرسومات")

# ensure numeric_cols up-to-date
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 1) Histogram
st.markdown("### 1) Histogram لعمود رقمي")
if len(numeric_cols) > 0:
    col_hist = st.selectbox("اختار عمود لرسم الـ Histogram:", numeric_cols, index=0, key="hist_col")
    bins = st.slider("عدد bins", min_value=5, max_value=100, value=30, key="hist_bins")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col_hist].dropna(), bins=bins)
    ax.set_title(f"Histogram of {col_hist}")
    ax.set_xlabel(col_hist)
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.info("لا توجد أعمدة رقمية لعمل Histogram.")

# 2) Scatter plot between price and square_feet
st.markdown("### 2) Scatter Plot بين السعر والمساحة")
if "price" in df.columns and "square_feet" in df.columns:
    fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="square_feet", y="price", ax=ax_sc)
    ax_sc.set_title("Price vs Square Feet")
    ax_sc.set_xlabel("Square Feet")
    ax_sc.set_ylabel("Price")
    st.pyplot(fig_sc)
else:
    st.info("لا يوجد عمود price أو square_feet لعمل Scatter Plot.")

# 3) Boxplot لعمود السعر
st.markdown("### 3) Boxplot لعمود السعر")
if "price" in df.columns:
    fig_box, ax_box = plt.subplots(figsize=(4, 6))
    sns.boxplot(y=df["price"].dropna(), ax=ax_box)
    ax_box.set_title("Price Boxplot")
    ax_box.set_ylabel("Price")
    st.pyplot(fig_box)
else:
    st.info("لا يوجد عمود price لعمل Boxplot.")

# 4) Correlation Heatmap
st.markdown("### 4) Correlation Heatmap")
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="vlag", ax=ax2, fmt=".2f")
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

    # Simple Data Analysis expander تحت الـ Correlation Heatmap
    with st.expander("Simple Data Analysis (click to view)"):
        st.write("Below are quick insights extracted from the correlation heatmap and the dataset:")
        st.markdown("""
1. Correlations close to zero suggest features are generally not strongly related.  
2. If `price` and `square_feet` exist but show weak correlation, size alone may not explain price.  
3. `price_per_sqft` (if present) often behaves independently from other features.  
4. Encoding or cleaning of categorical columns may change relationships — try different strategies above.  
5. Consider enriching the dataset with external features (location demand, year, amenities) for better modeling.
        """)
else:
    st.info("لا يوجد عدد كافي من الأعمدة الرقمية لعرض Correlation Heatmap.")
    with st.expander("Simple Data Analysis (click to view)"):
        st.write("Quick notes when numeric features are insufficient:")
        st.markdown("""
- تأكد من تحويل الأعمدة النصية التي تحتوي أرقام (مثل '1200 sqft' أو '₹ 1,23,456') إلى صيغة رقمية.  
- حاول استخدام خيار الـ `Run quick automatic cleaning` أعلى التطبيق لاستخراج الأرقام الشائعة.  
- بعد التحويل، عد إلى هذه الشاشة لتحديث الرسومات.
        """)

st.markdown("---")

# ----------------- Simple Data Analysis expander (insights) -----------------
# NOTE: already included directly after the heatmap above as requested, leaving this section minimal to avoid duplication.

# ----------------- Optional: Quick Modeling playground -----------------
st.subheader("Quick Modeling Playground (اختياري)")
if st.checkbox("Enable simple classification workflow"):
    target = st.selectbox("Select target column (label) for classification", options=list(df.columns))
    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Basic preprocessing: drop non-numeric or fill na
        X_numeric = X.select_dtypes(include=["number"]).copy()
        if X_numeric.shape[1] == 0:
            st.error("No numeric features available for modeling. Try encoding categorical columns or choose a different target.")
        else:
            X_numeric = X_numeric.fillna(X_numeric.median())
            # split
            test_size = st.slider("Test size (%)", 5, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=test_size/100, random_state=42)

            model_choice = st.selectbox("Choose model", ("RandomForest", "SVM", "XGBoost (if available)"))
            if st.button("Train and Evaluate"):
                if model_choice == "RandomForest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "SVM":
                    model = SVC()
                else:
                    if _HAS_XGBOOST:
                        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    else:
                        st.error("XGBoost is not available in the environment. Choose another model.")
                        model = None

                if model is not None:
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        st.write("Accuracy:", accuracy_score(y_test, preds))
                        st.write("Classification Report:")
                        st.text(classification_report(y_test, preds))
                    except Exception as e:
                        st.error(f"Training failed: {e}")

st.markdown("---")

st.success("Ready — يمكنك استكشاف، تنظيف، وتحليل البيانات ثم تجربة نموذج بسيط داخل التطبيق.")
