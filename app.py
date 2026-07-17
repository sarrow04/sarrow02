# -*- coding: utf-8 -*-
"""
データ探偵事務所 分析アプリ (Streamlit版)
実行方法: streamlit run app.py
必要なライブラリ: pip install streamlit pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
任意(あれば使います): pip install shap lightgbm
"""
import io

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="データ探偵事務所 分析アプリ", layout="wide")


def _rerun():
    """Streamlitのバージョン差異を吸収する再実行ヘルパー"""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ============================================================
# コアロジック（Streamlit非依存・純粋関数）
# ============================================================

def read_csv_safely(file_obj):
    """文字コードを自動判定してCSVを読み込む"""
    last_err = None
    for enc in ["utf-8-sig", "cp932", "utf-8"]:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc), enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise ValueError("文字コードを判定できませんでした") from last_err


def load_file(uploaded_file, filename, encoding_choice="auto"):
    lower = filename.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(uploaded_file), "excel"
    if encoding_choice == "auto":
        return read_csv_safely(uploaded_file)
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding=encoding_choice), encoding_choice


def identify_id_like_columns(df):
    num_cols = df.select_dtypes(include="number").columns
    return [c for c in num_cols if df[c].nunique(dropna=True) == len(df)]


def missing_summary(df):
    m = pd.DataFrame({
        "欠損数": df.isna().sum(),
        "欠損率(%)": (df.isna().mean() * 100).round(1),
    })
    return m[m["欠損数"] > 0]


def normalize_strings(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.normalize("NFKC").str.strip()
    return df


def clean_numeric_column(series):
    """円・カンマ・単位などの文字を除去して数値化"""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def apply_column_types(df, column_types):
    df = df.copy()
    for col, t in column_types.items():
        if col not in df.columns:
            continue
        if t == "str":
            df[col] = df[col].astype("string")
        elif t == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif t == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        elif t == "category":
            df[col] = df[col].astype("category")
    return df


def fill_missing(df, method):
    df = df.copy()
    if method == "dropna":
        return df.dropna()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median() if method == "median" else df[col].mean()
            else:
                mode = df[col].mode()
                fill_val = mode[0] if len(mode) > 0 else ""
            df[col] = df[col].fillna(fill_val)
    return df


def clip_outliers(df, cols=None):
    df = df.copy()
    target_cols = cols if cols else list(df.select_dtypes(include="number").columns)
    for col in target_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype("float64")
        lower = s.quantile(0.01)
        upper = s.quantile(0.99)
        df[col] = s.clip(lower=lower, upper=upper)
    return df


def extract_date_features(df, col):
    df = df.copy()
    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = dt.dt.year
    df[f"{col}_month"] = dt.dt.month
    df[f"{col}_day"] = dt.dt.day
    df[f"{col}_dow"] = dt.dt.dayofweek
    df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")
    return df


def encode_flag(df, col):
    df = df.copy()
    u_vals = df[col].dropna().unique()
    if len(u_vals) != 2:
        return df, None
    df[col] = df[col].map({u_vals[0]: 0, u_vals[1]: 1})
    return df, f"{col}: '{u_vals[0]}' → 0, '{u_vals[1]}' → 1"


def one_hot_encode(df, cols):
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return df
    return pd.get_dummies(df, columns=valid, drop_first=True)


def correlation_matrix(df):
    return df.select_dtypes(include="number").corr(numeric_only=True)


def run_ttest(df, group_col, value_col):
    from scipy import stats
    levels = list(df[group_col].dropna().unique())
    if len(levels) < 2:
        return {"error": f"グループが2つ未満のため検定できません: {group_col}"}
    note = None
    if len(levels) > 2:
        note = f"グループが3つ以上のため先頭の2グループで検定しました -> {levels[0]} / {levels[1]}"
    g1 = pd.to_numeric(df.loc[df[group_col] == levels[0], value_col], errors="coerce").dropna()
    g2 = pd.to_numeric(df.loc[df[group_col] == levels[1], value_col], errors="coerce").dropna()
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    return {"note": note, "levels": levels[:2], "mean1": g1.mean(), "n1": len(g1),
            "mean2": g2.mean(), "n2": len(g2), "t": t_stat, "p": p_val}


def run_mannwhitney(df, group_col, value_col):
    from scipy import stats
    levels = list(df[group_col].dropna().unique())
    if len(levels) < 2:
        return {"error": f"グループが2つ未満のため検定できません: {group_col}"}
    note = None
    if len(levels) > 2:
        note = f"グループが3つ以上のため先頭の2グループで検定しました -> {levels[0]} / {levels[1]}"
    g1 = pd.to_numeric(df.loc[df[group_col] == levels[0], value_col], errors="coerce").dropna()
    g2 = pd.to_numeric(df.loc[df[group_col] == levels[1], value_col], errors="coerce").dropna()
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {"note": note, "levels": levels[:2], "median1": g1.median(), "n1": len(g1),
            "median2": g2.median(), "n2": len(g2), "u": u_stat, "p": p_val}


def run_chi2(df, col_a, col_b):
    from scipy import stats
    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    return {"crosstab": ct, "chi2": chi2, "dof": dof, "p": p_val}


def build_model(task, algo, random_state=42):
    if algo == "lgbm":
        import lightgbm as lgb
        return (lgb.LGBMRegressor(random_state=random_state, verbose=-1) if task == "reg"
                else lgb.LGBMClassifier(random_state=random_state, verbose=-1))
    if algo == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        return (HistGradientBoostingRegressor(random_state=random_state) if task == "reg"
                else HistGradientBoostingClassifier(random_state=random_state))
    if algo == "rf":
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        return (RandomForestRegressor(n_estimators=300, random_state=random_state) if task == "reg"
                else RandomForestClassifier(n_estimators=300, random_state=random_state))
    if algo == "dt":
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        return (DecisionTreeRegressor(max_depth=4, random_state=random_state) if task == "reg"
                else DecisionTreeClassifier(max_depth=4, random_state=random_state))
    if task == "reg":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=random_state)


def train_and_evaluate(df, target, features, task, algo, use_dummies=False, scale=False,
                        test_size=0.3, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                                  accuracy_score, classification_report)

    work = df.copy()
    feat_df = (work[features].copy() if features
               else work.select_dtypes(include="number").drop(columns=[target], errors="ignore"))

    if use_dummies:
        cat_cols = [c for c in work.select_dtypes(include=["object", "string", "category"]).columns if c != target]
        if cat_cols:
            dummies = pd.get_dummies(work[cat_cols], drop_first=True).astype(int)
            feat_df = pd.concat([feat_df.select_dtypes(include="number"), dummies], axis=1)

    data = pd.concat([feat_df, work[target]], axis=1).dropna()
    X = data[list(feat_df.columns)]
    y = data[target]
    if len(X) < 10:
        return {"error": "有効なデータが少なすぎます（欠損値を除くと10件未満）"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    model = build_model(task, algo, random_state=random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = {"model": model, "X_train": X_train, "X_test": X_test, "y_test": y_test, "pred": pred}
    if task == "reg":
        result["metrics"] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
        }
    else:
        result["metrics"] = {"Accuracy": accuracy_score(y_test, pred)}
        result["report"] = classification_report(y_test, pred, zero_division=0)
    return result


def compute_shap_values(model, X_train, X_test, algo, task):
    import shap
    if algo == "lr":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if task == "clf":
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    return shap_values


# ============================================================
# セッション状態の初期化
# ============================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "trained" not in st.session_state:
    st.session_state.trained = None  # train_and_evaluateの結果を保持(SHAPで使うため)
if "trained_algo" not in st.session_state:
    st.session_state.trained_algo = None
if "trained_task" not in st.session_state:
    st.session_state.trained_task = None

st.title("🔍 データ探偵事務所 分析アプリ")
st.caption("CSV/Excelをアップロードすると、その場でPythonが動いて分析します")

# ============================================================
# サイドバー: ファイルアップロード
# ============================================================
with st.sidebar:
    st.header("📂 データ読込")
    uploaded = st.file_uploader("CSV / Excel ファイル", type=["csv", "xlsx", "xls"])
    encoding_choice = st.selectbox("エンコーディング (CSVのみ)", ["auto", "utf-8", "cp932"], index=0)

    if uploaded is not None and st.session_state.uploaded_name != uploaded.name:
        try:
            df_loaded, used_enc = load_file(uploaded, uploaded.name, encoding_choice)
            st.session_state.df_raw = df_loaded
            st.session_state.df = df_loaded.copy()
            st.session_state.uploaded_name = uploaded.name
            st.session_state.trained = None
            st.success(f"読み込み成功: {uploaded.name} ({used_enc})")
        except Exception as e:
            st.error(f"読み込みに失敗しました: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        if st.button("🔄 生データにリセット"):
            st.session_state.df = st.session_state.df_raw.copy()
            st.session_state.trained = None
            st.success("読み込み直後の状態にリセットしました")
            _rerun()

if st.session_state.df is None:
    st.info("👈 サイドバーからCSVまたはExcelファイルをアップロードしてください")
    st.stop()

df = st.session_state.df

# ============================================================
# 1. データプレビュー（常に現在の状態を表示）
# ============================================================
st.header("1. データプレビュー")
c1, c2 = st.columns(2)
c1.metric("行数", len(df))
c2.metric("列数", len(df.columns))

tab_head, tab_tail, tab_info, tab_missing = st.tabs(["先頭5行", "末尾5行", "データ型", "欠損値"])
with tab_head:
    st.dataframe(df.head(), use_container_width=True)
with tab_tail:
    st.dataframe(df.tail(), use_container_width=True)
with tab_info:
    st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
with tab_missing:
    ms = missing_summary(df)
    if len(ms) > 0:
        st.dataframe(ms, use_container_width=True)
    else:
        st.success("欠損値はありません")

id_like = identify_id_like_columns(df)
if id_like:
    st.caption(f"※ 値が全てユニークな数値カラム（ID等の可能性）: {', '.join(id_like)}")

# ============================================================
# 2. カラムの型変換
# ============================================================
with st.expander("🔤 2. カラムの型変換", expanded=False):
    st.caption("型を変えたいカラムだけ選んでください。「そのまま」以外を選んだカラムのみ変換されます。")
    type_options = ["そのまま", "文字列", "整数", "小数", "日付", "カテゴリ"]
    type_map = {"そのまま": None, "文字列": "str", "整数": "int", "小数": "float",
                "日付": "date", "カテゴリ": "category"}
    with st.form("type_form"):
        col_list = list(df.columns)
        selections = {}
        cols_per_row = 2
        for i in range(0, len(col_list), cols_per_row):
            row = st.columns(cols_per_row)
            for j, c in enumerate(col_list[i:i + cols_per_row]):
                with row[j]:
                    selections[c] = st.selectbox(f"{c}  (現在: {df[c].dtype})", type_options,
                                                  key=f"type_{c}")
        submitted = st.form_submit_button("型変換を適用")
        if submitted:
            column_types = {c: type_map[v] for c, v in selections.items() if type_map[v] is not None}
            if column_types:
                st.session_state.df = apply_column_types(df, column_types)
                st.success(f"{len(column_types)}カラムの型を変換しました")
                _rerun()
            else:
                st.info("変更対象がありませんでした")

# ============================================================
# 3. 数値クリーニング（円・カンマなど除去）
# ============================================================
with st.expander("💴 3. 数値クリーニング（円・カンマなどの文字を除去）", expanded=False):
    numclean_cols = st.multiselect("対象カラム", options=list(df.columns), key="numclean_cols")
    if st.button("数値クリーニングを適用", key="btn_numclean"):
        if numclean_cols:
            new_df = df.copy()
            for c in numclean_cols:
                new_df[c] = clean_numeric_column(new_df[c])
            st.session_state.df = new_df
            st.success(f"{len(numclean_cols)}カラムを数値化しました（変換できなかった値は欠損になります）")
            _rerun()
        else:
            st.info("対象カラムを選んでください")

# ============================================================
# 4. 不要カラムの削除
# ============================================================
with st.expander("🗑️ 4. 不要カラムの削除", expanded=False):
    drop_cols = st.multiselect("削除するカラム", options=list(df.columns), key="drop_cols")
    if st.button("削除を適用", key="btn_drop"):
        if drop_cols:
            st.session_state.df = df.drop(columns=drop_cols, errors="ignore")
            st.success(f"{len(drop_cols)}カラムを削除しました")
            _rerun()
        else:
            st.info("削除するカラムを選んでください")

# ============================================================
# 5. 欠損値・外れ値
# ============================================================
with st.expander("🧹 5. 欠損値・外れ値の処理", expanded=False):
    st.subheader("欠損値")
    fillna_method = st.radio("方法", ["数値は中央値・カテゴリは最頻値", "数値は平均値・カテゴリは最頻値", "欠損を含む行を削除"],
                              key="fillna_method")
    method_map = {"数値は中央値・カテゴリは最頻値": "median", "数値は平均値・カテゴリは最頻値": "mean",
                  "欠損を含む行を削除": "dropna"}
    if st.button("欠損値処理を適用", key="btn_fillna"):
        st.session_state.df = fill_missing(df, method_map[fillna_method])
        st.success("欠損値処理を適用しました")
        _rerun()

    st.subheader("外れ値クリッピング (1%-99%)")
    outlier_cols = st.multiselect("対象カラム（空欄で数値カラム全体）",
                                   options=list(df.select_dtypes(include="number").columns),
                                   key="outlier_cols")
    if st.button("外れ値クリッピングを適用", key="btn_outlier"):
        st.session_state.df = clip_outliers(df, cols=outlier_cols if outlier_cols else None)
        st.success("外れ値をクリッピングしました")
        _rerun()

# ============================================================
# 6. 特徴量エンジニアリング
# ============================================================
with st.expander("🛠️ 6. 特徴量エンジニアリング", expanded=False):
    st.subheader("日付分解 (年・月・日・曜日・週末フラグ)")
    date_col = st.selectbox("対象カラム", options=["(選択しない)"] + list(df.columns), key="date_col")
    if st.button("日付分解を適用", key="btn_datefeat"):
        if date_col != "(選択しない)":
            st.session_state.df = extract_date_features(df, date_col)
            st.success(f"{date_col} から5つのカラムを作成しました")
            _rerun()
        else:
            st.info("対象カラムを選んでください")

    st.subheader("フラグの数値化 (2値カラムを0/1に)")
    flag_cols = st.multiselect("対象カラム", options=list(df.columns), key="flag_cols")
    if st.button("フラグ数値化を適用", key="btn_flag"):
        if flag_cols:
            new_df = df.copy()
            messages = []
            for c in flag_cols:
                new_df, msg = encode_flag(new_df, c)
                if msg:
                    messages.append(msg)
                else:
                    messages.append(f"{c}: 値の種類が2種類ではないためスキップ")
            st.session_state.df = new_df
            for m in messages:
                st.write("- " + m)
            _rerun()
        else:
            st.info("対象カラムを選んでください")

    st.subheader("One-Hot Encoding")
    cat_cols_options = list(df.select_dtypes(include=["object", "string", "category"]).columns)
    onehot_cols = st.multiselect("対象カラム", options=cat_cols_options, key="onehot_cols")
    if st.button("One-Hot Encodingを適用", key="btn_onehot"):
        if onehot_cols:
            st.session_state.df = one_hot_encode(df, onehot_cols)
            st.success(f"{len(onehot_cols)}カラムをOne-Hot化しました")
            _rerun()
        else:
            st.info("対象カラムを選んでください")

    st.subheader("カスタム特徴量 (Pandasコード)")
    st.caption("⚠️ ここに入力したコードはそのまま実行されます。信頼できる自分のコードのみ入力してください。")
    feat_code = st.text_area("例: df['単価'] = df['売上'] / df['数量']", key="feat_code", height=80)
    if st.button("カスタム特徴量を適用", key="btn_featcode"):
        if feat_code.strip():
            try:
                local_vars = {"df": df.copy(), "pd": pd, "np": np}
                exec(feat_code, {}, local_vars)
                st.session_state.df = local_vars["df"]
                st.success("特徴量を作成しました")
                _rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
        else:
            st.info("コードを入力してください")

# ============================================================
# 7. 可視化
# ============================================================
with st.expander("📊 7. 可視化", expanded=False):
    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols_for_viz = list(df.select_dtypes(include=["object", "string", "category"]).columns)
    viz_tab_hist, viz_tab_box, viz_tab_scatter, viz_tab_heatmap = st.tabs(
        ["ヒストグラム", "箱ひげ図", "散布図", "相関ヒートマップ"])

    with viz_tab_hist:
        if numeric_cols:
            hist_col = st.selectbox("対象カラム", options=numeric_cols, key="hist_col")
            import matplotlib.pyplot as plt
            try:
                import japanize_matplotlib  # noqa: F401
            except ImportError:
                pass
            fig, ax = plt.subplots(figsize=(7, 4))
            df[hist_col].dropna().astype(float).plot.hist(bins=30, ax=ax, edgecolor="black", color="#16213e")
            ax.set_title(f"分布: {hist_col}")
            ax.set_xlabel(hist_col)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("数値カラムがありません")

    with viz_tab_box:
        if numeric_cols:
            box_val = st.selectbox("数値カラム", options=numeric_cols, key="box_val")
            box_group = st.selectbox("グループ分け（任意）", options=["(グループなし)"] + cat_cols_for_viz, key="box_group")
            import matplotlib.pyplot as plt
            import seaborn as sns
            try:
                import japanize_matplotlib  # noqa: F401
            except ImportError:
                pass
            fig, ax = plt.subplots(figsize=(7, 4))
            if box_group != "(グループなし)":
                sns.boxplot(data=df, x=box_group, y=box_val, ax=ax, color="skyblue")
                ax.set_title(f"箱ひげ図: {box_val} ({box_group}別)")
            else:
                sns.boxplot(y=df[box_val].dropna().astype(float), ax=ax, color="skyblue")
                ax.set_title(f"箱ひげ図: {box_val}")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("数値カラムがありません")

    with viz_tab_scatter:
        if len(numeric_cols) >= 2:
            scatter_x = st.selectbox("X軸", options=numeric_cols, key="scatter_x")
            scatter_y = st.selectbox("Y軸", options=numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
            import matplotlib.pyplot as plt
            try:
                import japanize_matplotlib  # noqa: F401
            except ImportError:
                pass
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(df[scatter_x], df[scatter_y], alpha=0.6, color="#16213e")
            ax.set_xlabel(scatter_x)
            ax.set_ylabel(scatter_y)
            ax.set_title(f"散布図: {scatter_x} x {scatter_y}")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("数値カラムが2つ未満のため散布図は表示できません")

    with viz_tab_heatmap:
        corr = correlation_matrix(df)
        if len(corr.columns) > 1:
            st.dataframe(corr.round(3), use_container_width=True)
            import matplotlib.pyplot as plt
            import seaborn as sns
            try:
                import japanize_matplotlib  # noqa: F401
            except ImportError:
                pass
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
            ax.set_title("相関ヒートマップ")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("数値カラムが2つ未満のため、相関・ヒートマップは計算できません")

# ============================================================
# 8. 統計検定
# ============================================================
with st.expander("🧪 8. 統計検定", expanded=False):
    test_type = st.selectbox("検定手法", ["実行しない", "t検定 (2群の平均の差)", "Mann-Whitney U検定",
                                        "カイ二乗検定 (独立性)"], key="test_type")
    col_a = st.selectbox("比較する群のカラム", options=list(df.columns), key="test_col_a")
    col_b = st.selectbox("対象カラム (数値 or カテゴリ)", options=list(df.columns), key="test_col_b")
    if st.button("検定を実行", key="btn_test"):
        if test_type == "t検定 (2群の平均の差)":
            res = run_ttest(df, col_a, col_b)
            if "error" in res:
                st.error(res["error"])
            else:
                if res["note"]:
                    st.warning(res["note"])
                st.write(f"{res['levels'][0]} 平均: {res['mean1']:.2f} (n={res['n1']})")
                st.write(f"{res['levels'][1]} 平均: {res['mean2']:.2f} (n={res['n2']})")
                st.write(f"t = {res['t']:.3f}, p = {res['p']:.4f}")
                if res["p"] < 0.05:
                    st.success("2群の平均には統計的に有意な差があります (p < 0.05)")
                else:
                    st.info("有意な差は確認できませんでした (p >= 0.05)")
        elif test_type == "Mann-Whitney U検定":
            res = run_mannwhitney(df, col_a, col_b)
            if "error" in res:
                st.error(res["error"])
            else:
                if res["note"]:
                    st.warning(res["note"])
                st.write(f"{res['levels'][0]} 中央値: {res['median1']:.2f} (n={res['n1']})")
                st.write(f"{res['levels'][1]} 中央値: {res['median2']:.2f} (n={res['n2']})")
                st.write(f"U = {res['u']:.3f}, p = {res['p']:.4f}")
                if res["p"] < 0.05:
                    st.success("2群の分布には統計的に有意な差があります (p < 0.05)")
                else:
                    st.info("有意な差は確認できませんでした (p >= 0.05)")
        elif test_type == "カイ二乗検定 (独立性)":
            res = run_chi2(df, col_a, col_b)
            st.dataframe(res["crosstab"], use_container_width=True)
            st.write(f"chi2 = {res['chi2']:.3f}, 自由度 = {res['dof']}, p = {res['p']:.4f}")
            if res["p"] < 0.05:
                st.success("2つのカラムには統計的に有意な関連があります (p < 0.05)")
            else:
                st.info("有意な関連は確認できませんでした (p >= 0.05)")
        else:
            st.info("検定手法を選んでください")

# ============================================================
# 9. 機械学習 + SHAP
# ============================================================
with st.expander("🤖 9. 機械学習 + SHAP要因分析", expanded=False):
    target = st.selectbox("目的変数", options=list(df.columns), key="ml_target")
    feature_candidates = [c for c in df.columns if c != target]
    features = st.multiselect("特徴量 (空欄なら目的変数以外の数値カラム全て)", options=feature_candidates, key="ml_features")
    task = st.radio("タスク", ["回帰 (数値予測)", "分類 (カテゴリ予測)"], key="ml_task")
    task_code = "reg" if task.startswith("回帰") else "clf"
    algo_label = st.selectbox("アルゴリズム", ["線形/ロジスティック回帰", "決定木", "ランダムフォレスト",
                                          "勾配ブースティング (sklearn)", "LightGBM (要インストール)"], key="ml_algo")
    algo_map = {"線形/ロジスティック回帰": "lr", "決定木": "dt", "ランダムフォレスト": "rf",
                "勾配ブースティング (sklearn)": "hgb", "LightGBM (要インストール)": "lgbm"}
    algo_code = algo_map[algo_label]
    use_dummies = st.checkbox("カテゴリ変数をダミー変数化して特徴量に含める", key="ml_dummies")
    scale = st.checkbox("特徴量を標準化する (線形回帰系で推奨)", key="ml_scale")

    if st.button("モデルを学習", key="btn_train"):
        try:
            result = train_and_evaluate(df, target, features, task_code, algo_code,
                                         use_dummies=use_dummies, scale=scale)
            if "error" in result:
                st.error(result["error"])
                st.session_state.trained = None
            else:
                st.session_state.trained = result
                st.session_state.trained_algo = algo_code
                st.session_state.trained_task = task_code
                st.subheader("モデル精度")
                st.json(result["metrics"])
                if task_code == "clf":
                    st.text(result["report"])
        except ImportError as e:
            st.error(f"必要なライブラリが見つかりません: {e}")
            st.session_state.trained = None
        except Exception as e:
            st.error(f"学習中にエラーが発生しました: {e}")
            st.session_state.trained = None

    if st.session_state.trained is not None:
        st.markdown("---")
        if st.button("SHAPで要因分析", key="btn_shap"):
            try:
                result = st.session_state.trained
                shap_values = compute_shap_values(result["model"], result["X_train"], result["X_test"],
                                                   st.session_state.trained_algo, st.session_state.trained_task)
                import matplotlib.pyplot as plt
                import shap
                try:
                    import japanize_matplotlib  # noqa: F401
                except ImportError:
                    pass
                fig = plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, result["X_test"], show=False)
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.warning("shapが未導入のためSHAP分析はスキップされました（pip install shap で導入できます）")
            except Exception as e:
                st.error(f"SHAP計算中にエラーが発生しました: {e}")

# ============================================================
# 10. ダウンロード
# ============================================================
st.header("10. ダウンロード")
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("📥 現在のデータをCSVでダウンロード", data=csv_bytes,
                    file_name="cleaned_data.csv", mime="text/csv")
