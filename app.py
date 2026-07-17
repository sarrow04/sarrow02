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


def setup_japanese_font():
    """グラフの日本語文字化けをできる限り防ぐ（一度だけ呼び出す）"""
    try:
        import japanize_matplotlib  # noqa: F401  最も手軽で確実な方法
        return "japanize_matplotlib"
    except ImportError:
        pass

    import matplotlib
    import matplotlib.font_manager as fm

    candidates = [
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "YuGothic",
        "Meiryo", "TakaoPGothic", "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


_JP_FONT = setup_japanese_font()


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


def fill_missing(df, method, cols=None, fill_value=None):
    """method: 'median'|'mean'|'constant'|'ffill'|'bfill'|'dropna'
    cols: 対象カラムを限定する場合に指定（Noneなら全カラム対象）"""
    df = df.copy()
    target_cols = [c for c in (cols if cols else df.columns) if c in df.columns]

    if method == "dropna":
        return df.dropna(subset=target_cols) if cols else df.dropna()
    if method == "ffill":
        df[target_cols] = df[target_cols].ffill()
        return df
    if method == "bfill":
        df[target_cols] = df[target_cols].bfill()
        return df
    if method == "constant":
        for col in target_cols:
            df[col] = df[col].fillna(fill_value)
        return df

    for col in target_cols:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median() if method == "median" else df[col].mean()
            else:
                mode = df[col].mode()
                fill_val = mode[0] if len(mode) > 0 else ""
            df[col] = df[col].fillna(fill_val)
    return df


def clip_outliers(df, cols=None, lower_pct=0.01, upper_pct=0.99):
    """lower_pct/upper_pct: クリップする分位点（例: 0.01/0.99 で下位1%・上位1%をクリップ）"""
    df = df.copy()
    target_cols = cols if cols else list(df.select_dtypes(include="number").columns)
    for col in target_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype("float64")
        lower = s.quantile(lower_pct)
        upper = s.quantile(upper_pct)
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


def mask_pii_hash(series, salt):
    import hashlib

    def _h(v):
        if pd.isna(v):
            return v
        return hashlib.sha256((salt + str(v)).encode("utf-8")).hexdigest()[:12]
    return series.map(_h)


def mask_pii_sequential(series):
    uniq = list(pd.Series(series.dropna().unique()))
    mapping = {v: f"ID_{i + 1:04d}" for i, v in enumerate(uniq)}
    return series.map(mapping)


def filter_by_date_range(df, col, start, end):
    """dfをcolがstart〜end(両端含む)の範囲に絞り込む。日付として解釈できない行は除外される"""
    parsed = pd.to_datetime(df[col], errors="coerce")
    mask = (parsed >= pd.Timestamp(start)) & (parsed <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True), int(mask.sum())


def label_encode_columns(df, cols):
    """カテゴリカラムをラベルエンコーディング（{col}_le を追加）"""
    df = df.copy()
    messages = []
    for c in cols:
        if c in df.columns:
            codes, _ = pd.factorize(df[c])
            df[f"{c}_le"] = codes
            messages.append(f"{c} → {c}_le")
    return df, messages


def bin_column(df, col, bins=4):
    """数値カラムを等頻度でビン分割（{col}_bin を追加）"""
    df = df.copy()
    df[f"{col}_bin"] = pd.qcut(df[col], q=bins, duplicates="drop").astype(str)
    return df


def standardize_columns(df, cols):
    """数値カラムを標準化（{col}_z を追加）"""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            std = df[c].std()
            df[f"{c}_z"] = (df[c] - df[c].mean()) / std if std and std > 0 else 0.0
    return df


def add_interaction_feature(df, col_a, col_b):
    """2つの数値カラムの掛け算で交互作用特徴量を作成（{colA}_x_{colB} を追加）"""
    df = df.copy()
    df[f"{col_a}_x_{col_b}"] = pd.to_numeric(df[col_a], errors="coerce") * pd.to_numeric(df[col_b], errors="coerce")
    return df


def log_transform_columns(df, cols):
    """右に裾が長い数値カラムを対数変換（{col}_log を追加）"""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_log"] = np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0))
    return df


def run_kmeans(df, cols, n_clusters=3, scale=True, random_state=42):
    """指定カラムでK-meansクラスタリングを実行し、クラスタ番号のSeriesを返す"""
    from sklearn.cluster import KMeans

    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < n_clusters:
        return {"error": f"有効なデータが{len(work)}件しかなく、グループ数({n_clusters})より少ないため実行できません"}

    if scale:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(work)
    else:
        X = work.values

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    label_series = pd.Series(labels, index=work.index, name="cluster")
    return {"labels": label_series, "inertia": float(model.inertia_), "index": work.index}


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


def partial_correlation_matrix(df, cols):
    """相関行列の逆行列（精度行列）から偏相関係数を計算する。
    偏相関＝他の全変数の影響を差し引いた上での、2変数間の相関。"""
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < len(cols) + 2:
        return None
    corr = sub.corr().values
    try:
        inv_corr = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return None
    d = np.sqrt(np.diag(inv_corr))
    if np.any(d == 0):
        return None
    pcorr = -inv_corr / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return pd.DataFrame(pcorr, index=sub.columns, columns=sub.columns)


def bayesian_linear_regression(X, y, prior_scale=10.0, a0=0.001, b0=0.001):
    """正規事前分布 + 逆ガンマ事前分布による閉形式ベイズ線形回帰。
    Xは標準化済みを想定。切片項を自動で追加する。
    戻り値: [切片, 特徴量1, 特徴量2, ...] の順でcoef/se/信用区間を持つ辞書のリスト"""
    from scipy import stats as sstats

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    X_design = np.column_stack([np.ones(n), X])
    k = p + 1

    Lambda0 = np.eye(k) / (prior_scale ** 2)
    beta0 = np.zeros(k)

    XtX = X_design.T @ X_design
    Lambda_n = XtX + Lambda0
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    beta_n = Lambda_n_inv @ (Lambda0 @ beta0 + X_design.T @ y)

    an = a0 + n / 2
    bn = b0 + 0.5 * (y @ y + beta0 @ Lambda0 @ beta0 - beta_n @ Lambda_n @ beta_n)

    scale_matrix = (bn / an) * Lambda_n_inv
    dof = 2 * an
    se = np.sqrt(np.clip(np.diag(scale_matrix), 0, None))
    t_crit = sstats.t.ppf(0.975, dof)

    results = []
    for j in range(k):
        results.append({
            "coef": float(beta_n[j]),
            "se": float(se[j]),
            "ci_lower": float(beta_n[j] - t_crit * se[j]),
            "ci_upper": float(beta_n[j] + t_crit * se[j]),
        })
    return results


def bayesian_group_diff(group_a, group_b, n_samples=20000, random_state=42):
    """2群の平均の差について、共役事前分布 + モンテカルロサンプリングでベイズ推定する"""
    rng = np.random.default_rng(random_state)

    def sample_posterior_mean(data):
        data = np.asarray(data, dtype=float)
        n = len(data)
        mean = data.mean()
        a0, b0 = 0.001, 0.001
        an = a0 + n / 2
        var = data.var(ddof=1) if n > 1 else 1.0
        bn = b0 + 0.5 * (n - 1) * var
        sigma2_samples = 1 / rng.gamma(shape=an, scale=1 / bn, size=n_samples)
        mu_samples = rng.normal(loc=mean, scale=np.sqrt(sigma2_samples / n))
        return mu_samples

    samples_a = sample_posterior_mean(group_a)
    samples_b = sample_posterior_mean(group_b)
    diff = samples_a - samples_b
    ci_lower, ci_upper = np.percentile(diff, [2.5, 97.5])
    return {
        "prob_a_gt_b": float((diff > 0).mean()),
        "diff_mean": float(diff.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def bottleneck_analysis(df, target, factors):
    """各要因について、単純相関・偏相関・ベイズ回帰係数(信用区間つき)を算出し、
    「本当に効いていそうな要因」を偏相関の絶対値順にランキングする。
    因果関係を証明するものではなく、統計的な関連の強さを多角的に比較するもの。"""
    sub = df[[target] + factors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < len(factors) + 5:
        return {"error": f"有効なデータが{len(sub)}件と少なすぎます（要因数+5件以上が目安です）"}

    std = sub.std(ddof=0)
    if (std == 0).any():
        zero_cols = std[std == 0].index.tolist()
        return {"error": f"値が一定で分散が無いカラムが含まれています: {', '.join(zero_cols)}"}

    standardized = (sub - sub.mean()) / std
    y = standardized[target].values
    X = standardized[factors].values

    simple_corr = {f: float(standardized[f].corr(standardized[target])) for f in factors}

    pcorr_matrix = partial_correlation_matrix(standardized, [target] + factors)
    if pcorr_matrix is not None:
        partial_corr = {f: float(pcorr_matrix.loc[target, f]) for f in factors}
    else:
        partial_corr = {f: None for f in factors}

    bayes_results = bayesian_linear_regression(X, y, prior_scale=10.0)

    rows = []
    for i, f in enumerate(factors):
        b = bayes_results[i + 1]
        sc, pc = simple_corr[f], partial_corr[f]
        gap = (abs(sc) - abs(pc)) if pc is not None else None
        rows.append({
            "要因": f, "単純相関": sc, "偏相関": pc,
            "標準化回帰係数": b["coef"], "係数_信用区間下限": b["ci_lower"], "係数_信用区間上限": b["ci_upper"],
            "相関の差(見せかけ度)": gap,
        })
    table = pd.DataFrame(rows)
    sort_key = table["偏相関"].abs() if pcorr_matrix is not None else table["単純相関"].abs()
    table = table.iloc[sort_key.sort_values(ascending=False).index].reset_index(drop=True)
    return {"table": table, "n": len(sub)}


def recommend_algorithm(n_rows):
    """データ件数に応じたおすすめアルゴリズムを返す（あくまで目安）"""
    try:
        import lightgbm  # noqa: F401
        lgbm_available = True
    except ImportError:
        lgbm_available = False

    if n_rows < 50:
        return {"algo": "lr",
                "label": "線形/ロジスティック回帰",
                "reason": f"データが{n_rows}件と少なめです。複雑なモデルは少ないデータで過学習しやすいため、"
                          "まずはシンプルな線形/ロジスティック回帰がおすすめです。"}
    if n_rows < 300:
        return {"algo": "dt",
                "label": "決定木",
                "reason": f"データが{n_rows}件です。決定木のようなシンプルなモデルが扱いやすくおすすめです。"}
    if n_rows < 2000:
        return {"algo": "rf",
                "label": "ランダムフォレスト",
                "reason": f"データが{n_rows}件あります。複数の木を組み合わせるランダムフォレストがバランス良くおすすめです。"}
    if lgbm_available:
        return {"algo": "lgbm",
                "label": "LightGBM",
                "reason": f"データが{n_rows}件と十分にあります。LightGBMのような高精度なモデルの効果を発揮しやすいデータ量です。"}
    return {"algo": "hgb",
            "label": "勾配ブースティング (sklearn)",
            "reason": f"データが{n_rows}件と十分にあります。sklearn内蔵の勾配ブースティングがおすすめです"
                      "（lightgbmを追加インストールすると、さらに精度が上がる場合があります）。"}


def judge_regression_metrics(metrics, y_test):
    """回帰の指標が良いか悪いかを日本語で判定する"""
    r2 = metrics["R2"]
    if r2 >= 0.7:
        r2_judge = "とても良い当てはまりです"
    elif r2 >= 0.5:
        r2_judge = "まずまず良い当てはまりです"
    elif r2 >= 0.2:
        r2_judge = "当てはまりはやや弱めです"
    elif r2 >= 0:
        r2_judge = "ほとんど説明できていません"
    else:
        r2_judge = "平均値で予測するより悪い結果です（見直しをおすすめします）"

    y_std = float(pd.Series(y_test).std())
    if not y_std or y_std <= 0:
        mae_ratio, mae_judge = None, "目的変数にばらつきが無いため判定できません"
    else:
        mae_ratio = metrics["MAE"] / y_std
        if mae_ratio < 0.5:
            mae_judge = "誤差は比較的小さめです"
        elif mae_ratio < 1.0:
            mae_judge = "誤差はまずまずの大きさです"
        else:
            mae_judge = "誤差が大きく、予測はあまり当てになりません"

    return {"r2_judge": r2_judge, "mae_ratio": mae_ratio, "mae_judge": mae_judge}


def judge_classification_metrics(metrics, y_test):
    """分類の精度が良いか悪いかを、多数派予測との比較で判定する"""
    baseline = float(pd.Series(y_test).value_counts(normalize=True).max())
    acc = metrics["Accuracy"]
    diff = acc - baseline
    if diff >= 0.20:
        judge = "とても良い精度です（多数派をそのまま予測する場合より大きく上回っています）"
    elif diff >= 0.10:
        judge = "まずまず良い精度です"
    elif diff >= 0.03:
        judge = "多少の改善はありますが、大きな差ではありません"
    else:
        judge = "多数派を予測するのとほとんど変わりません。特徴量やモデルの見直しをおすすめします"
    return {"baseline": baseline, "diff": diff, "judge": judge}


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
        result["judge"] = judge_regression_metrics(result["metrics"], y_test)
    else:
        result["metrics"] = {"Accuracy": accuracy_score(y_test, pred)}
        result["report"] = classification_report(y_test, pred, zero_division=0)
        result["judge"] = judge_classification_metrics(result["metrics"], y_test)
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
# 4. 個人情報のマスク
# ============================================================
with st.expander("🕵️ 4. 個人情報のマスク", expanded=False):
    st.caption("氏名・メールアドレスなどをハッシュ化または連番IDに置き換えます。同じ値は必ず同じ結果になるので、集計・結合には使えます。")
    pii_cols = st.multiselect("対象カラム", options=list(df.columns), key="pii_cols")
    pii_method = st.radio("方法", ["ハッシュ化 (SHA-256)", "連番ID (ID_0001形式)"], key="pii_method")
    pii_salt = ""
    if pii_method == "ハッシュ化 (SHA-256)":
        if "pii_auto_salt" not in st.session_state:
            import secrets
            st.session_state.pii_auto_salt = secrets.token_hex(8)
        pii_salt = st.text_input("ソルト（空欄なら自動生成された値を使用）",
                                  value="", placeholder=st.session_state.pii_auto_salt, key="pii_salt")
        if not pii_salt:
            pii_salt = st.session_state.pii_auto_salt

    if st.button("マスクを適用", key="btn_pii"):
        if pii_cols:
            new_df = df.copy()
            for c in pii_cols:
                if pii_method == "ハッシュ化 (SHA-256)":
                    new_df[c] = mask_pii_hash(new_df[c], pii_salt)
                else:
                    new_df[c] = mask_pii_sequential(new_df[c])
            st.session_state.df = new_df
            st.success(f"{len(pii_cols)}カラムをマスクしました")
            _rerun()
        else:
            st.info("対象カラムを選んでください")

# ============================================================
# 5. 不要カラムの削除
# ============================================================
with st.expander("🗑️ 5. 不要カラムの削除", expanded=False):
    drop_cols = st.multiselect("削除するカラム", options=list(df.columns), key="drop_cols")
    if st.button("削除を適用", key="btn_drop"):
        if drop_cols:
            st.session_state.df = df.drop(columns=drop_cols, errors="ignore")
            st.success(f"{len(drop_cols)}カラムを削除しました")
            _rerun()
        else:
            st.info("削除するカラムを選んでください")

# ============================================================
# 6. 欠損値・外れ値・日付範囲
# ============================================================
with st.expander("🧹 6. 欠損値・外れ値・日付範囲の処理", expanded=False):
    st.subheader("欠損値")
    missing_col_options = list(missing_summary(df).index)
    missing_target_cols = st.multiselect("対象カラム（空欄なら欠損のある全カラム）",
                                          options=missing_col_options, key="missing_cols")
    fillna_method = st.radio("方法", [
        "数値は中央値・カテゴリは最頻値", "数値は平均値・カテゴリは最頻値",
        "指定した値で埋める", "前の行の値で埋める (ffill)", "後の行の値で埋める (bfill)",
        "欠損を含む行を削除",
    ], key="fillna_method")
    method_map = {
        "数値は中央値・カテゴリは最頻値": "median", "数値は平均値・カテゴリは最頻値": "mean",
        "指定した値で埋める": "constant", "前の行の値で埋める (ffill)": "ffill",
        "後の行の値で埋める (bfill)": "bfill", "欠損を含む行を削除": "dropna",
    }
    fill_value_input = None
    if fillna_method == "指定した値で埋める":
        fill_value_input = st.text_input("埋める値（数値カラムは数字として解釈されます）", key="fillna_value")

    if st.button("欠損値処理を適用", key="btn_fillna"):
        method_code = method_map[fillna_method]
        value_to_use = fill_value_input
        if method_code == "constant" and fill_value_input is not None:
            try:
                value_to_use = float(fill_value_input) if "." in fill_value_input else int(fill_value_input)
            except ValueError:
                value_to_use = fill_value_input  # 数値に変換できなければ文字列のまま埋める
        st.session_state.df = fill_missing(df, method_code,
                                            cols=missing_target_cols if missing_target_cols else None,
                                            fill_value=value_to_use)
        st.success("欠損値処理を適用しました")
        _rerun()

    st.subheader("外れ値クリッピング")
    outlier_pct = st.slider("外れ値とみなす範囲（下位・上位それぞれ何%をクリップするか）",
                             min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="outlier_pct")
    outlier_cols = st.multiselect("対象カラム（空欄で数値カラム全体）",
                                   options=list(df.select_dtypes(include="number").columns),
                                   key="outlier_cols")
    if st.button("外れ値クリッピングを適用", key="btn_outlier"):
        st.session_state.df = clip_outliers(df, cols=outlier_cols if outlier_cols else None,
                                             lower_pct=outlier_pct / 100, upper_pct=1 - outlier_pct / 100)
        st.success(f"下位{outlier_pct}% / 上位{outlier_pct}%を外れ値としてクリッピングしました")
        _rerun()

    st.subheader("日付で絞り込む")
    date_filter_col = st.selectbox("対象の日付カラム", options=["(選択しない)"] + list(df.columns), key="date_filter_col")
    if date_filter_col != "(選択しない)":
        parsed_for_filter = pd.to_datetime(df[date_filter_col], errors="coerce")
        valid_dates = parsed_for_filter.dropna()
        if len(valid_dates) > 0:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input("期間（開始日〜終了日）", value=(min_d, max_d),
                                        min_value=min_d, max_value=max_d, key="date_range")
            if st.button("日付で絞り込みを適用", key="btn_date_filter"):
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start, end = date_range
                    new_df, n_kept = filter_by_date_range(df, date_filter_col, start, end)
                    st.session_state.df = new_df
                    st.success(f"{start} 〜 {end} の範囲に絞り込みました（{n_kept}件が残りました）")
                    _rerun()
                else:
                    st.info("開始日と終了日の両方を選んでください")
        else:
            st.warning("この列を日付として解釈できませんでした")

# ============================================================
# 7. 特徴量エンジニアリング
# ============================================================
with st.expander("🛠️ 7. 特徴量エンジニアリング", expanded=False):
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

    st.subheader("ラベルエンコーディング")
    st.caption("文字列を整数コードに変換します（決定木系モデル向け）")
    label_cols = st.multiselect("対象カラム", options=cat_cols_options, key="fe_label_cols")
    if st.button("ラベルエンコーディングを適用", key="btn_label"):
        if label_cols:
            new_df, msgs = label_encode_columns(df, label_cols)
            st.session_state.df = new_df
            for m in msgs:
                st.write("- " + m)
            _rerun()
        else:
            st.info("対象カラムを選んでください")

    st.subheader("ビニング（数値を等頻度で分割）")
    bin_col_options = ["(選択しない)"] + list(df.select_dtypes(include="number").columns)
    bin_col = st.selectbox("対象カラム", options=bin_col_options, key="fe_bin_col")
    bin_count = st.slider("分割数", min_value=2, max_value=10, value=4, key="fe_bin_count")
    if st.button("ビニングを適用", key="btn_bin"):
        if bin_col != "(選択しない)":
            try:
                st.session_state.df = bin_column(df, bin_col, bins=bin_count)
                st.success(f"{bin_col} を{bin_count}分割し、{bin_col}_bin を追加しました")
                _rerun()
            except ValueError as e:
                st.error(f"分割できませんでした（同じ値が多すぎる可能性があります）: {e}")
        else:
            st.info("対象カラムを選んでください")

    st.subheader("標準化")
    st.caption("(x - 平均) / 標準偏差 に変換します。空欄なら数値カラム全体が対象です")
    scale_cols_fe = st.multiselect("対象カラム（空欄で数値カラム全体）",
                                    options=list(df.select_dtypes(include="number").columns), key="fe_scale_cols")
    if st.button("標準化を適用", key="btn_scale_fe"):
        target = scale_cols_fe if scale_cols_fe else list(df.select_dtypes(include="number").columns)
        st.session_state.df = standardize_columns(df, target)
        st.success(f"{len(target)}カラムを標準化しました")
        _rerun()

    st.subheader("交互作用特徴量")
    numeric_cols_fe = list(df.select_dtypes(include="number").columns)
    inter_a = st.selectbox("カラムA", options=["(選択しない)"] + numeric_cols_fe, key="fe_inter_a")
    inter_b = st.selectbox("カラムB", options=["(選択しない)"] + numeric_cols_fe, key="fe_inter_b")
    if st.button("交互作用特徴量を作成", key="btn_inter"):
        if inter_a != "(選択しない)" and inter_b != "(選択しない)" and inter_a != inter_b:
            st.session_state.df = add_interaction_feature(df, inter_a, inter_b)
            st.success(f"{inter_a}_x_{inter_b} を追加しました")
            _rerun()
        else:
            st.info("異なる2つのカラムを選んでください")

    st.subheader("対数変換")
    st.caption("右に裾が長い分布（金額・件数など）の是正に使えます")
    log_cols = st.multiselect("対象カラム", options=list(df.select_dtypes(include="number").columns), key="fe_log_cols")
    if st.button("対数変換を適用", key="btn_log"):
        if log_cols:
            st.session_state.df = log_transform_columns(df, log_cols)
            st.success(f"{len(log_cols)}カラムを対数変換しました")
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
# 8. 可視化
# ============================================================
with st.expander("📊 8. 可視化", expanded=False):
    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols_for_viz = list(df.select_dtypes(include=["object", "string", "category"]).columns)
    viz_tab_hist, viz_tab_box, viz_tab_scatter, viz_tab_heatmap = st.tabs(
        ["ヒストグラム", "箱ひげ図", "散布図", "相関ヒートマップ"])

    with viz_tab_hist:
        if numeric_cols:
            hist_col = st.selectbox("対象カラム", options=numeric_cols, key="hist_col")
            import matplotlib.pyplot as plt
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
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
            ax.set_title("相関ヒートマップ")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("数値カラムが2つ未満のため、相関・ヒートマップは計算できません")

# ============================================================
# 9. 統計検定
# ============================================================
with st.expander("🧪 9. 統計検定", expanded=False):
    test_type = st.selectbox("検定手法", ["実行しない", "t検定 (2群の平均の差)", "Mann-Whitney U検定",
                                        "カイ二乗検定 (独立性)", "ベイズ推定 (2群の平均の差)"], key="test_type")
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
        elif test_type == "ベイズ推定 (2群の平均の差)":
            levels = list(df[col_a].dropna().unique())
            if len(levels) < 2:
                st.error(f"グループが2つ未満のため実行できません: {col_a}")
            else:
                if len(levels) > 2:
                    st.warning(f"グループが3つ以上のため先頭の2グループで比較します -> {levels[0]} / {levels[1]}")
                g1 = pd.to_numeric(df.loc[df[col_a] == levels[0], col_b], errors="coerce").dropna()
                g2 = pd.to_numeric(df.loc[df[col_a] == levels[1], col_b], errors="coerce").dropna()
                bres = bayesian_group_diff(g1, g2)
                st.write(f"**{levels[0]} の平均が {levels[1]} より高い確率: {bres['prob_a_gt_b']:.1%}**")
                st.write(f"平均の差 ({levels[0]} − {levels[1]}): {bres['diff_mean']:.2f}")
                st.write(f"95%信用区間: [{bres['ci_lower']:.2f}, {bres['ci_upper']:.2f}]")
                p = bres["prob_a_gt_b"]
                if p > 0.95 or p < 0.05:
                    st.success("非常に高い確率で差があると言えます")
                elif p > 0.90 or p < 0.10:
                    st.info("差がある可能性が高いですが、断定はできません")
                else:
                    st.info("差があるとは言い切れません")
                st.caption("p値の代わりに「Aの方が高い確率」を直接示すのがベイズ推定の特徴です。")
        else:
            st.info("検定手法を選んでください")

# ============================================================
# 10. ボトルネック分析（相関・偏相関・ベイズ推定）
# ============================================================
with st.expander("🔬 10. ボトルネック分析（相関・偏相関・ベイズ推定）", expanded=False):
    st.caption("「何が結果を左右しているか」を、単純な相関だけでなく、他の要因の影響を差し引いた"
               "**偏相関**や、不確実性を示す**ベイズ信用区間**つきで比較します。")
    st.warning("⚠️ これは統計的な関連の強さを比較するものであり、因果関係を証明するものではありません。"
               "また偏相関は、ここで選んだ要因どうしの関係を調整するだけで、"
               "分析に含めていない別の隠れた要因の影響までは取り除けません。")

    numeric_cols_bn = list(df.select_dtypes(include="number").columns)
    bn_target = st.selectbox("結果（ボトルネックを探したい変数）", options=numeric_cols_bn, key="bn_target")
    bn_factor_candidates = [c for c in numeric_cols_bn if c != bn_target]
    bn_factors = st.multiselect("候補となる要因（2つ以上選んでください）", options=bn_factor_candidates, key="bn_factors")

    if st.button("ボトルネック分析を実行", key="btn_bottleneck"):
        if len(bn_factors) < 2:
            st.info("要因を2つ以上選んでください（偏相関の計算に必要です）")
        else:
            bn_result = bottleneck_analysis(df, bn_target, bn_factors)
            if "error" in bn_result:
                st.error(bn_result["error"])
            else:
                st.caption(f"有効データ: {bn_result['n']}件")
                display_table = bn_result["table"].copy()
                num_cols_disp = ["単純相関", "偏相関", "標準化回帰係数",
                                  "係数_信用区間下限", "係数_信用区間上限", "相関の差(見せかけ度)"]
                display_table[num_cols_disp] = display_table[num_cols_disp].round(3)
                st.dataframe(display_table, use_container_width=True)

                top = bn_result["table"].iloc[0]
                st.success(f"**最有力候補: {top['要因']}**（偏相関 {top['偏相関']:.3f}、"
                           f"標準化係数の95%信用区間 [{top['係数_信用区間下限']:.3f}, {top['係数_信用区間上限']:.3f}]）")

                suspicious = bn_result["table"][bn_result["table"]["相関の差(見せかけ度)"] > 0.2]
                if len(suspicious) > 0:
                    st.warning("⚠️ 以下は単純相関が高い一方、他の要因を考慮すると関連が弱まりました。"
                               "見せかけの相関（他の要因を介した間接的な関係）の可能性があります: "
                               + "、".join(suspicious["要因"].tolist()))

# ============================================================
# 11. クラスタリング (グループ分け)
# ============================================================
with st.expander("🧭 11. クラスタリング（似たものどうしをグループ分け）", expanded=False):
    st.caption("目的変数を決めずに、似たデータどうしを自動でグループ分けします（教師なし学習）。")
    cluster_cols = st.multiselect("グループ分けに使うカラム（数値のみ、2つ以上推奨）",
                                   options=list(df.select_dtypes(include="number").columns), key="cluster_cols")
    n_clusters = st.slider("グループ数", min_value=2, max_value=10, value=3, key="cluster_n")
    cluster_scale = st.checkbox("標準化してから実行する（おすすめ・スケールの違う項目を混ぜるときは特に重要）",
                                 value=True, key="cluster_scale")

    if st.button("クラスタリングを実行", key="btn_cluster"):
        if len(cluster_cols) < 1:
            st.info("カラムを1つ以上選んでください")
        else:
            cluster_result = run_kmeans(df, cluster_cols, n_clusters=n_clusters, scale=cluster_scale)
            if "error" in cluster_result:
                st.error(cluster_result["error"])
            else:
                new_df = df.copy()
                new_df["cluster"] = pd.NA
                new_df.loc[cluster_result["index"], "cluster"] = cluster_result["labels"].values
                st.session_state.df = new_df
                st.session_state.cluster_cols_used = cluster_cols
                st.success(f"{n_clusters}グループに分け、「cluster」カラムを追加しました")
                _rerun()

    if "cluster" in df.columns and st.session_state.get("cluster_cols_used"):
        st.markdown("---")
        st.write("**各グループの件数**")
        st.dataframe(df["cluster"].value_counts().sort_index().rename("件数"), use_container_width=True)

        used_cols = [c for c in st.session_state.cluster_cols_used if c in df.columns]
        plot_rows = df.dropna(subset=["cluster"] + used_cols)
        if len(used_cols) >= 2 and len(plot_rows) > 0:
            import matplotlib.pyplot as plt
            if len(used_cols) == 2:
                plot_x, plot_y = plot_rows[used_cols[0]], plot_rows[used_cols[1]]
                xlabel, ylabel = used_cols[0], used_cols[1]
            else:
                from sklearn.decomposition import PCA
                coords = PCA(n_components=2).fit_transform(plot_rows[used_cols])
                plot_x, plot_y = coords[:, 0], coords[:, 1]
                xlabel, ylabel = "主成分1 (PC1)", "主成分2 (PC2)"
            fig, ax = plt.subplots(figsize=(7, 5))
            scatter = ax.scatter(plot_x, plot_y, c=plot_rows["cluster"].astype(int), cmap="tab10", alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"クラスタリング結果 ({n_clusters}グループ)")
            legend1 = ax.legend(*scatter.legend_elements(), title="グループ")
            ax.add_artist(legend1)
            st.pyplot(fig)
            plt.close(fig)

# ============================================================
# 12. 機械学習 + SHAP
# ============================================================
with st.expander("🤖 12. 機械学習 + SHAP要因分析", expanded=False):
    target = st.selectbox("目的変数", options=list(df.columns), key="ml_target")

    n_for_target = int(df[target].dropna().shape[0])
    recommendation = recommend_algorithm(n_for_target)
    st.info(f"📌 データ量からのおすすめ: **{recommendation['label']}**\n\n{recommendation['reason']}")

    feature_candidates = [c for c in df.columns if c != target]
    features = st.multiselect("特徴量 (空欄なら目的変数以外の数値カラム全て)", options=feature_candidates, key="ml_features")
    task = st.radio("タスク", ["回帰 (数値予測)", "分類 (カテゴリ予測)"], key="ml_task")
    task_code = "reg" if task.startswith("回帰") else "clf"

    algo_labels = ["線形/ロジスティック回帰", "決定木", "ランダムフォレスト",
                   "勾配ブースティング (sklearn)", "LightGBM (要インストール)"]
    algo_codes = ["lr", "dt", "rf", "hgb", "lgbm"]
    default_index = algo_codes.index(recommendation["algo"]) if recommendation["algo"] in algo_codes else 0
    algo_label = st.selectbox("アルゴリズム", algo_labels, index=default_index, key="ml_algo")
    algo_code = algo_codes[algo_labels.index(algo_label)]

    use_dummies = st.checkbox("カテゴリ変数をダミー変数化して特徴量に含める", key="ml_dummies")
    scale = st.checkbox("特徴量を標準化する (線形回帰系で推奨)", key="ml_scale")
    test_size_pct = st.slider("テストデータの割合 (残りを学習に使います)", min_value=10, max_value=50,
                               value=30, step=5, key="ml_test_size")

    if st.button("モデルを学習", key="btn_train"):
        try:
            result = train_and_evaluate(df, target, features, task_code, algo_code,
                                         use_dummies=use_dummies, scale=scale,
                                         test_size=test_size_pct / 100)
            if "error" in result:
                st.error(result["error"])
                st.session_state.trained = None
            else:
                st.session_state.trained = result
                st.session_state.trained_algo = algo_code
                st.session_state.trained_task = task_code
                st.caption(f"🔀 学習データ {len(result['X_train'])}件 / テストデータ {len(result['X_test'])}件 に分割"
                           "（精度はモデルが学習中に見ていないテストデータだけで計算しています）")
                st.subheader("モデル精度")
                m, j = result["metrics"], result["judge"]
                if task_code == "reg":
                    st.write(f"**R2 (決定係数): {m['R2']:.3f}** — {j['r2_judge']}")
                    if j["mae_ratio"] is not None:
                        st.write(f"**MAE (平均絶対誤差): {m['MAE']:.2f}** — {j['mae_judge']}"
                                 f"（目的変数のばらつきの約{j['mae_ratio'] * 100:.0f}%）")
                    else:
                        st.write(f"**MAE (平均絶対誤差): {m['MAE']:.2f}**")
                    st.write(f"**RMSE: {m['RMSE']:.2f}**")
                else:
                    st.write(f"**Accuracy (正解率): {m['Accuracy']:.1%}** — {j['judge']}")
                    st.caption(f"（何も考えずに一番多いクラスだけを予測した場合の正解率: {j['baseline']:.1%}）")
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
                fig = plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, result["X_test"], show=False)
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.warning("shapが未導入のためSHAP分析はスキップされました（pip install shap で導入できます）")
            except Exception as e:
                st.error(f"SHAP計算中にエラーが発生しました: {e}")

# ============================================================
# 13. ダウンロード
# ============================================================
st.header("13. ダウンロード")
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("📥 現在のデータをCSVでダウンロード", data=csv_bytes,
                    file_name="cleaned_data.csv", mime="text/csv")
