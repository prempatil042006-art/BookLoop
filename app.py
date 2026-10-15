from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "books_dataset.csv"
FEATURES = ["subject", "original_price", "age_years", "condition", "demand_level", "latest_edition"]
CATEGORICAL = ["subject", "condition", "demand_level", "latest_edition"]
NUMERIC = ["original_price", "age_years"]
REQUIRED = ["book_title", *FEATURES, "resale_price"]

BRANCHES = {
    "Computer Engineering": ["Python", "Data Structures", "Algorithms", "Database Management Systems", "Operating Systems", "Computer Networks", "Computer Organization", "Software Engineering", "Cyber Security", "Machine Learning"],
    "AI & Data Science": ["Python", "Statistics", "Machine Learning", "Artificial Intelligence", "Data Science", "Deep Learning", "Natural Language Processing", "Data Visualization"],
    "Information Technology": ["Python", "Web Technology", "Database Management Systems", "Cloud Computing", "Cyber Security", "Computer Networks", "Software Engineering"],
    "Mechanical Engineering": ["Engineering Mechanics", "Strength of Materials", "Thermodynamics", "Fluid Mechanics", "Manufacturing Processes", "Machine Design", "Heat Transfer"],
    "Electrical Engineering": ["Engineering Mathematics", "Electrical Machines", "Power Systems", "Control Systems", "Signals and Systems", "Power Electronics", "Circuit Theory"],
    "Electronics & Telecommunication": ["Digital Electronics", "Analog Electronics", "Microprocessors", "Signals and Systems", "Communication Systems", "Control Systems", "VLSI Design"],
    "Civil Engineering": ["Surveying", "Structural Engineering", "Environmental Engineering", "Engineering Mechanics", "Fluid Mechanics", "Geotechnical Engineering", "Concrete Technology"],
    "Chemical Engineering": ["Chemical Process Calculations", "Chemical Reaction Engineering", "Heat Transfer", "Mass Transfer", "Fluid Mechanics", "Process Control"],
    "Aerospace Engineering": ["Aerodynamics", "Aircraft Structures", "Flight Mechanics", "Propulsion", "Fluid Mechanics"],
    "Common First Year": ["Engineering Mathematics", "Engineering Physics", "Engineering Chemistry", "Basic Electrical Engineering", "Engineering Mechanics", "Programming for Problem Solving"],
}

st.set_page_config(page_title="BookLoop | Buy · Sell · Reuse", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# Navigation callbacks update the sidebar radio's state before the next rerun.
def go_to(page_name):
    st.session_state["active_page"] = page_name
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #10265b 0%, #071329 38%, #040b18 78%); color: #eef5ff; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1320px; padding-top: 1.2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#061127,#081a36); border-right: 1px solid #153665; }
section[data-testid="stSidebar"] * { color: #eaf2ff; }
.hero { border: 1px solid #174a83; border-radius: 26px; padding: 2.2rem 2.4rem; margin: .3rem 0 1.4rem; background: radial-gradient(circle at 82% 15%,rgba(0,209,255,.25),transparent 28%), linear-gradient(115deg,#0a1a3b 0%,#102b69 55%,#35106a 100%); box-shadow: 0 18px 55px rgba(0,0,0,.25); }
.kicker { color:#3ee7f2; font-size:.76rem; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { font-family:Manrope,sans-serif; font-size:clamp(2rem,4vw,3.25rem); line-height:1.08; font-weight:800; color:#fff; margin:.6rem 0; letter-spacing:-1.3px; }
.hero p { color:#c6dcff; font-size:1.02rem; max-width:720px; margin:.45rem 0; }
.glow { color:#32d8ff; }
.panel { background:linear-gradient(145deg,rgba(13,35,72,.92),rgba(7,20,43,.96)); border:1px solid #174477; border-radius:19px; padding:1.1rem 1.25rem; margin-bottom:.8rem; }
.small-label { color:#47dff0; text-transform:uppercase; letter-spacing:1.5px; font-size:.72rem; font-weight:800; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#0e2851,#0a1b38); border:1px solid #1a4b7e; border-radius:17px; padding:15px 18px; }
div[data-testid="stMetricLabel"] { color:#a9c6ed; }
div[data-testid="stMetricValue"] { color:#3ee7d4; }
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button { border:1px solid #2786dc; border-radius:12px; background:linear-gradient(100deg,#6139f5,#078ce8); color:white; font-weight:800; min-height:2.8rem; }
.stButton>button:hover, .stFormSubmitButton>button:hover { border-color:#54e7ff; color:white; box-shadow:0 0 18px rgba(27,184,255,.2); }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea { background:#0b2041!important; color:#f3f8ff!important; border-color:#245184!important; border-radius:10px!important; }
label, .stMarkdown, p, li { color:#e2edff; }
[data-testid="stForm"] { background:rgba(7,22,46,.82); border:1px solid #1b4b7d; border-radius:20px; padding:1.25rem; }
hr { border-color:#18375e; }
[data-testid="stDataFrame"] { border:1px solid #1b4775; border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

def clean_data(raw):
    d = raw.copy()
    if "book_title" not in d.columns and "cbook_title" in d.columns:
        d = d.rename(columns={"cbook_title": "book_title"})
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError("CSV mein ye required columns missing hain: " + ", ".join(missing))
    for c in ["book_title", "subject", "condition", "demand_level", "latest_edition"]:
        d[c] = d[c].astype("string").str.strip()
    for c in ["original_price", "age_years", "resale_price"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=REQUIRED)
    d = d[(d.original_price > 0) & (d.age_years >= 0) & (d.resale_price >= 0)].copy()
    for c in ["book_title", "subject", "condition", "demand_level", "latest_edition"]:
        d[c] = d[c].astype(str)
    d["condition"] = d["condition"].str.strip().str.title()
    d["demand_level"] = d["demand_level"].str.strip().str.title()
    d["latest_edition"] = d["latest_edition"].str.strip().str.title()
    return d.reset_index(drop=True)

def make_pipeline(estimator):
    prep = ColumnTransformer([
        ("cats", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("nums", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])

def train_models(data):
    X, y = data[FEATURES], data["resale_price"]
    builders = {
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=42),
        "Linear Regression": LinearRegression(),
    }
    rows, fitted = [], {}
    if len(data) >= 12:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
        for name, estimator in builders.items():
            test_model = make_pipeline(estimator)
            test_model.fit(X_train, y_train)
            pred = test_model.predict(X_test)
            rows.append({"Model": name, "MAE (₹)": round(mean_absolute_error(y_test, pred), 2), "RMSE (₹)": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2), "R²": round(r2_score(y_test, pred), 3)})
    else:
        rows = [{"Model": name, "MAE (₹)": np.nan, "RMSE (₹)": np.nan, "R²": np.nan} for name in builders]
    for name, estimator in builders.items():
        model = make_pipeline(estimator)
        model.fit(X, y)
        fitted[name] = model
    return pd.DataFrame(rows).sort_values("RMSE (₹)", na_position="last").reset_index(drop=True), fitted

with st.sidebar:
    st.markdown("# 📚 BookLoop")
    st.caption("BUY  ·  SELL  ·  REUSE")
    st.divider()
    page = st.radio("MENU", ["🏠 Home", "✨ Predict Price", "🔎 Browse Books", "🏷️ My Listings", "📊 Model Insights", "🧩 Book Groups"], key="active_page", label_visibility="visible")
    st.divider()
    st.caption("Give books a second life ♻️")
    uploaded = st.file_uploader("Use another dataset (optional)", type=["csv"])

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_data()
    books = clean_data(raw)
except Exception as err:
    st.markdown('<div class="hero"><div class="kicker">Campus book marketplace</div><h1>📚 BookLoop</h1><p>Buy · Sell · Reuse</p></div>', unsafe_allow_html=True)
    st.error(f"Dataset load nahi hua: {err}")
    st.info("Confirm karo ki app.py aur books_dataset.csv same project folder mein hain.")
    st.stop()

if len(books) < 2:
    st.error("Prediction ke liye dataset mein kam se kam 2 valid rows honi chahiye.")
    st.stop()
metrics, models = train_models(books)

# Hero header stays consistent across pages.
st.markdown('<div class="hero"><div class="kicker">Your campus book marketplace assistant</div><h1>Know Your Book’s<br><span class="glow">Real Resale Value.</span></h1><p>Smart second-hand engineering book price estimates. Add book details, choose your branch, and make a better-informed selling decision.</p><p>⚡ Quick estimate &nbsp; · &nbsp; 📘 Branch-wise subjects &nbsp; · &nbsp; ♻️ Reuse more books</p></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("📚 Books in dataset", f"{len(books):,}")
c2.metric("📖 Subjects in dataset", f"{books.subject.nunique():,}")
c3.metric("🎓 Branches", f"{len(BRANCHES)}")
st.caption("Note: prediction quality depends on the quality and size of your actual sold-book data. Demo/sample data cannot guarantee market prices.")

if page == "🏠 Home":
    st.markdown("## Start here")
    st.markdown('<div class="panel"><div class="small-label">QUICK ACTION</div><h3>✨ Predict a book’s resale price</h3><p>Choose your engineering branch, subject and book condition. Add the author, publisher and ISBN to keep the book details together.</p></div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.button("✨ Predict Book Price", use_container_width=True, on_click=go_to, args=("✨ Predict Price",))
        st.markdown('<div class="panel"><h4>📘 Branch-wise subjects</h4><p>Computer, AI & Data Science, Electrical, Mechanical, Civil and more.</p></div>', unsafe_allow_html=True)
    with b:
        st.button("🔎 Browse Books", use_container_width=True, on_click=go_to, args=("🔎 Browse Books",))
        st.markdown('<div class="panel"><h4>🏷️ Keep track of listings</h4><p>Save book details entered during this session in My Listings.</p></div>', unsafe_allow_html=True)

elif page == "✨ Predict Price":
    st.markdown("## ✨ Predict Book Price")
    st.write("Pehle branch select karo — subject list turant uske hisaab se change hogi.")
    branch = st.selectbox("Engineering branch *", list(BRANCHES.keys()), key="predict_branch")
    subject = st.selectbox("Book subject *", BRANCHES[branch], key="predict_subject")
    with st.form("book_predict_form", border=True):
        st.markdown("### 📚 Book information")
        r1, r2 = st.columns(2, gap="large")
        with r1:
            title = st.text_input("Book title *", placeholder="e.g. Introduction to Algorithms")
            author = st.text_input("Author name", placeholder="e.g. Thomas H. Cormen")
            publisher = st.text_input("Publisher", placeholder="e.g. Pearson / McGraw Hill")
            isbn = st.text_input("ISBN (optional)", placeholder="ISBN-10 or ISBN-13")
        with r2:
            book_edition = st.text_input("Edition / year", placeholder="e.g. 4th edition, 2023")
            latest = st.selectbox("Is it the latest edition?", ["Yes", "No"])
            condition = st.selectbox("Book condition *", ["Like New", "Good", "Fair", "Poor"])
            demand = st.select_slider("Estimated demand", options=["Low", "Medium", "High"], value="Medium")
        p1, p2, p3 = st.columns(3)
        with p1:
            price = st.number_input("Original price (₹) *", min_value=1.0, max_value=100000.0, value=800.0, step=50.0)
        with p2:
            age = st.number_input("Book age (years) *", min_value=0.0, max_value=50.0, value=2.0, step=1.0)
        with p3:
            model_choice = st.selectbox("Prediction model", list(models.keys()))
        submitted = st.form_submit_button("⚡ Predict Resale Price", use_container_width=True)
    if submitted:
        if not title.strip():
            st.error("Bhai, book title zaroor bharna.")
        else:
            inp = pd.DataFrame([{"subject": subject, "original_price": price, "age_years": age, "condition": condition, "demand_level": demand, "latest_edition": latest}])
            estimate = max(0.0, min(float(models[model_choice].predict(inp)[0]), float(price)))
            new_listing = {"Book Title": title.strip(), "Author": author.strip(), "Publisher": publisher.strip(), "ISBN": isbn.strip(), "Branch": branch, "Subject": subject, "Edition": book_edition.strip(), "Original Price (₹)": price, "Age (years)": age, "Condition": condition, "Demand": demand, "Estimated Resale (₹)": round(estimate, 2)}
            st.session_state["last_prediction"] = {"listing": new_listing, "estimate": estimate, "model": model_choice}
    result = st.session_state.get("last_prediction")
    if result:
        item = result["listing"]
        estimate = result["estimate"]
        st.markdown('<div class="panel"><div class="small-label">ESTIMATED RESALE PRICE</div></div>', unsafe_allow_html=True)
        left, right = st.columns([1, 2])
        with left:
            st.metric("Estimated value", f"₹{estimate:,.0f}")
            st.caption(f"Around {estimate / item['Original Price (₹)'] * 100:.0f}% of original price")
        with right:
            st.success(f"**{item['Book Title']}** ka estimated resale price: **₹{estimate:,.0f}**")
            st.write(f"**Author:** {item['Author'] or 'Not provided'}  ")
            st.write(f"**Publisher:** {item['Publisher'] or 'Not provided'}  ")
            st.write(f"**ISBN:** {item['ISBN'] or 'Not provided'}  ")
            st.write(f"**Branch / subject:** {item['Branch']} / {item['Subject']}")
            st.write(f"**Edition:** {item['Edition'] or 'Not provided'} · **Condition:** {item['Condition']}")
        st.warning("Estimate dataset ke patterns par based hai; actual selling price alag ho sakti hai. Author, publisher aur ISBN abhi book details ke liye capture hote hain — current CSV mein inka training data nahi hai, isliye model inka effect calculate nahi karta.")
        if st.button("＋ Save to My Listings", use_container_width=True):
            st.session_state.setdefault("my_listings", []).append(item)
            st.success("Listing save ho gayi. My Listings page par dekh sakte ho.")

elif page == "🔎 Browse Books":
    st.markdown("## 🔎 Browse Books")
    q = st.text_input("Search title or subject", placeholder="Search for Python, DBMS...")
    branch_filter = st.selectbox("Filter by subject", ["All subjects"] + sorted(books["subject"].dropna().astype(str).unique().tolist()))
    d = books.copy()
    if q.strip():
        mask = d["book_title"].astype(str).str.contains(q, case=False, na=False) | d["subject"].astype(str).str.contains(q, case=False, na=False)
        d = d[mask]
    if branch_filter != "All subjects":
        d = d[d["subject"].astype(str) == branch_filter]
    st.caption(f"Showing {len(d)} of {len(books)} books")
    st.dataframe(d, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download books CSV", d.to_csv(index=False).encode("utf-8"), "bookloop_books.csv", "text/csv")

elif page == "🏷️ My Listings":
    st.markdown("## 🏷️ My Listings")
    listings = st.session_state.get("my_listings", [])
    if listings:
        st.dataframe(pd.DataFrame(listings), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download my listings", pd.DataFrame(listings).to_csv(index=False).encode("utf-8"), "my_bookloop_listings.csv", "text/csv")
        if st.button("🗑️ Clear all my listings", type="secondary"):
            st.session_state["my_listings"] = []
            st.success("My Listings cleared.")
            st.rerun()
    else:
        st.info("Abhi koi listing saved nahi hai. Predict Price page par details bharke ‘Save to My Listings’ dabao. Listings filhaal current session tak rehti hain.")

elif page == "📊 Model Insights":
    st.markdown("## 📊 Model Insights")
    st.write("Models ko held-out test split par compare kiya gaya hai; lower MAE/RMSE generally better hota hai.")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    with st.expander("Metrics ka matlab"):
        st.markdown("- **MAE:** average prediction error rupees mein. Lower is better.\n- **RMSE:** large errors ko zyada penalty deta hai. Lower is better.\n- **R²:** model fit ka measure; negative score bhi possible hai.")
    st.caption("Sample/generated data ke scores real-world accuracy prove nahi karte.")

else:
    st.markdown("## 🧩 Similar Book Groups")
    d = books.copy()
    d["condition_score"] = d.condition.map({"Poor": 1, "Fair": 2, "Good": 3, "Like New": 4}).fillna(2)
    d["demand_score"] = d.demand_level.map({"Low": 1, "Medium": 2, "High": 3}).fillna(2)
    d["edition_score"] = d.latest_edition.map({"No": 0, "Yes": 1}).fillna(0)
    cols = ["original_price", "age_years", "condition_score", "demand_score", "edition_score"]
    k = min(3, len(d))
    d["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(d[cols]))
    summary = d.groupby("Cluster").agg(Books=("book_title", "count"), Avg_Original_Price=("original_price", "mean"), Avg_Resale_Price=("resale_price", "mean"), Avg_Age=("age_years", "mean")).round(2).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("Cluster")["Books"])
    st.caption("Cluster numbers are just group labels, not quality rankings.")

st.divider()
st.caption("BookLoop · Campus book resale assistant · Python + Streamlit + scikit-learn")
