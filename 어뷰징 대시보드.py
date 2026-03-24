import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------
# 페이지 설정
# --------------------------------

st.set_page_config(
    page_title="매체별 어뷰징 모니터링",
    layout="wide"
)

# --------------------------------
# 스타일
# --------------------------------

st.markdown("""
<style>

.kpi-card {
background: linear-gradient(145deg,#0f172a,#111827);
border-radius:16px;
padding:24px;
border:1px solid rgba(255,255,255,0.08);
box-shadow:0 10px 25px rgba(0,0,0,0.35);
}

.kpi-title {
font-size:14px;
color:#9CA3AF;
margin-bottom:10px;
}

.kpi-value {
font-size:36px;
font-weight:700;
color:white;
margin-bottom:6px;
}

.kpi-desc {
font-size:13px;
color:#9CA3AF;
margin-bottom:14px;
}

.kpi-badge {
display:inline-block;
padding:6px 12px;
border-radius:20px;
font-size:12px;
font-weight:600;
}

.badge-red {
background:#3b0d0d;
color:#ef4444;
}

.badge-yellow {
background:#3b2f0d;
color:#facc15;
}

.badge-blue {
background:#0b2545;
color:#3b82f6;
}

.badge-green {
background:#062e1f;
color:#22c55e;
}

</style>
""", unsafe_allow_html=True)

card_start = """
<div style="
background:#0b172a;
padding:20px;
border-radius:12px;
border:1px solid rgba(255,255,255,0.05);
margin-bottom:20px;
">
"""

card_end = "</div>"
# --------------------------------
# 데이터 로드
# --------------------------------

@st.cache_data
def load_media():
    return pd.read_parquet("media_dashboard3.parquet")

@st.cache_data
def load_hour():
    return pd.read_parquet("hour_dashboard.parquet")

@st.cache_data
def load_kpi():
    return pd.read_parquet("kpi_dashboard.parquet")

@st.cache_data
def load_media_detail():
    return pd.read_parquet("media_detail_dashboard.parquet")

media_detail_df = load_media_detail()
media_df = load_media()
hour_df = load_hour()
kpi_df = load_kpi()
# --------------------------------
# KPI 계산
# --------------------------------

# 위험 매체
danger_df = media_df[
    media_df["Risk_Label"] == "매우위험"
]

danger_media = len(danger_df)

total_media = len(media_df)

# 클릭 점유율
danger_clicks = media_df["fraud_clicks"].sum()
total_clicks = media_df["clicks"].sum()

danger_click_share = danger_clicks / total_clicks

# Fraud 손실
total_loss = media_df["fraud_loss"].sum()

loss_display = f"{total_loss:,.0f}원" #단위 변환은 대시보드 확인후 처리 

total_ad_cost = media_df["adv_cost"].sum() - media_df["earn_cost"].sum()

loss_ratio = total_loss / total_ad_cost

# --------------------------------
# TOP 10 매체 계산
# --------------------------------
top_media = (
    media_df[media_df["Risk_Label"].isin(["매우위험", "위험"])]
    .sort_values("fraud_ratio", ascending=False)
    .head(10)
    .copy()
)
top_media["rank"] = range(1, len(top_media)+1)

top_media["매체"] = "mda_" + top_media["mda_idx"].astype(str)

top_media["어뷰징비율"] = (top_media["fraud_ratio"] * 100).round(1)

top_media["이벤트"] = top_media["clicks"]

top_media["손실"] = (top_media["fraud_loss"] / 10000).astype(int)

top_media["등급"] = top_media["Risk_Label"]

# --------------------------------
# CVR KPI (kpi parquet 사용)
# --------------------------------

total_cvr = kpi_df["total_cvr"].iloc[0]
clean_cvr = kpi_df["clean_cvr"].iloc[0]
cvr_change = kpi_df["cvr_change"].iloc[0]

# --------------------------------
# 정상 매체
# --------------------------------

normal_media_df = media_df[
    media_df["Risk_Label"] == "정상"
]

normal_media_count = len(normal_media_df)

normal_media_cvr = normal_media_df["CVR"].mean()

# --------------------------------
# 제목
# --------------------------------

st.title("어뷰징 모니터링 대시보드")

# --------------------------------
# KPI
# --------------------------------

col1,col2,col3,col4 = st.columns(4)

# KPI1 위험 매체

with col1:

    st.markdown(f"""
    <div class="kpi-card">

    <div class="kpi-title">위험 매체 수</div>

    <div class="kpi-value">{danger_media}</div>

    <div class="kpi-desc">
    전체 {total_media}개 중
    </div>

    <span class="kpi-badge badge-red">
    클릭 점유율 {danger_click_share:.1%}
    </span>

    </div>
    """, unsafe_allow_html=True)


# KPI2 Fraud 손실

with col2:

    st.markdown(f"""
    <div class="kpi-card">

    <div class="kpi-title">확정 손실액 (광고주)</div>

    <div class="kpi-value">{loss_display}</div>

    <div class="kpi-desc">
    전체 광고비의 {loss_ratio:.1%}
    </div>

    <span class="kpi-badge badge-yellow">
    극단값 전환 비용
    </span>
    """, unsafe_allow_html=True)


# KPI3 CVR 변화

with col3:

    st.markdown(f"""
    <div class="kpi-card">

    <div class="kpi-title">전체 CVR</div>

    <div class="kpi-value">{total_cvr:.2%}</div>

    <div class="kpi-desc">
    극단값 제거 후 {clean_cvr:.2%}
    </div>

    <span class="kpi-badge badge-blue">
    변화 {cvr_change:+.2%}
    </span>

    </div>
    """, unsafe_allow_html=True)


# KPI4 정상 매체

with col4:

    st.markdown(f"""
    <div class="kpi-card">

    <div class="kpi-title">정상 매체 CVR</div>

    <div class="kpi-value">{normal_media_cvr:.2%}</div>

    <div class="kpi-desc">
    정상 매체 평균
    </div>

    <span class="kpi-badge badge-green">
    {normal_media_count}개 매체
    </span>

    </div>
    """, unsafe_allow_html=True)


# --------------------------------
# 1️⃣ 위험 매체 TOP10 / 손실 매체 TOP10
# --------------------------------
st.divider()

st.subheader("위험 매체 TOP10")
st.markdown("### ⚠️ 위험 매체 TOP10")
st.caption("#매체별 어뷰징 비율 및 예상 손실")
st.markdown("""
<div style="
display:flex;
background:#0b172a;            
padding:12px 18px;
border-radius:0px;
font-size:16px;
font-weight:700;
color:#94a3b8;
border-bottom:1px solid rgba(255,255,255,0.05);
">

<div style="width:5%">#</div>
<div style="width:20%">매체</div>
<div style="width:30%">어뷰징 비율</div>
<div style="width:15%;text-align:right">이벤트</div>
<div style="width:15%;text-align:right">손실</div>
<div style="width:15%;text-align:right">등급</div>

</div>
""", unsafe_allow_html=True)
for _, row in top_media.iterrows():

    bar_width = row["어뷰징비율"]

    # 등급 색상 결정
    if row["등급"] == "매우위험":
        label = "🔴 매우위험"
        color = "#ef4444"   # 빨강
    elif row["등급"] == "위험":
        label = "🟡 위험"
        color = "#facc15"   # 노랑
    else:
        color = "#22c55e"
    
    st.markdown(
        f"""
<div style="
display:flex;
align-items:center;
background:#0b172a;
padding:12px 18px;
border-radius:0px;
border:1px solid rgba(255,255,255,0.05);
width:100%;
">

<div style="width:5%;color:#94a3b8">
{row['rank']}
</div>

<div style="width:20%;font-size:16px;font-weight:600;color:white">
{row['매체']}
</div>

<div style="width:30%">

<div style="
background:#1f2937;
height:10px;
border-radius:8px;
overflow:hidden;
">

<div style="
width:{bar_width}%;
background:#ef4444;
height:100%;
"></div>

</div>

<div style="font-size:13px;color:#cbd5e1;margin-top:3px">
{row['어뷰징비율']}%
</div>

</div>

<div style="width:15%;text-align:right;font-size:15px;color:white">
{row['이벤트']:,}
</div>

<div style="width:15%;text-align:right;font-size:15px;color:#f87171;font-weight:600">
{row['손실']:,}만원
</div>

<div style="width:15%;text-align:right;font-size:15px;color:{color};font-weight:700">
{label}
</div>

</div>
""",
        unsafe_allow_html=True,
    )
# --------------------------------
# 2️⃣ 어뷰징 분포
# --------------------------------
import plotly.graph_objects as go

filter_col, scatter_col = st.columns([1,1])

with filter_col:

    st.markdown(card_start, unsafe_allow_html=True)
    title_col, filter_col = st.columns([3,1])
    with title_col:
     st.subheader("📊 클릭 / 전환 비교")

    with filter_col:
        selected_media = st.selectbox(
        "",
        top_media["매체"],
        label_visibility="collapsed",
        key="media_filter"
    )
    
    selected_media_id = int(selected_media.replace("mda_",""))

    detail = media_detail_df[
    media_detail_df["mda_idx"] == selected_media_id
    ]

    normal_click = detail["normal_click"].values[0]
    fraud_click = detail["fraud_click"].values[0]

    normal_conv = detail["normal_conv"].values[0]
    fraud_conv = detail["fraud_conv"].values[0]


    fig = go.Figure()

    fig.add_bar(
        name="클릭",
        x=["정상","어뷰징"],
        y=[normal_click, fraud_click],
        marker_color=["#2176C5","#E4990E"],
        opacity=0.8
    )

    fig.add_bar(
        name="전환",
        x=["정상","어뷰징"],
        y=[normal_conv, fraud_conv],
        marker_color=["#32AF47","#FA093E"],
        opacity=0.8
    )
    fig.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    legend=dict(
        orientation="h",
        y=1.05,
        x=0.5,
        xanchor="center"
    )
)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_end, unsafe_allow_html=True)
# --------------------------------
# Fraud Scatter
# --------------------------------

with scatter_col:
    
    st.markdown(card_start, unsafe_allow_html=True)
    st.subheader(" 클릭수 vs 어뷰징 비율")

    fig2 = px.scatter(
        media_df,
        x="clicks",
        y="fraud_ratio",
        size="fraud_loss",
        color="fraud_ratio",
        hover_data={
        "mda_idx":True,
        "clicks":":,",
        "fraud_ratio":":.2%",
        "fraud_loss":":,"
        },
        color_continuous_scale=[
            "#2DC50E",
            "#e27924",
            "#eb9c25",
            "#eb5029",
            "#fd1e00"
        ]
    )
    fig2.update_traces(marker=dict(opacity=0.7))
    fig2.update_xaxes(type="log")
    fig2.add_hline(
        y=0.3,
        line_dash="dash",
        line_color="red"
    )
    fig2.add_hrect(
    y0=0.3,
    y1=1,
    fillcolor="#ef4444",
    opacity=0.08,
    line_width=0
    )
    fig2.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Clicks (log scale)",
    yaxis_title="Fraud Ratio",
    font=dict(color="white")
    )

    fig2.update_traces(
    marker=dict(
        opacity=0.8,
        line=dict(width=1, color="#0b172a")
    ))
    
    

    
    
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(card_end, unsafe_allow_html=True)

# --------------------------------
# 3️⃣ 시간대 어뷰징 분석
# --------------------------------

hour_media_df = pd.read_parquet("hour_media_dashboard.parquet")

media_hour = hour_media_df[
    hour_media_df["mda_idx"] == selected_media_id
]
selected_media_id_hour = int(selected_media.replace("mda_",""))
fig = px.line(
    media_hour,
    x="hour",
    y="clicks",
    color="type",
    markers=True,
    color_discrete_map={
        "정상": "#38bdf8",
        "어뷰징": "#ef4444"
    }
)

fig.update_layout(
    title=f"{selected_media} 시간대별 클릭 패턴",
    xaxis_title="Hour",
    yaxis_title="Clicks",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=350
)

st.plotly_chart(fig, use_container_width=True)



# --------------------------------
# 4️⃣ 매체 상세
# --------------------------------
st.divider()
st.subheader("매체 상세 분석")

selected_media = st.selectbox(
    "매체 선택",
    media_df["mda_idx"]
)

detail = media_df[
    media_df["mda_idx"] == selected_media
]

st.dataframe(detail)

st.divider()

# --------------------------------
# 5️⃣ 전체 데이터
# --------------------------------

st.subheader("전체 매체 데이터")

st.dataframe(media_df)
