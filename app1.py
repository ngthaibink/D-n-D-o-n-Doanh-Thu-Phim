import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =============================
# Load model và columns
# =============================
@st.cache_resource
def load_assets():
    try:
        with open("movie_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("model_columns.pkl", "rb") as f:
            columns = pickle.load(f)

        return model, columns

    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, None


model, model_columns = load_assets()

# Nếu load lỗi thì dừng app
if model is None or model_columns is None:
    st.stop()


# =============================
# Giao diện chính
# =============================

st.title("🎬 Dự đoán doanh thu phim")
st.markdown("Ứng dụng Machine Learning để dự đoán doanh thu phim dựa vào các mô hình đã thử nghiệm từ trước \
để đưa ra kết quả gần đúng nhất")
st.divider()
st.header("📖 About the Model")
st.write("""
Model sử dụng Random Forest Regression để dự đoán doanh thu phim dựa trên
các đặc trưng của phim như ngân sách, điểm IMDb, thời lượng, số lượt đánh giá
và thể loại phim.
""")


# =============================
# Sidebar nhập dữ liệu
# =============================

st.sidebar.title("📊 Nhập thông tin phim")

budget = st.sidebar.number_input("Ngân sách (USD)", value=10000000)

score = st.sidebar.slider("Điểm IMDb", 0.0, 10.0, 7.0)

runtime = st.sidebar.number_input("Thời lượng (phút)", value=120)

votes = st.sidebar.number_input("Số lượng votes", value=1000000)

genre_display = st.sidebar.selectbox(
    "Thể loại",
    [
        "Animation","Biography","Comedy","Crime","Drama",
        "Family","Fantasy","Horror","Mystery","Romance",
        "Sci-Fi","Thriller","Action"
    ]
)

predict_button = st.sidebar.button("🚀 Dự đoán đii")


# =============================
# Khu vực hiển thị kết quả
# =============================

st.subheader("💰 Kết quả dự đoán")

if predict_button:

    try:

        # tạo dataframe đúng format model
        input_df = pd.DataFrame(
            np.zeros((1, len(model_columns))),
            columns=model_columns
        )

        # điền dữ liệu numeric
        if "budget" in input_df.columns:
            input_df["budget"] = budget

        if "score" in input_df.columns:
            input_df["score"] = score

        if "runtime" in input_df.columns:
            input_df["runtime"] = runtime

        if "votes" in input_df.columns:
            input_df["votes"] = votes


        # one-hot encode genre
        genre_col = f"genre_{genre_display}"

        if genre_col in input_df.columns:
            input_df[genre_col] = 1


        # predict
        prediction = model.predict(input_df)[0]

        st.divider()

        st.metric(
            label="Doanh thu dự đoán",
            value=f"${prediction:,.0f}"
        )
        st.subheader("📊 So sánh Budget và Doanh thu dự đoán")

        chart_data = pd.DataFrame({
             "Giá trị": [budget, prediction]
               }, index=["Budget", "Predicted Revenue"])

        st.bar_chart(chart_data)

    except Exception as e:

        st.error(f"Lỗi dự đoán: {e}")
   