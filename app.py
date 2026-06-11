import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import io

# ==========================================
# CẤU HÌNH TRANG STREAMLIT ĐẦU TIÊN
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==========================================
# CÁC HÀM CACHE & BỔ TRỢ
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hash của cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Dấu hiệu nhận biết tập biến X từ dữ liệu mẫu
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện mẫu (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các biến từ X_1 đến X_14 và cột mục tiêu 'default'"
    )
    
    st.divider()
    
    # 2. Cấu hình tham số mô hình AI (Random Forest dựa trên Notebook)
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: RandomForestClassifier")
    
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng độc lập."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        min_value=0, 
        max_value=9999, 
        value=42, 
        step=1,
        help="Đảm bảo tính tái lập kết quả huấn luyện mô hình."
    )
    
    with st.expander("Tham số nâng cao"):
        criterion = st.selectbox(
            "Tiêu chí phân tách (criterion)", 
            options=["gini", "entropy", "log_loss"], 
            index=0,
            help="Hàm đo lường chất lượng của phép phân tách."
        )
        max_depth = st.slider(
            "Độ sâu tối đa (max_depth)", 
            min_value=1, 
            max_value=50, 
            value=20, 
            help="Độ sâu tối đa của các cây quyết định. Để trống nếu muốn phát triển tối đa."
        )
        test_size = st.slider(
            "Tỷ lệ dữ liệu kiểm định (test_size)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2, 
            step=0.05,
            help="Tỷ lệ tập dữ liệu dùng để đánh giá mô hình."
        )

    st.divider()
    
    # 3. Nút hành động duy nhất để kích hoạt huấn luyện
    train_button = st.button(
        "🚀 Huấn luyện Mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu quy trình trích xuất, phân tách và huấn luyện mô hình AI."
    )

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Ứng dụng Phát hiện Giao dịch Gian lận")
st.caption("Hệ thống hỗ trợ phân tích rủi ro tín dụng và phát hiện hành vi gian lận tài chính dựa trên học máy.")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên tệp dữ liệu mẫu (.csv hoặc .xlsx) tại Sidebar bên trái để bắt đầu khám phá.")
    st.stop()

# Đọc dữ liệu khi đã tải file lên thành công
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("Dữ liệu không hợp lệ hoặc không thể đọc đúng định dạng.")
    st.stop()

# Kiểm tra schema dữ liệu cơ bản
missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
if missing_cols:
    st.error(f"Cấu trúc tệp dữ liệu không hợp lệ. Thiếu các cột bắt buộc: {missing_cols}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.divider()

# ==========================================
# KHỐI HUẤN LUYỆN (Chỉ chạy khi bấm nút và lưu vào session_state)
# ==========================================
if train_button:
    with st.spinner("⏳ Đang huấn luyện mô hình... Vui lòng chờ trong giây lát."):
        # Chuẩn bị dữ liệu đầu vào
        X = df_raw[FEATURES]
        y = df_raw[TARGET]
        
        # Phân tách tập dữ liệu giống cấu trúc kiểm định
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Khởi tạo và huấn luyện mô hình RandomForestClassifier theo tùy chỉnh
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Đánh giá và chấm điểm dữ liệu
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu kết quả vào session state để tái sử dụng trên các tab
        st.session_state['trained_model'] = model
        st.session_state['features_list'] = FEATURES
        st.session_state['evaluation'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report_dict': classification_report(y_test, y_pred, output_dict=True),
            'y_test': y_test.values,
            'y_pred': y_pred
        }
        st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab tiếp theo để xem kết quả chi tiết.")

# ==========================================
# KHỞI TẠO HỆ THỐNG TAB NỘI DUNG CHÍNH
# ==========================================
tab_overview, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab_overview:
    st.subheader("Phân tích Cấu trúc & Thống kê Dữ liệu Thô")
    
    # 1. Kích thước dữ liệu thông qua metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số dòng dữ liệu", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số cột dữ liệu", f"{df_raw.shape[1]}")
    col_m3.metric("Dung lượng file", f"{len(file_bytes)/(1024*1024):.2f} MB")
    
    # 2. Xem dữ liệu mẫu ban đầu
    st.markdown("#### 🔍 Xem trước 5 hàng dữ liệu đầu tiên")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    # 3. Thống kê mô tả các biến đặc trưng tham gia mô hình
    st.markdown("#### 📐 Chỉ số thống kê mô tả (Biến mô hình)")
    selected_features = FEATURES + [TARGET]
    st.dataframe(df_raw[selected_features].describe().T, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab_viz:
    st.subheader("Trực quan hóa Phân phối Biến và Tương quan")
    
    # Lựa chọn hiển thị linh hoạt cho 4 biểu đồ cân đối
    viz_features = st.multiselect(
        "Chọn các biến đặc trưng để hiển thị trực quan (Mặc định 3 biến đầu tiên + Biến mục tiêu)",
        options=FEATURES,
        default=FEATURES[:3]
    )
    
    # Layout Lưới 2x2 cho biểu đồ trực quan
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    
    # 1. Biểu đồ biến mục tiêu (y) - Bắt buộc ưu tiên hàng đầu
    with row1_c1:
        target_counts = df_raw[TARGET].value_counts().reset_index()
        target_counts.columns = [TARGET, 'Số lượng']
        target_counts[TARGET] = target_counts[TARGET].map({0: 'Hợp lệ (0)', 1: 'Gian lận/Rủi ro (1)'})
        fig_target = px.bar(
            target_counts, x=TARGET, y='Số lượng',
            title=f"Phân phối của biến mục tiêu ({TARGET})",
            color=TARGET, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    # 2. Phân phối các biến đặc trưng X được chọn từ người dùng
    containers = [row1_c2, row2_c1, row2_c2]
    for idx, feature in enumerate(viz_features[:3]):
        if idx < len(containers):
            with containers[idx]:
                fig_feat = px.histogram(
                    df_raw, x=feature, color=TARGET,
                    title=f"Biểu đồ phân phối của biến {feature}",
                    barmode='overlay', marginal='box',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH"
# ------------------------------------------
with tab_metrics:
    st.subheader("Đánh giá độ chính xác & Chỉ tiêu Mô hình")
    
    if 'evaluation' not in st.session_state:
        st.info("💡 Vui lòng thiết lập cấu hình và nhấn nút **[🚀 Huấn luyện Mô hình]** tại Sidebar bên trái để xem kết quả kiểm định.")
    else:
        eval_data = st.session_state['evaluation']
        
        # 1. Hiển thị các chỉ tiêu vô hướng cốt lõi bằng metrics
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Độ chính xác (Accuracy)", f"{eval_data['accuracy']:.4f}")
        m_c2.metric("Độ chuẩn xác (Precision)", f"{eval_data['precision']:.4f}")
        m_c3.metric("Độ nhạy (Recall)", f"{eval_data['recall']:.4f}")
        m_c4.metric("Điểm F1-Score", f"{eval_data['f1']:.4f}")
        
        st.markdown("---")
        
        # 2. Phân tích chi tiết qua Ma trận nhầm lẫn và Bảng báo cáo
        g_c1, g_c2 = st.columns(2)
        
        with g_c1:
            st.markdown("#### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = eval_data['confusion_matrix']
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Dự báo Hợp lệ (0)', 'Dự báo Gian lận (1)'],
                y=['Thực tế Hợp lệ (0)', 'Thực tế Gian lận (1)'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14}
            ))
            fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with g_c2:
            st.markdown("#### 📝 Báo cáo phân loại chi tiết (Classification Report)")
            report_df = pd.DataFrame(eval_data['report_dict']).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab_inference:
    st.subheader("Chẩn đoán & Dự báo Giao dịch Mới")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng hoàn thành bước huấn luyện mô hình ở cấu hình trước khi thực hiện dự báo dữ liệu mới.")
    else:
        model = st.session_state['trained_model']
        
        # Lựa chọn chế độ dự báo đầu vào
        mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào:",
            options=["Nhập thông số trực tiếp", "Tải lên tệp chứa danh sách giao dịch mới"],
            horizontal=True
        )
        
        st.divider()
        
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP QUA FORM INPUT
        if mode == "Nhập thông số trực tiếp":
            st.markdown("#### 🎛️ Nhập thông tin chi tiết các biến X (Từ X_1 đến X_14)")
            
            # Khởi tạo giá trị mặc định dựa trên median của dữ liệu huấn luyện
            with st.form("inference_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, col_name in enumerate(FEATURES):
                    col_idx = idx % 3
                    # Lấy phân bố thống kê mẫu làm gợi ý thông minh
                    default_val = float(df_raw[col_name].median())
                    min_val = float(df_raw[col_name].min())
                    max_val = float(df_raw[col_name].max())
                    
                    with form_cols[col_idx]:
                        input_data[col_name] = st.number_input(
                            f"Thông số {col_name}",
                            min_value=min_val * 2 if min_val < 0 else 0.0,
                            max_value=max_value * 2,
                            value=default_val,
                            format="%.6f"
                        )
                
                submit_pred = st.form_submit_button("🔍 Chẩn đoán Giao dịch", type="primary")
                
            if submit_pred:
                # Chuyển đổi dữ liệu input sang cấu hình mô hình
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                # Hiển thị kết quả trực quan cho người dùng
                st.markdown("### Kết quả phân tích rủi ro:")
                if prediction == 1:
                    st.error(f"🚨 **CẢNH BÁO: Giao dịch có dấu hiệu Gian lận / Rủi ro cao!** (Xác suất rủi ro: {probabilities[1]*100:.2f}%)")
                else:
                    st.success(f"✅ **AN TOÀN: Giao dịch được thẩm định bình thường.** (Xác suất an toàn: {probabilities[0]*100:.2f}%)")
                    
                st.metric(label="Mức độ rủi ro tiềm ẩn (Gian lận)", value=f"{probabilities[1]*100:.2f}%")

        # CHẾ ĐỘ 2 — TẢI FILE HÀNG LOẠT
        elif mode == "Tải lên tệp chứa danh sách giao dịch mới":
            st.markdown("#### 📂 Dự báo hàng loạt theo lô dữ liệu")
            st.caption("Yêu cầu: Tệp phải chứa đầy đủ các cột thuộc tính đầu vào từ X_1 đến X_14.")
            
            new_file = st.file_uploader(
                "Tải lên tệp dữ liệu giao dịch cần chấm điểm (.csv, .xlsx)",
                type=["csv", "xlsx"],
                key="inference_file_uploader"
            )
            
            if new_file is not None:
                new_bytes = new_file.getvalue()
                df_new = load_data(new_bytes, new_file.name)
                
                if df_new is not None:
                    # Kiểm tra xem có đủ các biến X_1 đến X_14 không
                    missing_inf_cols = [c for c in FEATURES if c not in df_new.columns]
                    if missing_inf_cols:
                        st.error(f"Tệp tải lên thiếu các cột đầu vào cần thiết cho mô hình: {missing_inf_cols}")
                    else:
                        # Tiến hành dự báo hàng loạt
                        X_new = df_new[FEATURES]
                        batch_predictions = model.predict(X_new)
                        batch_probs = model.predict_proba(X_new)[:, 1]
                        
                        # Gắn ngược kết quả vào dataframe xuất ra
                        df_result = df_new.copy()
                        df_result['Dự_Báo_Kết_Quả'] = batch_predictions
                        df_result['Xác_Suất_Rủi_Rô_Gian_Lận'] = batch_probs
                        
                        st.success(f"Chấm điểm thành công {len(df_result)} giao dịch!")
                        
                        # Hiển thị tổng quan kết quả
                        fraud_count = int(np.sum(batch_predictions))
                        st.metric("Tổng số giao dịch cảnh báo rủi ro cao", f"{fraud_count} / {len(df_result)}")
                        
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Xuất dữ liệu đã chấm điểm dưới dạng CSV
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả dự báo (.CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
