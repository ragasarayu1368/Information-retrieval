import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from database.database import init_db, SessionLocal
from database.models import User, Document, DocumentChunk, QueryLog, AuditLog
from database.csv_db import sync_sqlalchemy_to_csv, sync_all_to_excel, load_table_df, TABLES, EXCEL_DB_FILE
from auth.authentication import authenticate_user, create_session_token, verify_session_token
from auth.authorization import can_access_page
from rag.embeddings import get_embedding_generator
from rag.faiss_store import FaissVectorStore
from rag.inverted_index import InvertedIndex
from services.user_service import UserService
from services.document_service import DocumentService
from services.audit_service import AuditService

# Page Views
from pages.dashboard import render_dashboard
from pages.app_overview import render_app_overview
from pages.assistant import render_assistant
from pages.knowledge_base import render_knowledge_base
from pages.company_documents import render_company_documents
from pages.analytics import render_analytics
from pages.evaluation import render_evaluation
from pages.users import render_users
from pages.access_control import render_access_control
from pages.audit_logs import render_audit_logs
from pages.settings import render_settings
from pages.profile import render_profile

# 1. Streamlit Page Configuration (Apple-Style Minimalism)
st.set_page_config(
    page_title="Enterprise Knowledge Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Apple Design Principles & Ultra-Clean Custom CSS
st.markdown("""
<style>
    /* Apple San Francisco & Modern Geometric Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Plus Jakarta Sans", "Helvetica Neue", sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.03em;
        font-weight: 700;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Clean Canvas Background */
    .stApp {
        background-color: #F8FAFC !important;
        color: #000000 !important;
    }
    .stApp p, .stApp span, .stApp div, .stApp label {
        color: #000000;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
        max-width: 1320px;
    }

    /* Apple-Style Frosted Cards */
    .apple-card {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04), 0 2px 6px -1px rgba(15, 23, 42, 0.02);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        color: #000000 !important;
    }
    .apple-card * {
        color: #000000;
    }
    .apple-card:hover {
        border-color: #94A3B8 !important;
        box-shadow: 0 12px 32px -4px rgba(15, 23, 42, 0.08);
        transform: translateY(-2px);
    }

    /* Metric Widget */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        height: 100%;
    }
    .metric-card:hover {
        border-color: #94A3B8;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* Status Pills */
    .badge-active {
        background: #ECFDF5;
        color: #047857;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid #A7F3D0;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-blocked {
        background: #FEF2F2;
        color: #B91C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid #FECACA;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-role {
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid #BFDBFE;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Apple-Style Dark Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #090D16 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 10px 14px;
        border-radius: 10px;
        transition: all 0.15s ease;
        font-weight: 500;
        font-size: 0.94rem;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.06);
    /* macOS / Apple Style Sidebar Navigation Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        font-weight: 550 !important;
        padding: 9px 16px !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        text-align: left !important;
        width: 100% !important;
        font-size: 0.92rem !important;
    }
    section[data-testid="stSidebar"] .stButton > button * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        transform: translateX(3px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    /* Active Sidebar Navigation Button */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0071E3 0%, #0058B6 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-left: 4px solid #38BDF8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 9px 16px !important;
        box-shadow: 0 4px 16px rgba(0, 113, 227, 0.45) !important;
        transform: translateX(2px) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Main Content Action & Suggested Buttons (Clean White Frosted Style) */
    .main .stButton > button {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 8px 14px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .main .stButton > button:hover {
        background: #F8FAFC !important;
        border-color: #0071E3 !important;
        color: #0071E3 !important;
        box-shadow: 0 4px 14px rgba(0, 113, 227, 0.12) !important;
        transform: translateY(-1px) !important;
    }

    /* Custom Chat Containers (Crisp Pure Black Text & High Contrast) */
    .chat-user {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-right: 4px solid #0071E3 !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 16px 22px !important;
        margin-bottom: 14px !important;
        margin-left: 15% !important;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04) !important;
        color: #000000 !important;
    }
    .chat-user div, .chat-user p, .chat-user span {
        color: #000000 !important;
    }
    .chat-ai {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 4px solid #4F46E5 !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 20px 24px !important;
        margin-bottom: 18px !important;
        margin-right: 5% !important;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05) !important;
        color: #000000 !important;
    }
    .chat-ai div, .chat-ai p, .chat-ai span {
        color: #000000 !important;
    }
    .evidence-box {
        background: #F8FAFC !important;
        color: #000000 !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #0071E3 !important;
        padding: 12px 16px !important;
        border-radius: 0 10px 10px 0 !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
    }
    .evidence-box * {
        color: #000000 !important;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Initialize Shared Singletons & Database
@st.cache_resource
def initialize_system_resources():
    init_db()
    faiss_store = FaissVectorStore()
    inverted_index = InvertedIndex()
    embedding_gen = get_embedding_generator()
    
    # Check if database needs initial demo seeding
    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        if user_count == 0:
            UserService.seed_demo_users(db)
            admin_u = UserService.get_user_by_email(db, "admin@example.com")
            DocumentService.seed_sample_documents(db, admin_u, faiss_store, inverted_index, embedding_gen)
            sync_sqlalchemy_to_csv(db)
    finally:
        db.close()

    return faiss_store, inverted_index, embedding_gen

faiss_store, inverted_index, embedding_gen = initialize_system_resources()

# 4. Session State Management
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

db = SessionLocal()

def get_current_user() -> User:
    if st.session_state.user_id:
        return UserService.get_user_by_id(db, st.session_state.user_id)
    return None

# ==============================================================================
# 5. SIGN IN / MULTIPLE ACCOUNT CREATION (Apple Design)
# ==============================================================================
if not st.session_state.authenticated:
    _, auth_center, _ = st.columns([1, 2.3, 1])

    with auth_center:
        st.markdown("""
            <div style="text-align: center; margin-top: 24px; margin-bottom: 24px;">
                <div style="font-size: 3.4rem; margin-bottom: 8px; filter: drop-shadow(0 6px 14px rgba(0, 113, 227, 0.3));">🧠</div>
                <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.03em;">
                    Enterprise Knowledge AI
                </h1>
                <p style="font-size: 0.95rem; color: #64748B; margin-top: 4px;">
                    Zero-Trust Permission-Aware Hybrid Intelligence Assistant
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Tab Segmented Control: Sign In vs Create Account
        tab_signin, tab_register = st.tabs(["🔑 Sign In", "👤 Create New Account"])

        # ----------------------------------------------------------------------
        # Tab 1: Sign In
        # ----------------------------------------------------------------------
        with tab_signin:
            with st.container():
                st.markdown("""
                    <div class="apple-card" style="margin-top: 12px;">
                        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.25rem; font-weight: 700; text-align: center; margin-bottom: 18px;">
                            Sign In to Enterprise Workspace
                        </h3>
                """, unsafe_allow_html=True)

                login_email = st.text_input("Corporate Email", placeholder="employee@example.com", key="login_email_input")
                login_password = st.text_input("Password", type="password", placeholder="••••••••••••", key="login_pwd_input")
                
                c_rem, c_forgot = st.columns([1, 1])
                with c_rem:
                    remember_me = st.checkbox("Remember Me", value=True, key="rem_me_cb")
                with c_forgot:
                    st.markdown("<div style='text-align: right; font-size: 0.82rem; color: #64748B; padding-top: 4px;'>Forgot password? Contact IT</div>", unsafe_allow_html=True)

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                login_btn = st.button("SIGN IN", type="primary", use_container_width=True, key="btn_signin_submit")

                if login_btn:
                    user = authenticate_user(db, login_email, login_password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user.id
                        st.session_state.token = create_session_token(user, remember_me)
                        AuditService.log_event(
                            db=db,
                            user_email=user.email,
                            user_role=user.role,
                            action="LOGIN",
                            status="SUCCESS",
                            details="Authenticated via credentials"
                        )
                        st.rerun()
                    else:
                        AuditService.log_event(
                            db=db,
                            user_email=login_email,
                            user_role="UNKNOWN",
                            action="FAILED_LOGIN",
                            status="FAILED",
                            details="Invalid credentials supplied"
                        )
                        st.error("Invalid email or password. Please verify credentials.")

                st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Tab 2: Multiple Account Creation / Registration Flow
        # ----------------------------------------------------------------------
        with tab_register:
            with st.container():
                st.markdown("""
                    <div class="apple-card" style="margin-top: 12px;">
                        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.25rem; font-weight: 700; text-align: center; margin-bottom: 18px;">
                            Create Enterprise Account
                        </h3>
                """, unsafe_allow_html=True)

                reg_name = st.text_input("Full Name", placeholder="e.g. Elena Rostova", key="reg_name_input")
                reg_email = st.text_input("Corporate Email", placeholder="e.g. elena@example.com", key="reg_email_input")
                
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    reg_dept = st.selectbox("Department", ["Engineering", "HR", "IT", "Finance", "Legal", "Executive", "General"], index=0, key="reg_dept_sel")
                with r_col2:
                    reg_role = st.selectbox("Account Role", ["EMPLOYEE", "MANAGER", "ADMIN"], index=0, key="reg_role_sel")

                reg_pwd = st.text_input("Create Password", type="password", placeholder="Minimum 6 characters", key="reg_pwd_input")
                reg_pwd_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_pwd_conf_input")

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                reg_btn = st.button("CREATE ACCOUNT & SIGN IN", type="primary", use_container_width=True, key="btn_register_submit")

                if reg_btn:
                    if not reg_name or not reg_email or not reg_pwd:
                        st.error("Please fill in all required fields.")
                    elif "@" not in reg_email or "." not in reg_email:
                        st.error("Please enter a valid corporate email address.")
                    elif len(reg_pwd) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif reg_pwd != reg_pwd_confirm:
                        st.error("Password confirmation does not match.")
                    else:
                        new_user, err = UserService.create_user(
                            db=db,
                            name=reg_name,
                            email=reg_email,
                            password=reg_pwd,
                            role=reg_role,
                            department=reg_dept
                        )
                        if err:
                            st.error(err)
                        else:
                            AuditService.log_event(
                                db=db,
                                user_email=new_user.email,
                                user_role=new_user.role,
                                action="ACCOUNT_SELF_REGISTERED",
                                status="SUCCESS",
                                details=f"Created {new_user.role} account in {new_user.department}"
                            )
                            # Instantly sign in
                            st.session_state.authenticated = True
                            st.session_state.user_id = new_user.id
                            st.session_state.token = create_session_token(new_user, True)
                            st.success(f"Welcome, {new_user.name}! Your account was created successfully.")
                            st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

        # Quick 1-Click Demo Login Bar for Presentation
        st.markdown("""
            <div style="text-align: center; margin-bottom: 12px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em;">
                    ⚡ 1-CLICK DEMO ACCOUNTS
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        demo_col1, demo_col2, demo_col3 = st.columns(3)

        with demo_col1:
            if st.button("👤 Employee\n(Sarah - Engineering)", use_container_width=True, key="demo_btn_emp"):
                emp_u = UserService.get_user_by_email(db, "employee@example.com")
                if emp_u:
                    st.session_state.authenticated = True
                    st.session_state.user_id = emp_u.id
                    AuditService.log_event(db, emp_u.email, emp_u.role, "LOGIN", status="SUCCESS", details="Demo 1-Click Employee Login")
                    st.rerun()

        with demo_col2:
            if st.button("👔 Manager\n(Marcus - Eng Mgr)", use_container_width=True, key="demo_btn_mgr"):
                mgr_u = UserService.get_user_by_email(db, "manager@example.com")
                if mgr_u:
                    st.session_state.authenticated = True
                    st.session_state.user_id = mgr_u.id
                    AuditService.log_event(db, mgr_u.email, mgr_u.role, "LOGIN", status="SUCCESS", details="Demo 1-Click Manager Login")
                    st.rerun()

        with demo_col3:
            if st.button("👑 Admin\n(Alexander - Exec)", use_container_width=True, key="demo_btn_adm"):
                adm_u = UserService.get_user_by_email(db, "admin@example.com")
                if adm_u:
                    st.session_state.authenticated = True
                    st.session_state.user_id = adm_u.id
                    AuditService.log_event(db, adm_u.email, adm_u.role, "LOGIN", status="SUCCESS", details="Demo 1-Click Admin Login")
                    st.rerun()

    db.close()
    st.stop()

# ==============================================================================
# 6. AUTHENTICATED APP WORKSPACE & SIDEBAR
# ==============================================================================
current_user = get_current_user()
if not current_user:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.rerun()

# Build Navigation items permitted for current user role
all_nav_items = [
    {"label": "🏠 Dashboard", "key": "dashboard", "roles": ["EMPLOYEE", "MANAGER", "ADMIN"]},
    {"label": "ℹ️ App", "key": "app", "roles": ["EMPLOYEE", "MANAGER", "ADMIN"]},
    {"label": "💬 Assistant", "key": "assistant", "roles": ["EMPLOYEE", "MANAGER", "ADMIN"]},
    {"label": "📚 Company Documents", "key": "company_documents", "roles": ["EMPLOYEE", "MANAGER", "ADMIN"]},
    {"label": "📊 Analytics", "key": "analytics", "roles": ["MANAGER", "ADMIN"]},
    {"label": "🧪 Evaluation", "key": "evaluation", "roles": ["ADMIN"]},
    {"label": "👥 Users", "key": "users", "roles": ["ADMIN"]},
    {"label": "🔐 Access Control", "key": "access_control", "roles": ["ADMIN"]},
    {"label": "📝 Audit Logs", "key": "audit_logs", "roles": ["ADMIN"]},
    {"label": "⚙️ Settings", "key": "settings", "roles": ["ADMIN"]},
    {"label": "👤 Profile", "key": "profile", "roles": ["EMPLOYEE", "MANAGER", "ADMIN"]}
]

allowed_nav_items = [item for item in all_nav_items if current_user.role.upper() in item["roles"]]

# Render Sidebar
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 22px; padding-bottom: 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2.1rem;">🧠</span>
                <div>
                    <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; font-family: 'Outfit', sans-serif;">ENTERPRISE AI</div>
                    <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Zero-Trust RAG</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Current User Identity Card
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 14px; margin-bottom: 22px;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Authenticated User</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 1.02rem; margin-top: 2px;">👤 {current_user.name}</div>
            <div style="display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap;">
                <span class="badge-role" style="font-size: 0.68rem; padding: 2px 8px;">{current_user.role}</span>
                <span style="background: rgba(255, 255, 255, 0.12); color: #E2E8F0; padding: 2px 8px; border-radius: 9999px; font-size: 0.68rem; font-weight: 600;">{current_user.department}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Menu Buttons (Apple Sidebar Layout)
    st.markdown("<div style='font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;'>Workspace Navigation</div>", unsafe_allow_html=True)
    
    current_key = st.session_state.current_page.lower().replace(" ", "_")
    
    for item in allowed_nav_items:
        is_active = (item["key"] == current_key)
        btn_label = f"▸ {item['label']}" if is_active else f"  {item['label']}"
        
        # Primary style for active page, regular for others
        btn_kind = "primary" if is_active else "secondary"
        if st.button(btn_label, key=f"nav_btn_{item['key']}", use_container_width=True):
            if st.session_state.current_page != item["key"]:
                st.session_state.current_page = item["key"]
                st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 📊 Live Data & Table Inspector in Sidebar
    with st.expander("🔍 Inspect Live Data & CSV", expanded=False):
        u_cnt = db.query(User).count()
        d_cnt = db.query(Document).count()
        q_cnt = db.query(QueryLog).count()
        
        st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.06); padding: 10px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 12px; font-size: 0.8rem; line-height: 1.4;">
                <div>👥 <b>{u_cnt}</b> Users Registered</div>
                <div>📚 <b>{d_cnt}</b> Documents Indexed</div>
                <div>💬 <b>{q_cnt}</b> Questions Answered</div>
            </div>
        """, unsafe_allow_html=True)

        sel_inspect_tbl = st.selectbox("Select Table", TABLES, index=0, key="sb_data_inspect_tbl")
        df_inspect = load_table_df(sel_inspect_tbl)
        
        if not df_inspect.empty:
            st.caption(f"`{sel_inspect_tbl}.csv` ({len(df_inspect)} rows)")
            st.dataframe(df_inspect.head(8), use_container_width=True, height=160)
            
            # 1-Click CSV Download
            st.download_button(
                label=f"📥 Download {sel_inspect_tbl}.csv",
                data=df_inspect.to_csv(index=False).encode("utf-8"),
                file_name=f"{sel_inspect_tbl}.csv",
                mime="text/csv",
                key=f"sb_dl_btn_{sel_inspect_tbl}"
            )
        else:
            st.caption(f"No records found in `{sel_inspect_tbl}.csv`")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Quick Demo Initializer / Reset Button
    if st.button("🚀 Re-seed Demo Knowledge", key="sidebar_demo_seed_btn", use_container_width=True):
        with st.spinner("Re-seeding demo users and knowledge base..."):
            UserService.seed_demo_users(db)
            admin_u = UserService.get_user_by_email(db, "admin@example.com")
            DocumentService.seed_sample_documents(db, admin_u, faiss_store, inverted_index, embedding_gen)
            sync_sqlalchemy_to_csv(db)
            AuditService.log_event(db, current_user.email, current_user.role, "DEMO_RESEED", status="SUCCESS", details="Seeded standard enterprise demo data")
            st.success("Sample enterprise data & CSV database synchronized!")
            st.rerun()

    # Logout Button at bottom
    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.1); margin: 20px 0;'>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", key="logout_btn", use_container_width=True):
        AuditService.log_event(db, current_user.email, current_user.role, "LOGOUT", status="SUCCESS")
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.rerun()

# ==============================================================================
# 7. MAIN CONTENT ROUTER & PERMISSION ENFORCEMENT
# ==============================================================================
page = st.session_state.current_page.lower().replace(" ", "_")

# Strict Role Permission Verification
if not can_access_page(page, current_user.role):
    AuditService.log_event(
        db=db,
        user_email=current_user.email,
        user_role=current_user.role,
        action="ACCESS_DENIED",
        resource=f"Page:{page}",
        status="BLOCKED",
        details="Attempted unauthorized page navigation"
    )
    st.markdown("""
        <div style="background: #FEF2F2; border: 1px solid #F87171; border-radius: 14px; padding: 28px; text-align: center; margin-top: 40px; box-shadow: 0 8px 24px rgba(239, 68, 68, 0.08);">
            <div style="font-size: 3rem; margin-bottom: 8px;">🔒</div>
            <h2 style="color: #DC2626; margin: 0 0 8px 0; font-weight: 800;">Access Restricted</h2>
            <p style="color: #7F1D1D; font-size: 1.05rem; margin: 0;">
                You do not have sufficient clearance to access this resource. Please contact your enterprise administrator.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    if page == "dashboard":
        render_dashboard(current_user, db, faiss_store, inverted_index)
    elif page == "app":
        render_app_overview(current_user, db, faiss_store, inverted_index)
    elif page == "assistant":
        render_assistant(current_user, db, faiss_store, inverted_index, embedding_gen)
    elif page in ["company_documents", "knowledge_base"]:
        render_company_documents(current_user, db, faiss_store, inverted_index, embedding_gen)
    elif page == "analytics":
        render_analytics(current_user, db)
    elif page == "evaluation":
        render_evaluation(current_user, db, faiss_store, inverted_index, embedding_gen)
    elif page == "users":
        render_users(current_user, db)
    elif page == "access_control":
        render_access_control(current_user, db)
    elif page == "audit_logs":
        render_audit_logs(current_user, db)
    elif page == "settings":
        render_settings(current_user, db, faiss_store, inverted_index, embedding_gen)
    elif page == "profile":
        render_profile(current_user, db)

db.close()
