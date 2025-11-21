"""
Streamlit app for testing Pinecone vector database search
Interactive UI for CDS code normalization testing
"""

import streamlit as st
import os
import sys
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Page configuration
st.set_page_config(
    page_title="CDS Normalization Tester",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 10px;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .excellent { border-left: 5px solid #28a745; }
    .good { border-left: 5px solid #ffc107; }
    .fair { border-left: 5px solid #fd7e14; }
    .poor { border-left: 5px solid #dc3545; }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = "cds-codes-embeddings"

# Mappings
HEADER_MAPPINGS = [
    {"type": "LocationCodes", "paths": [["Goods Location"]]},
    {"type": "DeliveryTerm", "paths": [["Delivery Terms"]]},
    {"type": "ObligationGuarantee", "paths": [["Guarantees", "type"]]},
    {"type": "NatureOfTransaction", "paths": [["Nature of Transaction"]]},
    {"type": "AuthorisationCode", "paths": [["Authorisation Holders", "code"]]},
    {"type": "ImportPreviousDocuments", "paths": [["Previous Documents", "type"]]},
    {"type": "DeclarationType", "paths": [["Declaration Type"]]},
    {"type": "AdditionalInformation", "paths": [["Additional Information", "code"]]},
    {"type": "WarehouseType", "paths": [["Warehouse"]]},
    {"type": "AdditionCode", "paths": [["Additions and Deductions", "code"]]},
    {"type": "DeductionCode", "paths": [["Additions and Deductions", "code"]]},
    {
        "type": "CurrencyCode",
        "paths": [
            ["Invoice Currency"],
            ["Additions and Deductions", "currency"]
        ]
    },
    {"type": "SupplyChainActorRoles", "paths": [["Additional Supply Chain Actors", "role"]]},
    {
        "type": "BorderTransportMode",
        "paths": [
            ["Border Transport Mode"],
            ["Inland Transport Type"]
        ]
    },
    {
        "type": "IdentityMeansOfTransport",
        "paths": [
            ["Arrival Transport Type"],
            ["Border Transport Type"]
        ]
    },
    {
        "type": "OfficeCodes",
        "paths": [
            ["Presentation Office"],
            ["Supervising Office"]
        ]
    },
    {"type": "FiscalReferenceRoles", "paths": [["Additional Fiscal Reference", "role"]]},
    {
        "type": "CountryCode",
        "paths": [
            ["Exporter", "country"],
            ["Importer", "country"],
            ["Declarant", "country"],
            ["Border Transport Country"],
            ["Representative", "country"],
            ["Seller", "country"],
            ["Buyer", "country"],
            ["Consignee", "country"],
            ["Consignor", "country"],
            ["Dispatch Country"],
            ["Destination Country"]
        ]
    },
]

ITEMS_MAPPING = [
    {"type": "NatureOfTransaction", "paths": [["Nature of Transaction"]]},
    {"type": "ImportPreviousDocuments", "paths": [["Previous Documents", "type"]]},
    {"type": "AdditionalInformation", "paths": [["Additional Information", "code"]]},
    {"type": "ProcedureCodes", "paths": [["Procedure Code"]]},
    {"type": "AdditionCode", "paths": [["Additions and Deductions", "code"]]},
    {"type": "DeductionCode", "paths": [["Additions and Deductions", "code"]]},
    {"type": "AdditionalDocuments", "paths": [["Documents and Certificates", "type"]]},
    {"type": "CurrencyCode", "paths": [["Additions and Deductions", "currency"]]},
    {"type": "SupplyChainActorRoles", "paths": [["Additional Supply Chain Actors", "role"]]},
    {"type": "PackageTypes", "paths": [["Packages", "kind"]]},
    {"type": "ValuationMethod", "paths": [["Valuation Method"]]},
    {"type": "FiscalReferenceRoles", "paths": [["Additional Fiscal Reference", "role"]]},
    {
        "type": "CountryCode",
        "paths": [
            ["Exporter", "country"],
            ["Seller", "country"],
            ["Buyer", "country"],
            ["Consignee", "country"],
            ["Consignor", "country"],
            ["Dispatch Country"],
            ["Destination Country"],
            ["Origin Country"],
            ["Country of Preferential Origin"]
        ]
    },
    {"type": "AdditionalProcedure", "paths": [["Additional Procedures", "code"]]},
]


@st.cache_data
def build_field_mapping(mappings):
    """Build a mapping from field paths to code types"""
    field_map = {}
    field_options = []
    
    for mapping in mappings:
        code_type = mapping["type"]
        for path in mapping["paths"]:
            if len(path) == 1:
                key = path[0]
                display = f"🔹 {key}"
                field_map[display] = {
                    "type": code_type,
                    "path": path,
                    "is_nested": False,
                    "display": key,
                    "key": key
                }
                field_options.append(display)
            else:
                dotted_key = ".".join(path)
                display = f"🔸 {path[0]} → {path[1]}"
                field_map[display] = {
                    "type": code_type,
                    "path": path,
                    "is_nested": True,
                    "display": f"{path[0]} → {path[1]}",
                    "parent": path[0],
                    "subfield": path[1],
                    "key": dotted_key
                }
                field_options.append(display)
    
    return field_map, sorted(field_options)


@st.cache_resource
def initialize_clients():
    """Initialize OpenAI and Pinecone clients"""
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return None, None
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    
    return openai_client, index


def create_embedding(openai_client, text):
    """Generate embedding for text"""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def search_vectors(index, embedding, code_type, top_k):
    """Query Pinecone index"""
    results = index.query(
        vector=embedding,
        top_k=top_k,
        filter={"type": {"$eq": code_type}},
        include_metadata=True
    )
    return results


def get_quality_badge(score):
    if score >= 0.8:
        return "🟢 EXCELLENT", "excellent", "#28a745"
    elif score >= 0.7:
        return "🟡 GOOD", "good", "#ffc107"
    elif score >= 0.6:
        return "🟠 FAIR", "fair", "#fd7e14"
    else:
        return "🔴 POOR", "poor", "#dc3545"


def display_result_card(rank, match):
    score = match.score
    code = match.metadata.get('code', 'N/A')
    short_desc = match.metadata.get('shortDescription', 'N/A')
    long_desc = match.metadata.get('longDescription', 'N/A')
    
    badge, quality_class, color = get_quality_badge(score)
    
    with st.container():
        st.markdown(f"""
        <div class="result-card {quality_class}">
            <h3>Rank #{rank} {badge}</h3>
            <hr style="margin: 0.5rem 0;">
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 3])
        
        with col1:
            st.metric("Code", code)
        
        with col2:
            st.metric("Similarity", f"{score*100:.2f}%")
        
        with col3:
            st.metric("Score", f"{score:.4f}")
        
        st.markdown(f"**Short Description:**")
        st.info(short_desc)
        
        with st.expander("📄 View Full Description"):
            st.write(long_desc)
        
        st.markdown("---")


def display_comparison_chart(matches):
    if not matches:
        return
    
    data = []
    for i, match in enumerate(matches, 1):
        data.append({
            'Rank': f"#{i}",
            'Code': match.metadata.get('code', 'N/A'),
            'Similarity': match.score * 100,
            'Score': match.score
        })
    
    df = pd.DataFrame(data)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Code', 
        y='Similarity',
        color='Similarity',
        color_continuous_scale=['#dc3545', '#fd7e14', '#ffc107', '#28a745'],
        title='Similarity Comparison',
        labels={'Similarity': 'Similarity Score (%)'},
        text='Similarity'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.markdown('<div class="main-header">🔍 CDS Vector Search Tester</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">QA testing for embedding-based normalization</p>', unsafe_allow_html=True)
    
    # Initialize clients
    openai_client, pinecone_index = initialize_clients()
    
    if not openai_client or not pinecone_index:
        st.error("⚠️ Missing API keys! Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/search.png", width=80)
        st.title("⚙️ Configuration")
        
        # Context selection
        st.markdown("### 📋 Context")
        context = st.radio(
            "Select field context:",
            ["Header Fields", "Items Fields"],
            index=0,
            help="Choose whether to search in header-level or item-level fields"
        )
        
        is_header = context == "Header Fields"
        mappings = HEADER_MAPPINGS if is_header else ITEMS_MAPPING
        field_map, field_options = build_field_mapping(mappings)
        
        st.markdown("### 🎯 Search Parameters")
        
        # Top K selection
        top_k = st.slider(
            "Number of results:",
            min_value=1,
            max_value=10,
            value=3,
            help="How many top matches to return"
        )
        
        # Similarity threshold
        threshold = st.slider(
            "Minimum similarity threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Filter results below this similarity score"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Fields", len(field_options))
        st.metric("Context", "Header" if is_header else "Items")
        
        # Info
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This app tests the Pinecone vector database used for CDS code normalization.
        
        **How it works:**
        1. Select a field from your context
        2. Enter a value to search
        3. Get top matching codes with similarity scores
        """)
    
    # Main content
    # tab1, tab2, tab3 = st.tabs(["🔍 Search", "📚 Field Reference", "🧪 Batch Test"])
    tab1, tab2 = st.tabs(["🔍 Search", "📚 Field Reference"])
    
    
    with tab1:
        st.markdown('<div class="section-header">Interactive Search</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Field selection
            st.markdown("#### 1️⃣ Select Field")
            selected_field = st.selectbox(
                "Choose a field to test:",
                field_options,
                help="Select from available fields in the chosen context"
            )
            
            field_info = field_map[selected_field]
            
            # Display field info
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**Field:** {field_info['display']}")
            st.markdown(f"**Type:** `{field_info['type']}`")
            st.markdown(f"**Nested:** {'Yes' if field_info['is_nested'] else 'No'}")
            if field_info['is_nested']:
                st.markdown(f"**Parent:** {field_info['parent']}")
                st.markdown(f"**Subfield:** {field_info['subfield']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Value input
            st.markdown("#### 2️⃣ Enter Value")
            
            # Provide examples based on field type
            examples = {
                "AdditionalDocuments": "commercial invoice, packing list, certificate of origin",
                "CountryCode": "United States, Germany, China",
                "PackageTypes": "cardboard boxes, wooden crates, pallets",
                "BorderTransportMode": "air transport, sea freight, road transport",
                "DeliveryTerm": "FOB, CIF, DDP"
            }
            
            example_text = examples.get(field_info['type'], "Enter the value to search for")
            
            field_value = st.text_input(
                "Field value:",
                placeholder=example_text,
                help="Enter the text value you want to normalize"
            )
            
            # Search button
            search_button = st.button("🚀 Search", type="primary", use_container_width=True)
        
        # Execute search
        if search_button:
            if not field_value:
                st.warning("⚠️ Please enter a field value to search.")
            else:
                with st.spinner("🔄 Searching vector database..."):
                    # Generate embedding
                    embedding = create_embedding(openai_client, field_value)
                    
                    # Query Pinecone
                    results = search_vectors(
                        pinecone_index,
                        embedding,
                        field_info['type'],
                        top_k
                    )
                    
                    # Store in session state
                    st.session_state['last_results'] = results
                    st.session_state['last_field'] = field_info
                    st.session_state['last_value'] = field_value
        
        # Display results
        if 'last_results' in st.session_state:
            results = st.session_state['last_results']
            field_info = st.session_state['last_field']
            field_value = st.session_state['last_value']
            
            st.markdown('<div class="section-header">Search Results</div>', unsafe_allow_html=True)
            
            if not results.matches:
                st.error(f"❌ No matches found for '{field_value}' in {field_info['type']}")
            else:
                # Filter by threshold
                filtered_matches = [m for m in results.matches if m.score >= threshold]
                
                if not filtered_matches:
                    st.warning(f"⚠️ No results above {threshold:.2f} similarity threshold")
                else:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Matches", len(filtered_matches))
                    
                    with col2:
                        best_score = filtered_matches[0].score
                        st.metric("Best Score", f"{best_score*100:.2f}%")
                    
                    with col3:
                        avg_score = sum(m.score for m in filtered_matches) / len(filtered_matches)
                        st.metric("Avg Score", f"{avg_score*100:.2f}%")
                    
                    with col4:
                        best_code = filtered_matches[0].metadata.get('code', 'N/A')
                        st.metric("Top Code", best_code)
                    
                    # Comparison chart
                    # st.markdown("#### 📊 Comparison Chart")
                    # display_comparison_chart(filtered_matches)
                    
                    # Detailed results
                    st.markdown("#### 📋 Detailed Results")
                    
                    for i, match in enumerate(filtered_matches, 1):
                        display_result_card(i, match)
                    
                    # Recommendation
                    best_match = filtered_matches[0]
                    if best_match.score >= 0.7:
                        st.success(f"✅ **RECOMMENDATION:** Use code `{best_match.metadata['code']}` (confidence: {best_match.score*100:.1f}%)")
                    else:
                        st.warning(f"⚠️ **WARNING:** Low confidence ({best_match.score*100:.1f}%). Consider manual review or LLM fallback.")
    
    with tab2:
        st.markdown('<div class="section-header">Field Reference</div>', unsafe_allow_html=True)
        
        # Group fields by type
        fields_by_type = {}
        for field_option in field_options:
            field_info = field_map[field_option]
            code_type = field_info['type']
            if code_type not in fields_by_type:
                fields_by_type[code_type] = []
            fields_by_type[code_type].append(field_info)
        
        # Display as table
        st.markdown("### 📋 All Available Fields")
        
        for code_type in sorted(fields_by_type.keys()):
            with st.expander(f"🏷️ {code_type} ({len(fields_by_type[code_type])} fields)"):
                fields = fields_by_type[code_type]
                
                data = []
                for field in fields:
                    data.append({
                        'Field': field['display'],
                        'Type': 'Nested' if field['is_nested'] else 'Scalar',
                        'Path': ' → '.join(field['path'])
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
    
   


if __name__ == "__main__":
    main()