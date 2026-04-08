import streamlit as st
# from QAGraph import QAGraph
import json
import tempfile
import pandas as pd
import io
from qa_agent.pdf_graph_agent import PDFGraph

st.set_page_config(layout="wide")

@st.fragment()
def downloaders():
    """Clean Excel download functionality using friend's solution"""
    raw = st.session_state.get("test_list_data", [])
    if not raw:
        st.info("No test-case data available yet.")
        return

    st.markdown("---")
    st.markdown("### 📊 Download Test Cases")
    
    # Display summary
    test_count = len(raw)
    st.info(f"Generated {test_count} test cases ready for download")

    # 1️⃣  Flatten nested 'scenario' dict -> columns like scenario.id, scenario.scenario …
    df = pd.json_normalize(raw)

    # (optional) add an empty 'Result' column the QE team can fill later
    df["Result"] = ""

    # 2️⃣  Write to Excel in-memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="TestCases")
    buffer.seek(0)  # rewind

    # 3️⃣  Download button
    st.download_button(
        label="📥 Download Test Cases (Excel)",
        data=buffer.getvalue(),
        file_name="test_cases.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_test_cases",
    )
    
    st.success("✅ Excel file is ready for download!")

# Initialize session_state variables
if 'design_documents' not in st.session_state:
    st.session_state.design_documents = []

if 'api_document' not in st.session_state:
    st.session_state.api_document = None

if 'pdf_graph_with_memory' not in st.session_state:     
    pdf_graph = PDFGraph()
    st.session_state['pdf_graph_with_memory'] = pdf_graph.get_memory_graph()

st.title('Document Upload and Processing')

# Upload Design Documents
st.write('### Upload Design Documents')
design_docs = st.file_uploader('Upload Design Documents', accept_multiple_files=True, key='design_docs_uploader')
if design_docs:
    st.session_state.design_documents = design_docs

# Upload API Document
st.write('### Upload API Document')
api_doc = st.file_uploader('Upload API Document', key='api_doc_uploader')
if api_doc:
    st.session_state.api_document = api_doc

# Print uploaded design documents
st.write('Uploaded Design Documents:')
for doc in st.session_state.design_documents:
    st.write(f'- {doc.name}')

# Print uploaded API document
st.write('Uploaded API Document:')
if st.session_state.api_document:
    st.write(f'- {st.session_state.api_document.name}')

# Process Button
if st.session_state.design_documents or st.session_state.api_document:
    if st.button('Process Documents'):
        st.write('Processing documents...')
        # Access the PDF graph with memory
        pdf_graph_with_memory = st.session_state['pdf_graph_with_memory']
        
        pdf_graph = PDFGraph()
        
        try:
            import tempfile
            import os
            
            print("💾 [App] Starting file saving process...")
            
            # Check if design documents are uploaded
            if st.session_state.design_documents:
                design_file_paths = []
                for doc in st.session_state.design_documents:
                    design_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc.name}")
                    design_temp_file.write(doc.read())
                    design_temp_file.close()
                    design_file_paths.append(design_temp_file.name)
                    temp_files = []  # Keep track for cleanup
                    temp_files.append(design_temp_file.name)
                    print(f"✅ [App] Saved design document to: {design_temp_file.name}")
                    st.write(f"✅ Saved design document: {doc.name}")
            else:
                design_file_paths = []
                print("⚠️ [App] No design documents uploaded.")

            # Check if API document is uploaded
            if st.session_state.api_document:
                api_doc_name = st.session_state.api_document.name
                print(f"⚙️ [App] Processing API document: {api_doc_name}")
                api_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{api_doc_name}")
                api_temp_file.write(st.session_state.api_document.read())
                api_temp_file.close()
                api_file_path = api_temp_file.name
                temp_files.append(api_temp_file.name)
                print(f"✅ [App] Saved API document to: {api_temp_file.name}")
                st.write(f"✅ Saved API document: {api_doc_name}")
            else:
                api_file_path = None
                print("⚠️ [App] No API document uploaded.")

            # Combine all file paths
            all_file_paths = design_file_paths + ([api_file_path] if api_file_path else [])
            print(f"📊 [App] Total files saved: {len(all_file_paths)}")
            print(f"📋 [App] File paths: {all_file_paths}")

            st.write(f"📁 Successfully saved {len(all_file_paths)} files for processing")
            
        except Exception as e:
            st.error(f"❌ Error saving files: {e}")
            # Continue without file paths - set empty defaults
            design_doc_names = [doc.name for doc in st.session_state.design_documents]
            api_doc_name = "No API document uploaded"
            all_file_paths = []
            temp_files = []
            st.warning("Files could not be saved - assistant will not have access to document content")
        
        # Prepare inputs for the graph
        design_file_paths_for_graph = all_file_paths[:-1] if len(all_file_paths) > 1 else []
        api_file_path_for_graph = all_file_paths[-1] if all_file_paths else ''
        
        print(f"🎯 [App] Preparing graph inputs...")
        print(f"📁 [App] design_file_paths_for_graph: {design_file_paths_for_graph}")
        print(f"📄 [App] api_file_path_for_graph: {api_file_path_for_graph}")
        
        inputs = {
            'input': '',
            'target_app': 'PDF Processing',
            'design_documents': [doc.name for doc in st.session_state.design_documents],  # Pass file names
            'api_document': st.session_state.api_document.name if st.session_state.api_document else '',  # Pass file name
            'design_file_paths': design_file_paths_for_graph,  # Pass actual file paths for design docs
            'api_file_path': api_file_path_for_graph,  # Pass actual file path for API doc
            'temp_files': temp_files,  # Keep track for cleanup
            'message_history': [],
            'test_list': [],
            'is_scenario_list_processed': False,
            'scenario_list': [],
            'current_scenario': (0, {}),
            'current_test': (0, ''),
            'current_test_details': [],
            'test_details_list': [],
            'is_test_list_processed': False,
            'question': '',
            'stage1_thread_id': '',
            'stage1_revisions': 0,
            'is_finished_stage1': False,
            'stage2_thread_id': '',
            'stage2_revisions': 0,
            'is_finished_stage2': False,
            'stage3_thread_id': '',
            'stage3_revisions': 0,
            'is_finished_stage3': False,
            'processed_scenarios': []
        }
        
        # Thread configuration for memory
        thread_config = {"configurable": {"thread_id": 1}}
        
        # Stream processing logic
        with st.status("Processing documents...") as status:
            step_count = 0
            total_steps = 4  # assist_stage1, reflect_stage1, assist_stage2
            
            for output in pdf_graph_with_memory.stream(inputs, thread_config):
                step_count += 1
                current_node = list(output.keys())[0]
                
                # Map node names to user-friendly descriptions
                node_descriptions = {
                    'assist_stage1': 'Analyzing documents and generating scenarios',
                    'reflect_stage1': 'Validating and refining scenarios', 
                    'assist_stage2': 'Creating detailed test cases'
                }
                
                description = node_descriptions.get(current_node, current_node)
                progress_text = f"Step {step_count}/{total_steps}: {description}"
                
                st.write(f"🔄 {progress_text}")
                status.update(label=progress_text, expanded=True)
                
                # Display the current state information
                current_state = output[current_node]
                if current_node == 'assist_stage1':
                    scenarios = current_state.get('scenario_list', [])
                    st.write(f"✅ Generated {len(scenarios)} test scenarios")
                elif current_node == 'reflect_stage1':
                    validated = current_state.get('scenario_list', [])
                    st.write(f"✅ Validated {len(validated)} scenarios")
                elif current_node == 'assist_stage2':
                    test_cases = current_state.get('test_details_list', [])
                    st.write(f"✅ Created {len(test_cases)} detailed test cases")
                
                # Show detailed output in expandable section
                with st.expander(f"View {current_node} details"):
                    st.json(output)
            
            # Mark as complete
            status.update(label="✅ Processing completed successfully!", state="complete")
            st.success("🎉 Document processing completed! Test scenarios and cases have been generated.")
            
            # Add Excel download functionality if test cases are available
            if current_state and current_state.get('test_details_list'):
                st.session_state['test_list_data'] = current_state.get('test_details_list', [])
                
                # Call the Excel download function
                downloaders()
