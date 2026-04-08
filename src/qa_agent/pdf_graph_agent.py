from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, TypedDict, Annotated, List
import operator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from .prompts.pdf_graph_prompts import get_advanced_tests_prompt, get_qa_reflection_prompt, get_pdf_processing_prompt
from .assistant_thread_manager import SimpleAssistantThreadManager
import json
import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

def update_testlist(left, right):
    """Custom reducer for test details list"""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right

class AutoconState(TypedDict):
    input: str
    target_app: str
    message_history: Annotated[list[BaseMessage], operator.add]
    test_list: list[tuple[int,dict]] # list of test cases
    is_scenario_list_processed: bool
    scenario_list: list[tuple[int, dict]] # list of scenarios
    current_scenario: tuple[int, dict] # current scenario
    current_test: tuple[int, str] # current test
    current_test_details: list[dict]
    test_details_list: Annotated[List[dict], update_testlist] # list of test details
    is_test_list_processed: bool
    attachments: dict
    question: str
    stage1_thread_id: str
    stage1_revisions: int
    is_finished_stage1: bool
    stage2_thread_id: str
    stage2_revisions: int
    is_finished_stage2: bool
    stage3_thread_id: str
    stage3_revisions: int
    is_finished_stage3: bool
    # PDF-specific fields
    design_documents: list
    api_document: str
    processed_scenarios: list
    # File path fields for vector store upload
    design_file_paths: list
    api_file_path: str
    temp_files: list

class PDFGraph:
    def __init__(self, assistant_id=""):
        self.assistant_manager = SimpleAssistantThreadManager(assistant_id=assistant_id)

    def get_memory_graph(self):
        builder = StateGraph(AutoconState)
        builder.add_node("assist_stage1", self._assist_stage1_node)
        builder.add_node("reflect_stage1", self._qa_reflection_stage1_node)
        builder.add_node("assist_stage2", self._assist_stage2_node)
        builder.add_node("reflect_stage2", self._qa_reflection_stage2_node)


        builder.set_entry_point("assist_stage1")
        
        builder.add_edge("assist_stage1", "reflect_stage1")
        builder.add_conditional_edges("reflect_stage1", self._should_continue_stage1_qa)
        builder.add_edge("assist_stage2", "reflect_stage2")
        # builder.add_edge("reflect_stage2", END)
        builder.add_conditional_edges("reflect_stage2", self._should_continue_stage2_qa)


        checkpointer = MemorySaver()
        lc_graph_with_memory = builder.compile(checkpointer=checkpointer)

        return lc_graph_with_memory

    def _assist_stage1_node(self, state):
        # Stage 1: Process design documents and API document using assistant
        print("🚀 [PDFGraph] Executing assist_stage1_node")
        
        # Get document names and file paths from state
        design_doc_names = state.get('design_documents', [])
        api_doc_name = state.get('api_document', '')
        design_file_paths = state.get('design_file_paths', [])
        api_file_path = state.get('api_file_path', '')
        
        print(f"📄 [PDFGraph] Processing documents: {len(design_doc_names)} design docs, 1 API doc")
        print(f"📁 [PDFGraph] File paths available: {len(design_file_paths)} design files, {'1' if api_file_path else '0'} API file")
        
        # Create user journey description based on document names
        user_journey = f"Testing and Validation of {design_doc_names or api_doc_name}"
        
        try:
            # Start thread if needed
            self.assistant_manager.start_thread()
            
            # Extract PDF content first using file paths from state
            design_file_paths = state.get('design_file_paths', [])
            api_file_path = state.get('api_file_path', '')
            
            all_file_paths = design_file_paths + ([api_file_path] if api_file_path else [])
            print(f"🔍 [PDFGraph] Extracting content from {len(all_file_paths)} PDF files...")
            print(f"📋 [PDFGraph] File paths to process: {all_file_paths}")
            
            content_result = self.assistant_manager.extract_pdf_content(all_file_paths)
            
            if content_result['status'] != 'completed':
                print(f"❌ [PDFGraph] Error extracting PDF content: {content_result.get('message', 'Unknown error')}")
                # Use fallback scenarios
                fallback_scenarios = [
                    {
                        "scenarioDescription": "Test API endpoint availability and response format",
                        "expectedResults": "API should return valid response with correct status code"
                    },
                    {
                        "scenarioDescription": "Validate authentication and authorization mechanisms", 
                        "expectedResults": "Authentication should succeed with valid credentials"
                    }
                ]
                return {**state, "test_list": fallback_scenarios, "scenario_list": fallback_scenarios}
            
            extracted_content = content_result['content']
            print(f"✅ [PDFGraph] Successfully extracted {len(extracted_content)} characters of content")
            print(f"📊 [PDFGraph] Processed {content_result['files_processed']} files")
            
            # Simple service name extraction (same logic as user_journey)
            service_name = design_doc_names[0] if design_doc_names else api_doc_name
            
            # Create special instructions with extracted content
            special_instructions = f"""
I have extracted the content from these PDF documents for analysis:
- Design Documents: {', '.join(design_doc_names)}
- API Document: {api_doc_name}

Here is the complete extracted content from all documents:

{extracted_content}

---

Based on the actual content above, please analyze the documents and generate comprehensive test scenarios for {service_name}. Focus on:
1. API endpoints and their specifications found in the content
2. Service functionality and business logic described in the documents
3. Data models and validation rules mentioned
4. Authentication and authorization requirements specified
5. Error handling and edge cases documented

Generate test scenarios based on the ACTUAL content extracted from the PDFs, not generic assumptions.
"""
            
            # Use the advanced prompt function with PDF content
            prompt_text = get_advanced_tests_prompt(user_journey, special_instructions)
            
            # Create HumanMessage with the prompt
            query_message = HumanMessage(content=prompt_text)
            
            print(f"🎯 [PDFGraph] Content ready for assistant analysis...")
            
            # Invoke assistant with extracted content (no vector store dependency)
            print(f"🤖 [PDFGraph] Calling assistant to analyze extracted document content")
            response_content = self.assistant_manager.invoke_assistant(prompt_text)
            print(f"✨ [PDFGraph] Assistant analysis completed")
            
            # Create AIMessage with the response
            out_message = AIMessage(content=response_content)
            
            # Debug: Print the full assistant response
            print(f"📝 [PDFGraph] Full assistant response: {response_content}")
            print(f"🔍 [PDFGraph] Response type: {type(response_content)}")
            print(f"📏 [PDFGraph] Response length: {len(str(response_content))}")
            
            # Try to parse JSON from response
            test_list = []
            try:
                # Look for JSON in the response
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}') + 1
                
                print(f"🔎 [PDFGraph] JSON search - start_idx: {start_idx}, end_idx: {end_idx}")
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx]
                    print(f"📄 [PDFGraph] Extracted JSON string: {json_str[:200]}...")
                    parsed_data = json.loads(json_str)
                    
                    if 'test_list' in parsed_data:
                        test_list = parsed_data['test_list']
                        print(f"🎉 [PDFGraph] Successfully parsed {len(test_list)} test scenarios from assistant response")
                    else:
                        print("⚠️ [PDFGraph] Warning: 'test_list' key not found in response")
                        print(f"🔑 [PDFGraph] Available keys: {list(parsed_data.keys())}")
                else:
                    print("⚠️ [PDFGraph] Warning: No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                print(f"❌ [PDFGraph] JSON parsing error: {e}")
                print(f"🔍 [PDFGraph] Attempted to parse: {json_str[:100] if 'json_str' in locals() else 'No JSON string extracted'}")
            
            # If no test scenarios were parsed, create fallback scenarios
            if not test_list:
                print("🔄 [PDFGraph] Creating fallback test scenarios")
                test_list = [
                    {
                        "scenarioDescription": f"Test API endpoints described in {api_doc_name}",
                        "expectedResults": "API endpoints should respond correctly according to specifications"
                    },
                    {
                        "scenarioDescription": f"Validate service functionality from {', '.join(design_doc_names[:2])}",
                        "expectedResults": "Service should function as per design specifications"
                    },
                    {
                        "scenarioDescription": "Test authentication and authorization mechanisms",
                        "expectedResults": "Authentication should work as specified in documentation"
                    }
                ]
            
            # Convert test scenarios to the expected format
            scenario_list = []
            for i, test in enumerate(test_list):
                scenario = {
                    "id": i + 1,
                    "scenario": test.get('scenarioDescription', f'Test scenario {i+1}'),
                    "expected_results": test.get('expectedResults', 'Expected result not specified'),
                    "status": "generated"
                }
                scenario_list.append((i + 1, scenario))
            
            # Update state with successful processing
            updated_state = state.copy()
            updated_state['scenario_list'] = scenario_list
            updated_state['test_list'] = scenario_list  # Also populate test_list
            # Note: Don't set is_scenario_list_processed here - let reflection decide when to finish
            updated_state['stage1_revisions'] = state.get('stage1_revisions', 0) + 1
            updated_state['message_history'] = state.get('message_history', []) + [query_message, out_message]
            updated_state['stage1_thread_id'] = self.assistant_manager.thread_id
            
            print(f"🎯 [PDFGraph] assist_stage1_node completed successfully. Generated {len(scenario_list)} scenarios")
            return updated_state
            
        except Exception as e:
            print(f"❌ [PDFGraph] Error in assist_stage1_node: {e}")
            # Return error state
            error_state = state.copy()
            error_state['stage1_revisions'] = state.get('stage1_revisions', 0) + 1
            error_state['message_history'] = state.get('message_history', []) + [
                HumanMessage(content=f"Error processing documents: {str(e)}")
            ]
            return error_state

    def _qa_reflection_stage1_node(self, state):
        # Stage 1 Reflection: Validate and refine the generated scenarios
        print("🔍 [PDFGraph] Executing qa_reflection_stage1_node")
        
        messages = state.get('message_history', [])
        scenario_list = state.get('scenario_list', [])
        stage1_revisions = state.get('stage1_revisions', 0)
        
        # Get document names from state for dynamic prompting
        design_doc_names = state.get('design_documents', [])
        api_doc_name = state.get('api_document', '')
        
        print(f"[PDFGraph] Reflection stage1 messages: {len(messages)} messages")
        print(f"[PDFGraph] Current revision count: {stage1_revisions}")
        print(f"[PDFGraph] Document context: {design_doc_names} + {api_doc_name}")
        
        # Filter last two messages from the history for reflection
        reflection_messages = messages[-2:] if len(messages) >= 2 else messages
        
        # Create reflection prompt using centralized function with document names
        reflection_prompt = get_qa_reflection_prompt(len(scenario_list), stage1_revisions, design_doc_names, api_doc_name)
        
        try:
            # Call assistant for reflection
            print("[PDFGraph] Calling assistant for reflection analysis")
            response_content = self.assistant_manager.invoke_assistant(reflection_prompt)
            print("[PDFGraph] Assistant reflection completed")
            
            # Parse the reflection response
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    reflection_data = json.loads(json_match.group())
                else:
                    # Fallback parsing
                    reflection_data = json.loads(response_content)
                
                is_finished = reflection_data.get('Finished', False)
                followup_question = reflection_data.get('follow_up_question', '')
                reasonings = reflection_data.get('reasonings', '')
                
                print(f"[PDFGraph] Reflection result - Finished: {is_finished}")
                print(f"[PDFGraph] Reasoning: {reasonings}")
                
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"[PDFGraph] JSON parsing error in reflection: {e}")
                print(f"[PDFGraph] Response content: {response_content[:200]}...")
                # Default to finished if parsing fails
                is_finished = True
                followup_question = ""
                reasonings = "Reflection parsing failed, assuming completion"
            
            # Update revision count and prepare next message
            revisions = stage1_revisions if stage1_revisions is not None else 0
            question = None
            
            if is_finished:
                # Reset revision count and mark as finished
                revisions = 0
                question = AIMessage(content="Finished generating and validating test scenario list.")
                print("[PDFGraph] Reflection determined task is complete")
            else:
                # Increment revisions and add follow-up question
                revisions = revisions + 1
                question = HumanMessage(content=followup_question)
                print(f"[PDFGraph] Reflection requesting revision #{revisions}: {followup_question}")
            
            # Prepare output state
            updated_state = state.copy()
            updated_state['message_history'] = messages + [question] if question else messages
            updated_state['stage1_revisions'] = revisions
            updated_state['is_finished_stage1'] = is_finished
            
            print(f"[PDFGraph] qa_reflection_stage1_node completed. Finished: {is_finished}, Revisions: {revisions}")
            return updated_state
            
        except Exception as e:
            print(f"[PDFGraph] Error in qa_reflection_stage1_node: {e}")
            # Default to finished on error
            updated_state = state.copy()
            updated_state['is_finished_stage1'] = True
            updated_state['stage1_revisions'] = stage1_revisions
            return updated_state

    async def run_with_timeout(self, coro, timeout):
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("Operation timed out")

    def _assist_stage2_node(self, state):
        # Stage 2: Generate detailed test cases from validated scenarios sequentially
        print("[PDFGraph] Executing assist_stage2_node with sequential processing")
        print(f"[PDFGraph] State type: {type(state)}, Keys: {list(state.keys())}")
        
        # Get test scenarios from Stage 1
        test_list = state.get('test_list', [])
        scenario_list = state.get('scenario_list', [])
        total_tests = len(test_list)
        
        print(f"[PDFGraph] Processing {total_tests} test scenarios sequentially")
        
        all_test_cases = []
        errors = []
        
        for idx, scenario in enumerate(test_list):
            scenario_id, scenario_item = scenario
            print(f"[PDFGraph] Processing scenario {scenario_id}: {scenario_item.get('scenario', 'Unknown')[:50] if isinstance(scenario_item, dict) else str(scenario_item)[:50]}...")
            
            try:
                # Create proper prompt for test case generation
                test_case_prompt = f"""Based on our previous conversation about the uploaded documents and the following test scenario, generate 2-3 comprehensive detailed test cases:

**Test Scenario:**
{scenario_item.get('scenarioDescription', scenario_item) if isinstance(scenario_item, dict) else scenario_item}

**Instructions:**
- Generate 2-3 detailed test cases for this ONE scenario
- Use the document context from our previous messages
- Each test case should be comprehensive and executable
- Focus on different aspects/edge cases of this scenario

**Output Format:**
Provide results in JSON format with exactly this structure:

```json
{{
    "test_cases": [
        {{
            "Test_Case_ID": "TC_{scenario_id:03d}_001_[logical_name]",
            "Title": "[Clear test case title]",
            "Description": "[Detailed test case description]",
            "Preconditions": "[Required setup conditions]",
            "Test_Steps": [
                "Step 1: [Detailed step]",
                "Step 2: [Detailed step]",
                "Step 3: [Detailed step]"
            ],
            "Test_Data": "[Required test data]",
            "Expected_Result": "[Expected outcome]",
            "Request_Body": {{
                "[field]": "[value based on API spec]"
            }},
            "Response": {{
                "[field]": "[expected response structure]"
            }},
            "Actual_Result": "[To be filled during execution]",
            "Status": "Not Executed",
            "Postconditions": "[System state after test]",
            "Tags": ["Service", "[relevant_tags]"],
            "Test_Type": "[Functional/API/Integration/etc]"
        }}
    ]
}}
```

**Requirements:**
- Generate at max 2-3 test cases (flexible count)
- Each test case should test different aspects of the scenario
- Use sequential numbering: TC_{scenario_id:03d}_001, TC_{scenario_id:03d}_002, etc.
- Include realistic service data and API structures based on the uploaded documentation
- Focus on comprehensive coverage of the scenario
"""
                
                # Use previous message context with proper prompt
                response_content = self.assistant_manager.invoke_assistant(test_case_prompt)
                
                # Parse JSON response
                test_cases_list = []
                try:
                    start_idx = response_content.find('{')
                    end_idx = response_content.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = response_content[start_idx:end_idx]
                        parsed_data = json.loads(json_str)
                        
                        if 'test_cases' in parsed_data and parsed_data['test_cases']:
                            test_cases_list = parsed_data['test_cases']
                            print(f"[PDFGraph] Successfully parsed {len(test_cases_list)} test cases for scenario {scenario_id}")
                        else:
                            print(f"[PDFGraph] Warning: 'test_cases' key not found for scenario {scenario_id}")
                            
                except json.JSONDecodeError as e:
                    print(f"[PDFGraph] JSON parsing error for scenario {scenario_id}: {e}")
                
                # Add scenario reference to all test cases
                for test_case in test_cases_list:
                    test_case['scenario'] = scenario_item
                    test_case['scenario_id'] = scenario_id
                
                all_test_cases.extend(test_cases_list)

            except TimeoutError:
                print(f"[PDFGraph] Timeout processing scenario {scenario_id}, moving to next")
                errors.append(f"Scenario {scenario_id}: Timeout")
            except Exception as e:
                print(f"[PDFGraph] Error processing scenario {scenario_id}: {e}")
                errors.append(f"Scenario {scenario_id}: {str(e)}")
        
        print(f"🎯 [PDFGraph] Sequential processing completed. Generated {len(all_test_cases)} detailed test cases")
        if errors:
            print(f"📄 [PDFGraph] Errors: {errors[:3]}...")  # Show first 3 errors
        
        # Update state with all generated test cases
        updated_state = state.copy()
        updated_state['test_details_list'] = all_test_cases
        updated_state['is_test_list_processed'] = True
        updated_state['is_finished_stage2'] = True
        
        # Update message history with summary
        success_count = len(all_test_cases)
        scenarios_processed = len(test_list)
        summary_message = f"Sequential processing completed: {success_count} test cases generated from {scenarios_processed} scenarios (avg {success_count//max(scenarios_processed,1)} test cases per scenario)"
        error_count = len(errors)

        if error_count > 0:
            summary_message += f", {error_count} errors encountered"
        
        updated_state['message_history'] = [HumanMessage(content=summary_message)]
        
        print(f"[PDFGraph] Stage 2 completed with {success_count} test cases from {scenarios_processed} scenarios")
        return updated_state
    
    def _qa_reflection_stage2_node(self, state):
        """
        Reflection node for stage 2 - evaluates the generated test cases
        and provides quality assessment. Always passes to end regardless of quality.
        """
        
        # Get the current test details list from state
        test_details_list = state.get('test_details_list', [])
        stage2_thread_id = state.get('stage2_thread_id', '')
        
        print(f"\n=== REFLECTION STAGE 2 ===")
        print(f"Thread ID: {stage2_thread_id}")
        print(f"Test cases generated: {len(test_details_list)}")
        print(f"Test details list type: {type(test_details_list)}")
        if test_details_list:
            print(f"First test case sample: {test_details_list[0] if len(test_details_list) > 0 else 'None'}")
        else:
            print("WARNING: test_details_list is empty!")
        
        # Simplified quality evaluation for test cases
        quality_score = self._evaluate_test_cases_simple(test_details_list)
        
        # Decision logic based on quality score (0.5 threshold)
        min_quality_threshold = 0.5  # 50% quality threshold
        quality_reasons = []
        
        if quality_score < min_quality_threshold:
            print(f"⚠️ Stage 2 quality below threshold (Quality: {quality_score:.2f} < {min_quality_threshold})")
            
            # Collect reasons for low quality
            quality_reasons = self._collect_quality_issues(test_details_list)
            print(f"⚠️ [PDFGraph] Quality issues found: {len(quality_reasons)} issues")
            for i, reason in enumerate(quality_reasons[:3], 1):  # Show first 3 reasons
                print(f"  {i}. {reason}")
            
            # Still pass to end despite low quality
            print("✓ Proceeding to end despite quality issues")
        else:
            print(f"✓ Stage 2 completed successfully (Quality: {quality_score:.2f})")
        
        # Always return finished state regardless of quality
        updated_state = state.copy()
        
        # Pass test data directly to state for frontend Excel generation
        print(f"[PDFGraph] Adding {len(test_details_list)} test cases to state for frontend Excel generation")
            
        updated_state.update({
            'is_finished_stage2': True,
            'stage2_quality_score': quality_score,
            'stage2_quality_reasons': quality_reasons,
            'test_list_data': test_details_list,  # Add for Streamlit Excel generation
            'message_history': state.get('message_history', []) + [
                HumanMessage(content=f"Stage 2 reflection completed. Quality score: {quality_score:.2f}. Generated {len(test_details_list)} test cases ready for download.")
            ]
        })
        
        print(f"🎯 [PDFGraph] Stage 2 reflection completed. Final quality: {quality_score:.2f}")
        print(f"📊 [PDFGraph] Generating Excel sheet with {len(test_details_list)} test cases...")
        return updated_state
    
    def _evaluate_test_cases_simple(self, test_details_list):
        """
        Simplified test case evaluation that checks basic quality metrics.
        Returns a score between 0.0 and 1.0.
        """
        if not test_details_list:
            return 0.0
        
        total_score = 0.0
        
        for test_case in test_details_list:
            case_score = 0.0
            
            # Check 1: Has required fields (30% weight)
            required_fields = ['Test_Case_ID', 'Title', 'Description', 'Expected_Result']
            field_score = sum(1 for field in required_fields if test_case.get(field)) / len(required_fields)
            case_score += field_score * 0.3
            
            # Check 2: Has meaningful content (25% weight)
            description = test_case.get('Description', '')
            title = test_case.get('Title', '')
            if len(description) > 20 and len(title) > 10:
                case_score += 0.25
            
            # Check 3: Has test steps (20% weight)
            test_steps = test_case.get('Test_Steps', [])
            if isinstance(test_steps, list) and len(test_steps) >= 2:
                case_score += 0.2
            
            # Check 4: Has request/response structure (25% weight)
            has_request = bool(test_case.get('Request_Body'))
            has_response = bool(test_case.get('Response'))
            if has_request and has_response:
                case_score += 0.25
            elif has_request or has_response:
                case_score += 0.125
            
            total_score += case_score
        
        # Average score across all test cases
        average_score = total_score / len(test_details_list)
        
        print(f"  - Average test case quality: {average_score:.2f}")
        print(f"  - Total test cases evaluated: {len(test_details_list)}")
        
        return average_score
    
    def _collect_quality_issues(self, test_details_list):
        """
        Collect specific quality issues with the generated test cases.
        """
        issues = []
        
        for i, test_case in enumerate(test_details_list, 1):
            # Check for missing required fields
            required_fields = ['Test_Case_ID', 'Title', 'Description', 'Expected_Result']
            missing_fields = [field for field in required_fields if not test_case.get(field)]
            if missing_fields:
                issues.append(f"Test case {i}: Missing fields {missing_fields}")
            
            # Check for short descriptions
            description = test_case.get('Description', '')
            if len(description) < 20:
                issues.append(f"Test case {i}: Description too short ({len(description)} chars)")
            
            # Check for missing test steps
            test_steps = test_case.get('Test_Steps', [])
            if not isinstance(test_steps, list) or len(test_steps) < 2:
                issues.append(f"Test case {i}: Insufficient test steps")
            
            # Check for missing request/response
            if not test_case.get('Request_Body') and not test_case.get('Response'):
                issues.append(f"Test case {i}: Missing both request and response structures")
        
        return issues
    
    def _generate_excel_sheet(self, test_details_list):
        """
        Generate Excel sheet data from test cases list.
        Returns bytes containing Excel data.
        """
        import pandas as pd
        import io
        
        print(f"📋 [PDFGraph] _generate_excel_sheet called with {len(test_details_list) if test_details_list else 0} test cases")
        print(f"📋 [PDFGraph] test_details_list type: {type(test_details_list)}")
        
        if not test_details_list:
            print("📋 [PDFGraph] ERROR: No test cases to export to Excel - returning None")
            return None
        
        print(f"📋 [PDFGraph] Sample test case structure: {test_details_list[0] if test_details_list else 'None'}")
        
        # Convert test cases to DataFrame
        print(f"📊 [PDFGraph] Converting {len(test_details_list)} test cases to DataFrame...")
        try:
            df = pd.DataFrame(test_details_list)
            print(f"✅ [PDFGraph] DataFrame created successfully with shape: {df.shape}")
            print(f"📊 [PDFGraph] DataFrame columns: {list(df.columns)}")
        except Exception as e:
            print(f"[PDFGraph] ERROR: Failed to create DataFrame: {e}")
            return None
        
        # Add Result column with predefined labels
        result_labels = ['Adopt', 'New Addition', 'Not Used(Basic)', 'Not Used(Irrelevant)', 'Not Used(Other)']
        df['Result'] = result_labels * (len(df) // len(result_labels)) + result_labels[:len(df) % len(result_labels)]
        
        # Reorder columns to put important ones first
        column_order = [
            'Test_Case_ID', 'Title', 'Description', 'Test_Type', 'Status', 'Result',
            'Preconditions', 'Test_Steps', 'Test_Data', 'Expected_Result',
            'Request_Body', 'Response', 'Actual_Result', 'Postconditions', 'Tags'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        df = df[final_columns]
        
        # Create Excel buffer
        print(f"[PDFGraph] Creating Excel buffer with {len(df)} rows...")
        output = io.BytesIO()
        try:
            print(f"[PDFGraph] Starting Excel writer with xlsxwriter engine...")
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                print(f"[PDFGraph] Writing DataFrame to Excel sheet...")
                df.to_excel(writer, index=False, sheet_name='Test_Cases')
                
                # Get workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets['Test_Cases']
                
                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Format headers
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-adjust column widths
                for i, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),  # Max length in column
                        len(str(col))  # Length of column name
                    )
                    worksheet.set_column(i, i, min(max_length + 2, 50))  # Cap at 50 chars
            
            # Get the bytes value and ensure it's properly serialized
            excel_data = output.getvalue()
            print(f"✅ [PDFGraph] Excel generation successful with {len(df)} rows and {len(excel_data)} bytes")
            return excel_data
            
        except Exception as e:
            print(f"❌ [PDFGraph] Excel generation failed: {e}")
            return None

    def _should_continue_stage1_qa(self, state):
        # End if we have processed all the test types    
        is_scenario_list_processed = state.get('is_scenario_list_processed')
        if is_scenario_list_processed:
            print("[PDFGraph] Finished all test scenarios.....")
            return END

        # Move to next test type if we have finished processing the current test type
        is_finished = state.get('is_finished_stage1', False)
        print(f"📊 [PDFGraph] Stage 1 finished ----> {is_finished}")
        
        # Set max revisions limit
        max_revisions = 3
        stage1_revisions = state.get('stage1_revisions', 0)
        
        if is_finished:
            # Stage 1 reflection determined we're done - move to stage 2
            print("✅ [PDFGraph] Stage 1 reflection complete, moving to stage 2")
            print(f"📊 [PDFGraph] Test count: {len(state.get('scenario_list', []))}")
            return "assist_stage2"
        elif stage1_revisions >= max_revisions:
            # End after max iterations
            print(f"⏰ [PDFGraph] Reached max revisions ({max_revisions}) for stage 1")
            return END
            
        # Revise answer with followup question - go back to assist_stage1
        print(f"🔄 [PDFGraph] Continuing stage 1 revision #{stage1_revisions + 1}")
        return "assist_stage1"

    def _should_continue_stage2_qa(self, state):
        # End if we have processed all test cases
        is_test_list_processed = state.get('is_test_list_processed')
        if is_test_list_processed:
            print("🏁 [PDFGraph] Finished all test cases.....")
            return END

        # Move to end if stage 2 is finished
        is_finished = state.get('is_finished_stage2', False)
        print(f"📊 [PDFGraph] Stage 2 finished ----> {is_finished}")
        
        # Set max revisions limit
        max_revisions = 3
        stage2_revisions = state.get('stage2_revisions', 0)
        
        if is_finished:
            # Stage 2 reflection determined we're done - end the process
            print("✅ [PDFGraph] Stage 2 reflection complete, ending process")
            print(f"📊 [PDFGraph] Test cases generated: {len(state.get('test_details_list', []))}")
            return END
        elif stage2_revisions >= max_revisions:
            # End after max iterations
            print(f"⏰ [PDFGraph] Reached max revisions ({max_revisions}) for stage 2")
            return END
            
        # Revise answer with followup question - go back to assist_stage2
        print(f"🔄 [PDFGraph] Continuing stage 2 revision #{stage2_revisions + 1}")
        return "assist_stage2"
