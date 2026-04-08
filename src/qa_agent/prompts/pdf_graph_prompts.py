def get_advanced_tests_prompt(user_journey="", special_instructions=""):
    prompt = f"""Design test SCENARIOS (not test cases) for user journey '{user_journey}' based on the provided Feature Requirement Document (FRD), Technical Design Document (TD), and API Documentation. Focus on user journeys or key features when the TD is available. Utilize advanced test design techniques to generate a comprehensive list of test scenarios.

                    **IMPORTANT NOTES:** 
                        - Generate TEST SCENARIOS (high-level descriptions), NOT detailed test cases
                        - Just generate requested JSON without any other details, observations or instructions
                        - Be comprehensive but focused on core functionality
                        - Generate 10-15 scenarios covering essential functionality
                        - If this is a revision, INCORPORATE the feedback provided and generate DIFFERENT scenarios

                    # Instructions:

                    1. **Review Documents:**
                    - Thoroughly analyze the FRD to identify all intended functionalities and user stories.
                    - Examine the TD to understand the technical implementation of the key features and user journeys.
                    - Review the API Documentation to understand endpoints, request/response formats, and integration points.
                    - Pay closer attention to High-level and Low-level designs to identify functional or technical aspects that need to be tested.
                    - Look for any schema changes, settings and flags to generate test scenarios around it.
                    - Consult the product knowledge base for additional context and historical information.
                    - **Review any other attached documents** (e.g., architecture diagrams, integration notes, test reports) that may provide relevant context, validations, or usage flows related to the feature or endpoint.

                    2. **Identify Key Features:**
                    - From the TD and API docs, pinpoint major features and user journeys critical to the application's functionality.
                    - Identify API endpoints that need testing for various scenarios.

                    3. **Apply Test Design Techniques:**
                    - Use advanced test design methods such as boundary value analysis, equivalence partitioning, state transition, decision table testing, and use case testing to formulate scenarios.
                    - Focus particularly on user journeys or critical features outlined in the TD.
                    - Include API-specific testing scenarios like authentication, authorization, data validation, error handling.

                    4. **Define Test Scenarios:**
                    - Create test scenarios that cover all possible paths, including positive, negative, edge cases, and unexpected inputs.
                    - Include scenarios for API testing: valid/invalid requests, authentication failures, rate limiting, etc.

                    5. **Document Scenario Descriptions:**
                    - Write clear and concise descriptions for each test scenario.

                    6. **Determine Expected Results:**
                    - Define the expected outcome for each test scenario based on the documents reviewed.

                    # Special Instructions:
                    {special_instructions}

                    # Output Format

                    Format the output as a JSON object with a key *strictly* named "test_list" containing a JSON list, where each list element is an object representing a test case. Each object should follow the outlined structure:
                        1. `scenarioDescription`: A string describing the test scenario.
                        2. `expectedResults`: A string detailing the expected results of that scenario.

                    Example JSON Output:
                        ```json
                        {{
                            'test_list':
                                [
                                    {{
                                        "scenarioDescription": "Scenario 1",
                                        "expectedResults": "Expected result for scenario 1"
                                    }},
                                    {{
                                        "scenarioDescription": "Scenario 2",
                                        "expectedResults": "Expected result for scenario 2"
                                    }}
                                    
                                ]
                        }}
                        ```

                        # Examples

                        **Example Input:**
                        - A snippet from an FRD or TD describing a functionality or feature.

                        **Example Output:**
                        ```json
                        {{
                            'test_list':

                                [
                                    {{
                                        "scenarioDescription": "Scenario 1",
                                        "expectedResults": "Expected result for scenario 1"
                                    }},
                                    {{
                                        "scenarioDescription": "Scenario 2",
                                        "expectedResults": "Expected result for scenario 2"
                                    }}
                                    
                                ]
                        }}
                        ```

                        # Notes

                        - Ensure comprehensive coverage by including various test case types: functional, performance, security, and user interface scenarios.
                        - Consider both positive and negative test cases to validate the robustness and error handling of the feature.
                        - If applicable, include scenarios for different user roles and permissions to ensure role-based access control is respected.
                        - Ensure that each scenario is aligned with the priority and criticality indicated in the documents.
                        - Include exploratory tests to discover unknown issues or behavior not specified in the FRD or TD.
                        - For API testing, include scenarios for different HTTP methods, status codes, payload validation, and error responses.

                """
    return prompt

def get_pdf_processing_prompt():
    """Prompt for processing uploaded PDF documents"""
    prompt = """You are analyzing uploaded PDF documents (Design Documents and API Documentation) to generate comprehensive test scenarios.

    Your task is to:
    1. Extract key features and functionalities from the design documents
    2. Identify API endpoints and their specifications from the API documentation
    3. Generate test scenarios that cover both functional and API testing aspects
    4. Focus on integration points between different components
    5. Consider edge cases and error scenarios

    Please provide a thorough analysis and generate comprehensive test scenarios based on the uploaded documents.
    """
    return prompt

def get_qa_reflection_prompt(scenario_count, stage1_revisions, design_doc_names=None, api_doc_name=None):
    """Prompt for QA reflection and validation of generated test scenarios"""
    
    # Simple dynamic service description (same logic as user_journey)
    service_description = f"{', '.join(design_doc_names)} and {api_doc_name}" if design_doc_names and api_doc_name else (', '.join(design_doc_names) if design_doc_names else api_doc_name)
    service_name = design_doc_names[0] if design_doc_names else api_doc_name
    main_segments = "main service components and modules"
    
    prompt = f"""You are an expert Quality Assurance engineer reviewing test scenarios generated from PDF documents.

The user has uploaded {service_description} and generated {scenario_count} test scenarios.

**THRESHOLD CRITERIA FOR COMPLETION:**
Consider the scenarios FINISHED if they meet these criteria:
✅ At least 10+ scenarios covering core functionality
✅ Include both positive and negative test scenarios
✅ Cover main segment types: {main_segments}
✅ Include API endpoint testing scenarios
✅ Cover authentication/authorization scenarios
✅ Include error handling scenarios
✅ Based on actual document content (not generic)

**CURRENT STATUS:**
- Test scenarios generated: {scenario_count}
- Current revision count: {stage1_revisions}
- Max revisions allowed: 3

**EVALUATION GUIDELINES:**
- If revision count is 2 or higher AND scenarios >= 10, lean towards FINISHED
- Only request revision if there are CRITICAL gaps in core functionality
- Don't request revisions for minor edge cases or nice-to-have scenarios
- Focus on essential {service_name} functionality coverage

**DECISION CRITERIA:**
- FINISHED = Core functionality well covered with good variety
- NOT FINISHED = Missing critical functionality or major gaps

Respond with your reflection in JSON format:
{{
    "Finished": true/false,
    "follow_up_question": "Your specific follow-up question if not finished",
    "reasonings": "Your detailed reasoning for the decision based on threshold criteria"
}}
"""
    return prompt
