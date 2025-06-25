#!/usr/bin/env python3
"""
test_templates.py - Test script for demonstrating specialized reasoning templates

This script demonstrates how the specialized reasoning templates work with different 
types of questions. It shows template detection, prompt formatting, response parsing,
and reasoning quality evaluation.
"""

import json
from yukon.templates import (
    detect_template,
    get_template_by_name,
    list_available_templates,
    get_template_details
)

# Define example questions for each domain
EXAMPLE_QUESTIONS = {
    "general": "What are the key factors that contributed to the decline of the Roman Empire?",
    "math": "Solve for x: 3x² + 6x - 9 = 0 using the quadratic formula.",
    "scientific": "Explain how mRNA vaccines work and how they differ from traditional vaccines.",
    "ethical": "Is it ethical to use genetic engineering to enhance human capabilities?",
    "programming": "Implement a function to find the longest common subsequence of two strings with O(n*m) time complexity."
}

# Define shortened example responses to avoid potential syntax issues with large multiline strings
EXAMPLE_RESPONSES = {
    "general": """
PREMISES:
- The Roman Empire reached its peak during the Pax Romana (27 BCE - 180 CE)
- The decline was a gradual process that occurred over several centuries

REASONING STEPS:
1. Internal political factors: Political instability weakened central authority.
2. Economic challenges: Inflation and overtaxation caused economic decline.
3. Military pressures: External invasions strained military resources.

CONCLUSION:
The decline of the Roman Empire resulted from a complex interaction of factors.
""",

    "math": """
DEFINITIONS:
- The quadratic formula for an equation in the form ax² + bx + c = 0 is:
  x = (-b ± √(b² - 4ac)) / 2a
- In our equation 3x² + 6x - 9 = 0, we have: a = 3, b = 6, c = -9

APPROACH:
I'll use the quadratic formula to solve for x.

SOLUTION STEPS:
1. Substitute the values into the quadratic formula:
   x = (-6 ± √(36 + 108)) / 6 = (-6 ± √144) / 6 = (-6 ± 12) / 6

FINAL ANSWER:
The solutions are x = 1 and x = -3.

VERIFICATION:
Both solutions satisfy the original equation.
""",

    "scientific": """
BACKGROUND:
Vaccines provide immunity against diseases. Traditional vaccines use weakened pathogens or components, while mRNA vaccines use different mechanisms.

HYPOTHESIS/QUESTION:
How do mRNA vaccines work, and what are their differences from traditional vaccines?

EVIDENCE ANALYSIS:
mRNA vaccines contain synthetic mRNA that codes for a viral protein, not the actual pathogen. Clinical trials showed high efficacy rates for COVID-19 mRNA vaccines.

EVALUATION:
Key differences include manufacturing process, mechanism of action, stability requirements, adaptability, safety profile, and immune response.

CONCLUSION:
mRNA vaccines deliver instructions for cells to produce viral proteins, triggering an immune response without using any pathogen components.
""",

    "ethical": """
ETHICAL ISSUE:
Is it morally permissible to use genetic engineering to enhance human capabilities beyond normal function?

STAKEHOLDERS:
1. Individuals seeking enhancement
2. Children and future generations
3. Medical professionals
4. Society at large
5. Regulatory bodies
6. Companies and investors

ETHICAL PRINCIPLES:
Various frameworks including consequentialism, deontology, virtue ethics, and bioethical principles apply.

PERSPECTIVE ANALYSIS:
Arguments supporting: autonomy, improved quality of life, scientific progress.
Arguments against: inequality, dignity concerns, safety risks, consent issues.

BALANCED CONCLUSION:
A nuanced approach is needed that balances competing ethical values and prioritizes safety, equity, and consent.
""",

    "programming": """
PROBLEM ANALYSIS:
Find the longest common subsequence (LCS) of two strings with O(n*m) time complexity.

ALGORITHM DESIGN:
Use dynamic programming with a 2D table where DP[i][j] represents the LCS length of the prefixes.

COMPLEXITY ANALYSIS:
Time: O(n*m)
Space: O(n*m)

IMPLEMENTATION:
```python
def longest_common_subsequence(str1, str2):
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[n][m]
```

TESTING & EDGE CASES:
Basic test case: "ABCBDAB" and "BDCABA" should return 4 ("BCBA").
Edge cases include empty strings and no common characters.
"""
}

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 80 + "\n")

def test_template_detection():
    """Test the template detection functionality."""
    print("TESTING TEMPLATE DETECTION\n")
    
    for domain, question in EXAMPLE_QUESTIONS.items():
        template = detect_template(question)
        print(f"Domain: {domain}")
        print(f"Question: {question}")
        print(f"Detected template: {template.name}")
        print(f"Match confidence: {template.matches(question):.2f}")
        print()
    
    print_separator()

def test_prompt_formatting():
    """Test the prompt formatting functionality."""
    print("TESTING PROMPT FORMATTING\n")
    
    for domain, question in EXAMPLE_QUESTIONS.items():
        template = detect_template(question)
        formatted_prompt = template.format_prompt(question)
        
        print(f"Domain: {domain}")
        print(f"Template: {template.name}")
        print("Formatted prompt:")
        print(formatted_prompt)
        print()
    
    print_separator()

def test_response_parsing():
    """Test the response parsing functionality."""
    print("TESTING RESPONSE PARSING\n")
    
    for domain, question in EXAMPLE_QUESTIONS.items():
        template = detect_template(question)
        response = EXAMPLE_RESPONSES[domain]
        
        parsed_response = template.parse_response(response)
        
        print(f"Domain: {domain}")
        print(f"Template: {template.name}")
        print("Parsed response components:")
        for key, value in parsed_response.items():
            if key != "full_response":
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
        
        quality_score = template.evaluate_reasoning(parsed_response)
        print(f"Reasoning quality score: {quality_score:.2f}")
        print()
    
    print_separator()

def test_template_management():
    """Test the template management functions."""
    print("TESTING TEMPLATE MANAGEMENT\n")
    
    # List available templates
    print("Available templates:")
    templates = list_available_templates()
    for template in templates:
        print(f"  - {template['name']} (ID: {template['id']})")
    print()
    
    # Get template by name
    print("Getting template by name:")
    math_template = get_template_by_name("math")
    print(f"  - Retrieved: {math_template.name}")
    
    # Get template details
    print("\nTemplate details:")
    details = get_template_details("ethical")
    for key, value in details.items():
        if key != "example_prompt":
            print(f"  - {key}: {value}")
    
    print_separator()

def main():
    """Run all tests."""
    print("\nSPECIALIZED REASONING TEMPLATES TEST SUITE\n")
    print("This script demonstrates the specialized reasoning templates functionality.")
    
    test_template_detection()
    test_prompt_formatting()
    test_response_parsing()
    test_template_management()
    
    print("All tests completed successfully.")

if __name__ == "__main__":
    main()

