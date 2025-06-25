"""
consensus_templates.py — Specialized reasoning templates for different domains

This module provides domain-specific reasoning templates that can be used with the
consensus agent system. Each template defines a structured approach to reasoning
that is tailored to a specific type of problem or domain.

Templates provide:
- Domain detection (determining if a question belongs to a domain)
- Prompt engineering (crafting prompts to elicit domain-specific reasoning)
- Response parsing (extracting structured reasoning from responses)
- Evaluation criteria (assessing quality of domain-specific reasoning)

Usage:
    from consensus_templates import detect_template
    
    # Automatically detect the appropriate template
    template = detect_template(question)
    
    # Generate a domain-specific prompt
    enhanced_prompt = template.format_prompt(question)
    
    # Parse and evaluate a response
    structured_response = template.parse_response(response_text)
    quality_score = template.evaluate_reasoning(structured_response)
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set

class ReasoningTemplate(ABC):
    """
    Base class for all specialized reasoning templates.
    
    Each template provides methods for detecting applicability, formatting prompts,
    parsing responses, and evaluating reasoning quality specific to a domain.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a reasoning template.
        
        Args:
            name: Short name of the template
            description: Longer description of the template's purpose
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def matches(self, question: str) -> float:
        """
        Determine if this template is applicable to the given question.
        
        Args:
            question: The input question to analyze
            
        Returns:
            Float between 0.0 and 1.0 indicating confidence that this template applies
        """
        pass
    
    @abstractmethod
    def format_prompt(self, question: str) -> str:
        """
        Format a question with the template's specialized reasoning structure.
        
        Args:
            question: The original question
            
        Returns:
            Enhanced prompt with template-specific reasoning instructions
        """
        pass
    
    def parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse a response into structured components based on the template.
        
        Args:
            response: The raw response text
            
        Returns:
            Dictionary of structured components (varies by template)
        """
        # Default implementation - subclasses should override for specialized parsing
        return {"full_response": response}
    
    def evaluate_reasoning(self, parsed_response: Dict[str, str]) -> float:
        """
        Evaluate the quality of reasoning in a parsed response.
        
        Args:
            parsed_response: The structured response components
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Default implementation - subclasses should override for specialized evaluation
        return 0.5  # Neutral score
    
    def extract_keywords(self, text: str) -> Set[str]:
        """
        Extract relevant keywords from text for matching purposes.
        
        Args:
            text: The text to analyze
            
        Returns:
            Set of extracted keywords
        """
        # Simple keyword extraction - remove common words, lowercase, etc.
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
        return {word for word in words if word not in stopwords and len(word) > 2}


class GeneralAnalyticalTemplate(ReasoningTemplate):
    """
    General analytical reasoning template for broad critical thinking.
    
    This is the default template used when no specialized template is a strong match.
    It focuses on premises, reasoning steps, and conclusions.
    """
    
    def __init__(self):
        super().__init__(
            name="General Analytical Reasoning",
            description="All-purpose reasoning template for general critical thinking tasks"
        )
        
    def matches(self, question: str) -> float:
        """
        General template should be chosen for history, social sciences, and general knowledge questions
        that don't fit well into other specialized templates.
        """
        # Extract keywords for detection
        keywords = self.extract_keywords(question)
        
        # Keywords that suggest general analytical reasoning
        general_keywords = {
            'history', 'society', 'culture', 'politics', 'economy', 'government', 
            'policy', 'social', 'movement', 'revolution', 'war', 'conflict', 
            'empire', 'civilization', 'leader', 'influence', 'impact', 'effect',
            'cause', 'factor', 'contribute', 'reason', 'explain', 'analyze',
            'discuss', 'evaluate', 'describe', 'compare', 'contrast', 'summarize'
        }
        
        # Count general keywords
        general_keyword_count = len(keywords.intersection(general_keywords))
        
        # Calculate match score
        keyword_score = min(0.7, general_keyword_count / 2.0)
        
        # Base confidence (for fallback purposes)
        base_confidence = 0.3
        
        return max(base_confidence, keyword_score)
    
    def format_prompt(self, question: str) -> str:
        return f"""
I need you to think through this step-by-step using structured reasoning.

Structure your answer as follows:
1. PREMISES: State your initial assumptions and key facts you're using
2. REASONING STEPS: Work through the problem step-by-step, explaining each logical step
3. CONCLUSION: State your final answer clearly

QUESTION: {question}
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        # Extract structured components from the response
        premises = ""
        steps = ""
        conclusion = ""
        
        # Look for premise section
        premise_match = re.search(r'(?:PREMISES|Premises|premises|1\.)[^\n]*\n(.*?)(?=(?:REASONING|Reasoning|reasoning|2\.|CONCLUSION|Conclusion|conclusion|3\.))', response, re.DOTALL)
        if premise_match:
            premises = premise_match.group(1).strip()
            
        # Look for reasoning steps section
        steps_match = re.search(r'(?:REASONING|Reasoning|reasoning|2\.)[^\n]*\n(.*?)(?=(?:CONCLUSION|Conclusion|conclusion|3\.))', response, re.DOTALL)
        if steps_match:
            steps = steps_match.group(1).strip()
            
        # Look for conclusion section
        conclusion_match = re.search(r'(?:CONCLUSION|Conclusion|conclusion|3\.)[^\n]*\n(.*?)(?=$|\n\n)', response, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            
        return {
            "premises": premises,
            "steps": steps,
            "conclusion": conclusion,
            "full_response": response
        }
    
    def evaluate_reasoning(self, parsed_response: Dict[str, str]) -> float:
        """Evaluate the quality of general analytical reasoning."""
        score = 0.0
        
        # Check for presence of all required components
        if parsed_response.get("premises"):
            score += 0.25
        if parsed_response.get("steps"):
            score += 0.25
        if parsed_response.get("conclusion"):
            score += 0.25
            
        # Look for logical connectors in reasoning steps
        reasoning = parsed_response.get("steps", "")
        logical_markers = ["therefore", "because", "since", "given that", "it follows", 
                          "consequently", "thus", "hence", "as a result"]
        if any(marker in reasoning.lower() for marker in logical_markers):
            score += 0.15
            
        # Check for structured format (numbered or bulleted list)
        if re.search(r'(\d+\.|•|\*)\s', reasoning):
            score += 0.1
            
        return min(1.0, score)


class MathematicalTemplate(ReasoningTemplate):
    """
    Mathematical reasoning template for problems involving math, logic, or computation.
    
    This template emphasizes formal definitions, theorem application, calculation steps,
    and precise solutions.
    """
    
    def __init__(self):
        super().__init__(
            name="Mathematical Reasoning",
            description="Structured approach to mathematical problem-solving"
        )
        
        self.math_keywords = {
            'calculate', 'compute', 'solve', 'equation', 'theorem', 'proof',
            'formula', 'function', 'variable', 'expression', 'algebra', 'geometry',
            'calculus', 'trigonometry', 'arithmetic', 'probability', 'statistics',
            'number', 'sum', 'product', 'difference', 'quotient', 'equals',
            'greater', 'less', 'integral', 'derivative', 'limit', 'infinity',
            'matrix', 'vector', 'series', 'sequence', 'prime', 'fraction',
            'decimal', 'percentage', 'logarithm', 'exponent', 'square', 'root',
            'graph', 'function', 'domain', 'range', 'minimum', 'maximum'
        }
    
    def matches(self, question: str) -> float:
        # Check for mathematical keywords and symbols
        question_lower = question.lower()
        keywords = self.extract_keywords(question)
        
        # Count mathematical keywords
        math_keyword_count = len(keywords.intersection(self.math_keywords))
        
        # Check for mathematical symbols
        math_symbols = sum(1 for c in question if c in "+-*/=^()[]{}≠≈≤≥∫∑∏√∞π")
        
        # Check for numbers
        number_count = len(re.findall(r'\d+(?:\.\d+)?', question))
        
        # Calculate match score
        keyword_score = min(1.0, math_keyword_count / 3.0)
        symbol_score = min(1.0, math_symbols / 3.0)
        number_score = min(1.0, number_count / 2.0)
        
        # Compute the combined score
        combined_score = (keyword_score + symbol_score + number_score) / 3
        
        # Lower the default confidence to avoid overmatching
        # Only return a high score if we're confident this is math
        if combined_score < 0.3:
            return 0.2  # Lower base confidence than before
        
        return combined_score
    
    def format_prompt(self, question: str) -> str:
        return f"""
I need you to solve this mathematical problem using a formal, step-by-step approach.

Structure your solution as follows:
1. DEFINITIONS: Define any relevant variables, notation, or mathematical concepts
2. APPROACH: Describe the mathematical technique or theorem you'll apply
3. SOLUTION STEPS: Show your work with clear mathematical steps (include equations)
4. FINAL ANSWER: State the precise answer, including units if applicable
5. VERIFICATION: Verify your answer if possible (e.g., checking boundary conditions)

PROBLEM: {question}
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        # Extract structured components from the response
        definitions = ""
        approach = ""
        steps = ""
        answer = ""
        verification = ""
        
        # Extract each section using regex
        definitions_match = re.search(r'(?:DEFINITIONS|Definitions|1\.)[^\n]*\n(.*?)(?=(?:APPROACH|Approach|2\.))', response, re.DOTALL)
        if definitions_match:
            definitions = definitions_match.group(1).strip()
            
        approach_match = re.search(r'(?:APPROACH|Approach|2\.)[^\n]*\n(.*?)(?=(?:SOLUTION|Solution|3\.))', response, re.DOTALL)
        if approach_match:
            approach = approach_match.group(1).strip()
            
        steps_match = re.search(r'(?:SOLUTION|Solution|3\.)[^\n]*\n(.*?)(?=(?:FINAL|Final|4\.|ANSWER|Answer))', response, re.DOTALL)
        if steps_match:
            steps = steps_match.group(1).strip()
            
        answer_match = re.search(r'(?:FINAL|Final|ANSWER|Answer|4\.)[^\n]*\n(.*?)(?=(?:VERIFICATION|Verification|5\.|$))', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            
        verification_match = re.search(r'(?:VERIFICATION|Verification|5\.)[^\n]*\n(.*?)(?=$|\n\n)', response, re.DOTALL)
        if verification_match:
            verification = verification_match.group(1).strip()
            
        return {
            "definitions": definitions,
            "approach": approach,
            "steps": steps,
            "answer": answer,
            "verification": verification,
            "full_response": response
        }
    
    def evaluate_reasoning(self, parsed_response: Dict[str, str]) -> float:
        """Evaluate the quality of mathematical reasoning."""
        score = 0.0
        
        # Check for presence of all required components
        if parsed_response.get("definitions"):
            score += 0.15
        if parsed_response.get("approach"):
            score += 0.20
        if parsed_response.get("steps"):
            score += 0.25
        if parsed_response.get("answer"):
            score += 0.20
        if parsed_response.get("verification"):
            score += 0.20
            
        # Look for mathematical notation in the solution steps
        steps = parsed_response.get("steps", "")
        if re.search(r'[=+\-*/^√∫∑∏]', steps):
            score += 0.10
            
        # Look for numbered equations or steps
        if re.search(r'(\d+\)|\d+\.|\(\d+\))', steps):
            score += 0.10
            
        # Penalty for lack of mathematical notation if this is truly a math problem
        if not re.search(r'[=+\-*/^√∫∑∏]', steps) and len(steps) > 100:
            score -= 0.20
            
        return min(1.0, max(0.0, score))


class ScientificTemplate(ReasoningTemplate):
    """
    Scientific reasoning template for questions involving scientific analysis.
    
    This template emphasizes hypotheses, evidence evaluation, and drawing
    conclusions based on scientific principles.
    """
    
    def __init__(self):
        super().__init__(
            name="Scientific Analysis",
            description="Scientific reasoning using hypothesis testing and evidence evaluation"
        )
        
        self.science_keywords = {
            'hypothesis', 'theory', 'experiment', 'evidence', 'data', 'observation',
            'research', 'study', 'test', 'analyze', 'measure', 'method', 'control',
            'variable', 'sample', 'result', 'conclusion', 'physics', 'chemistry',
            'biology', 'geology', 'astronomy', 'psychology', 'neuroscience', 'medicine',
            'genetics', 'evolution', 'ecology', 'climate', 'matter', 'energy', 'force',
            'cell', 'organism', 'species', 'molecule', 'atom', 'electron', 'neutron',
            'proton', 'photon', 'quantum', 'relativity', 'gravity', 'chemical',
            'reaction', 'compound', 'acid', 'base', 'enzyme', 'protein', 'dna', 'rna'
        }
    
    def matches(self, question: str) -> float:
        # Check for scientific keywords
        question_lower = question.lower()
        keywords = self.extract_keywords(question)
        
        # Count scientific keywords
        science_keyword_count = len(keywords.intersection(self.science_keywords))
        
        # Check for scientific phrases
        scientific_phrases = [
            'scientific method', 'research study', 'experimental evidence',
            'clinical trial', 'peer review', 'published research', 'data analysis',
            'statistical significance', 'control group', 'laboratory experiment',
            'empirical evidence', 'theoretical model', 'research findings'
        ]
        phrase_count = sum(1 for phrase in scientific_phrases if phrase in question_lower)
        
        # Calculate match score
        keyword_score = min(1.0, science_keyword_count / 3.0)
        phrase_score = min(1.0, phrase_count / 2.0)
        
        # Additional boost for science-specific content
        has_science_subject = any(term in question_lower for term in [
            'biology', 'chemistry', 'physics', 'astronomy', 'geology', 'medicine',
            'vaccine', 'climate', 'evolution', 'genetic', 'species', 'molecule',
            'experiment', 'lab', 'scientific', 'researcher', 'theory', 'hypothesis'
        ])
        
        subject_boost = 0.3 if has_science_subject else 0.0
        
        # Compute final score with a higher default for science questions
        combined_score = (keyword_score * 0.6 + phrase_score * 0.2 + subject_boost * 0.2)
        
        # Ensure science questions get priority when science terms are present
        if has_science_subject:
            return max(0.6, combined_score)
            
        return max(0.3, combined_score)
    
    def format_prompt(self, question: str) -> str:
        return f"""
I need you to analyze this scientific question using the scientific method and evidence-based reasoning.

Structure your analysis as follows:
1. BACKGROUND: Summarize relevant scientific principles and context
2. HYPOTHESIS/QUESTION: Clearly state the scientific question or hypothesis
3. EVIDENCE ANALYSIS: Examine relevant scientific evidence and data
4. EVALUATION: Critically evaluate the evidence and consider alternative explanations
5. CONCLUSION: Draw a scientifically sound conclusion based on the evidence

SCIENTIFIC QUESTION: {question}
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        # Extract structured components from the response
        background = ""
        hypothesis = ""
        evidence = ""
        evaluation = ""
        conclusion = ""
        
        # Improved regex patterns with more flexible matching
        # The key issue was that the patterns were too strict and didn't account for variations in formatting
        
        # Extract background section - look for content between BACKGROUND heading and the next heading
        background_match = re.search(r'(?:BACKGROUND|Background|1\.|^)[^\n]*\n(.*?)(?=(?:HYPOTHESIS|Hypothesis|QUESTION|Question|2\.|EVIDENCE|Evidence|3\.|$))', response, re.DOTALL)
        if background_match:
            background = background_match.group(1).strip()
        
        # Extract hypothesis/question section
        hypothesis_match = re.search(r'(?:HYPOTHESIS|Hypothesis|QUESTION|Question|2\.)[^\n]*\n(.*?)(?=(?:EVIDENCE|Evidence|3\.|EVALUATION|Evaluation|4\.|$))', response, re.DOTALL)
        if hypothesis_match:
            hypothesis = hypothesis_match.group(1).strip()
        
        # Extract evidence analysis section
        evidence_match = re.search(r'(?:EVIDENCE|Evidence|ANALYSIS|Analysis|3\.)[^\n]*\n(.*?)(?=(?:EVALUATION|Evaluation|4\.|CONCLUSION|Conclusion|5\.|$))', response, re.DOTALL)
        if evidence_match:
            evidence = evidence_match.group(1).strip()
        
        # Extract evaluation section
        evaluation_match = re.search(r'(?:EVALUATION|Evaluation|4\.)[^\n]*\n(.*?)(?=(?:CONCLUSION|Conclusion|5\.|$))', response, re.DOTALL)
        if evaluation_match:
            evaluation = evaluation_match.group(1).strip()
        
        # Extract conclusion section
        conclusion_match = re.search(r'(?:CONCLUSION|Conclusion|5\.)[^\n]*\n(.*?)(?=$)', response, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            
        # If no structured components were found, try a more flexible approach
        if not any([background, hypothesis, evidence, evaluation, conclusion]):
            # Try to find any paragraph mentioning key scientific terms
            scientific_terms = ['data', 'evidence', 'research', 'study', 'experiment', 'observation']
            paragraphs = re.split(r'\n\s*\n', response)
            
            for para in paragraphs:
                if any(term in para.lower() for term in scientific_terms):
                    evidence = para.strip()
                    break
            
        return {
            "background": background,
            "hypothesis": hypothesis,
            "evidence": evidence,
            "evaluation": evaluation,
            "conclusion": conclusion,
            "full_response": response
        }
    
    def evaluate_reasoning(self, parsed_response: Dict[str, str]) -> float:
        """Evaluate the quality of scientific reasoning."""
        score = 0.0
        
        # Check for presence of all required components
        if parsed_response.get("background"):
            score += 0.15
        if parsed_response.get("hypothesis"):
            score += 0.20
        if parsed_response.get("evidence"):
            score += 0.25
        if parsed_response.get("evaluation"):
            score += 0.20
        if parsed_response.get("conclusion"):
            score += 0.20
            
        # Check for citations or references to research
        evidence = parsed_response.get("evidence", "")
        if re.search(r'(study|research|experiment|trial|data|analysis|finding|published|journal|et al\.|\(\d{4}\))', evidence, re.IGNORECASE):
            score += 0.10
            
        # Check for consideration of alternative explanations
        evaluation = parsed_response.get("evaluation", "")
        if re.search(r'(alternative|different|other|contrary|opposing|another|possible explanation|different interpretation)', evaluation, re.IGNORECASE):
            score += 0.10
            
        # Check for scientific terminology
        full_text = parsed_response.get("full_response", "")
        science_term_count = sum(1 for term in self.science_keywords if term in full_text.lower().split())
        if science_term_count >= 5:
            score += 0.10
            
        return min(1.0, score)


class EthicalTemplate(ReasoningTemplate):
    """
    Ethical reasoning template for moral, ethical, and value-based questions.
    
    This template emphasizes multiple perspectives, stakeholder analysis,
    principles-based reasoning, and nuanced conclusions.
    """
    
    def __init__(self):
        super().__init__(
            name="Ethical Reasoning",
            description="Multi-perspective ethical analysis for moral questions"
        )
        
        self.ethics_keywords = {
            'ethics', 'moral', 'value', 'right', 'wrong', 'good', 'bad', 'just', 'unjust',
            'fair', 'unfair', 'virtue', 'vice', 'duty', 'obligation', 'responsibility',
            'rights', 'freedom', 'liberty', 'justice', 'equality', 'equity', 'dignity',
            'respect', 'harm', 'benefit', 'autonomy', 'consent', 'fairness', 'honesty',
            'truth', 'lie', 'deception', 'integrity', 'character', 'virtue', 'consequence',
            'utilitarian', 'deontological', 'kantian', 'aristotelian', 'principle',
            'dilemma', 'conflict', 'stakeholder', 'interest', 'welfare', 'wellbeing',
            'happiness', 'suffering', 'choice', 'decision', 'permissible', 'forbidden',
            'allowed', 'prohibited', 'should', 'ought', 'must', 'permissible'
        }
    
    def matches(self, question: str) -> float:
        # Check for ethical keywords
        question_lower = question.lower()
        keywords = self.extract_keywords(question)
        
        # Count ethical keywords
        ethics_keyword_count = len(keywords.intersection(self.ethics_keywords))
        
        # Check for ethical phrases and questions
        ethical_phrases = [
            'is it ethical', 'is it moral', 'is it right', 'is it wrong',
            'is it just', 'is it fair', 'should we', 'ought to', 'morally',
            'ethically', 'right thing', 'wrong thing', 'ethical implications',
            'moral considerations', 'ethical dilemma', 'moral question'
        ]
        phrase_count = sum(1 for phrase in ethical_phrases if phrase in question_lower)
        
        # Calculate match score
        keyword_score = min(1.0, ethics_keyword_count / 3.0)
        phrase_score = min(1.0, phrase_count)
        
        # Strong boost if it's clearly asking an ethical question
        if re.search(r'(is it (ethical|moral|right|wrong|just)|should we|ought to)', question_lower):
            return max(0.8, (keyword_score + phrase_score) / 2)
        
        return max(0.4, (keyword_score * 0.6 + phrase_score * 0.4))
    
    def format_prompt(self, question: str) -> str:
        return f"""
I need you to analyze this ethical question using a multi-perspective, principle-based approach.

Structure your ethical analysis as follows:
1. ETHICAL ISSUE: Clearly identify the ethical question or dilemma
2. STAKEHOLDERS: Identify the relevant parties and their interests
3. ETHICAL PRINCIPLES: Examine relevant ethical frameworks (e.g., consequentialism, deontology, virtue ethics)
4. PERSPECTIVE ANALYSIS: Consider multiple viewpoints and arguments for different positions
5. BALANCED CONCLUSION: Provide a nuanced conclusion that acknowledges the complexity of the issue

ETHICAL QUESTION: {question}
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        # Extract structured components from the response
        issue = ""
        stakeholders = ""
        principles = ""
        perspectives = ""
        conclusion = ""
        
        # Improved regex patterns with more flexible matching
        
        # Extract ethical issue section
        issue_match = re.search(r'(?:ETHICAL ISSUE|Ethical Issue|Issue|1\.)[^\n]*\n(.*?)(?=(?:STAKEHOLDERS|Stakeholders|2\.|$))', response, re.DOTALL)
        if issue_match:
            issue = issue_match.group(1).strip()
        
        # Extract stakeholders section
        stakeholders_match = re.search(r'(?:STAKEHOLDERS|Stakeholders|2\.)[^\n]*\n(.*?)(?=(?:ETHICAL PRINCIPLES|Ethical Principles|PRINCIPLES|Principles|3\.|$))', response, re.DOTALL)
        if stakeholders_match:
            stakeholders = stakeholders_match.group(1).strip()
        
        # Extract ethical principles section
        principles_match = re.search(r'(?:ETHICAL PRINCIPLES|Ethical Principles|PRINCIPLES|Principles|3\.)[^\n]*\n(.*?)(?=(?:PERSPECTIVE|Perspective|ANALYSIS|Analysis|4\.|$))', response, re.DOTALL)
        if principles_match:
            principles = principles_match.group(1).strip()
        
        # Extract perspective analysis section
        perspectives_match = re.search(r'(?:PERSPECTIVE|Perspective|ANALYSIS|Analysis|4\.)[^\n]*\n(.*?)(?=(?:BALANCED CONCLUSION|Balanced Conclusion|CONCLUSION|Conclusion|5\.|$))', response, re.DOTALL)
        if perspectives_match:
            perspectives = perspectives_match.group(1).strip()
        
        # Extract conclusion section
        conclusion_match = re.search(r'(?:BALANCED CONCLUSION|Balanced Conclusion|CONCLUSION|Conclusion|5\.)[^\n]*\n(.*?)(?=$)', response, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        
        # Additional parsing for the specific structure in our test example
        if not principles and "ETHICAL PRINCIPLES" in response:
            principles_section = response.split("ETHICAL PRINCIPLES")[1].split("PERSPECTIVE ANALYSIS")[0]
            principles = principles_section.strip()
            
        if not perspectives and "PERSPECTIVE ANALYSIS" in response:
            perspectives_section = response.split("PERSPECTIVE ANALYSIS")[1].split("BALANCED CONCLUSION")[0]
            perspectives = perspectives_section.strip()
            
        return {
            "issue": issue,
            "stakeholders": stakeholders,
            "principles": principles,
            "perspectives": perspectives,
            "conclusion": conclusion,
            "full_response": response
        }
    
    def evaluate_reasoning(self, parsed_response: Dict[str, str]) -> float:
        """Evaluate the quality of ethical reasoning."""
        score = 0.0
        
        # Check for presence of all required components
        if parsed_response.get("issue"):
            score += 0.15
        if parsed_response.get("stakeholders"):
            score += 0.15
        if parsed_response.get("principles"):
            score += 0.20
        if parsed_response.get("perspectives"):
            score += 0.25
        if parsed_response.get("conclusion"):
            score += 0.15
            
        # Check for reference to ethical frameworks
        principles = parsed_response.get("principles", "")
        frameworks = ['consequentialism', 'utilitarianism', 'deontology', 'kantian', 
                     'virtue ethics', 'aristotelian', 'social contract', 'justice', 
                     'care ethics', 'feminist ethics', 'natural law', 'divine command',
                     'relativism', 'absolutism', 'duty', 'rights-based']
        
        if any(framework in principles.lower() for framework in frameworks):
            score += 0.10
            
        # Check for multiple perspectives
        perspectives = parsed_response.get("perspectives", "")
        perspective_markers = ['on one hand', 'on the other hand', 'perspective', 
                              'viewpoint', 'argument', 'proponent', 'opponent', 
                              'advocate', 'critic', 'position', 'stance', 'side']
        
        perspective_count = sum(1 for marker in perspective_markers if marker in perspectives.lower())
        if perspective_count >= 3:
            score += 0.10
            
        # Check for nuance in conclusion
        conclusion = parsed_response.get("conclusion", "")
        nuance_markers = ['however', 'nevertheless', 'although', 'while', 'despite',
                         'balance', 'trade-off', 'complex', 'nuanced', 'tension',
                         'not straightforward', 'depends', 'context', 'situation']
        
        if any(marker in conclusion.lower() for marker in nuance_markers):
            score += 0.10
            
        return min(1.0, score)


class ProgrammingTemplate(ReasoningTemplate):
    """
    Programming and algorithm analysis template for coding and computational problems.
    
    This template emphasizes algorithm design, complexity analysis, implementation
    considerations, and testing approaches.
    """
    
    def __init__(self):
        super().__init__(
            name="Programming & Algorithm Analysis",
            description="Structured approach to coding problems and algorithm design"
        )
        
        self.programming_keywords = {
            'algorithm', 'code', 'program', 'function', 'variable', 'class', 'method',
            'object', 'compile', 'runtime', 'debug', 'error', 'exception', 'syntax',
            'implement', 'developer', 'software', 'application', 'system', 'database',
            'query', 'api', 'interface', 'framework', 'library', 'module', 'package',
            'dependency', 'import', 'export', 'data structure', 'array', 'list', 
            'stack', 'queue', 'tree', 'graph', 'hash', 'map', 'set', 'dictionary',
            'complexity', 'efficiency', 'performance', 'optimization', 'scalability',
            'big o', 'time complexity', 'space complexity', 'recursion', 'iteration',
            'loop', 'conditional', 'boolean', 'integer', 'string', 'float', 'null',
            'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'go', 'rust', 'swift'
        }
    
    def matches(self, question: str) -> float:
        # Check for programming keywords
        question_lower = question.lower()
        keywords = self.extract_keywords(question)
        
        # Count programming keywords
        programming_keyword_count = len(keywords.intersection(self.programming_keywords))
        
        # Check for code snippets
        code_snippet = re.search(r'```[\s\S]*?```|`[\s\S]*?`|\bdef\b|\bfunction\b|\bclass\b|\bfor\b.*:|if.*:|\bwhile\b', question)
        
        # Check for programming phrases
        programming_phrases = [
            'time complexity', 'space complexity', 'big o', 'algorithm', 'data structure',
            'implement', 'function', 'code', 'optimize', 'debug', 'refactor'
        ]
        phrase_count = sum(1 for phrase in programming_phrases if phrase in question_lower)
        
        # Calculate match score
        keyword_score = min(1.0, programming_keyword_count / 3.0)
        phrase_score = min(1.0, phrase_count / 2.0)
        code_score = 0.8 if code_snippet else 0.0
        
        return max(0.4, (keyword_score * 0.4 + phrase_score * 0.3 + code_score * 0.3))
    
    def format_prompt(self, question: str) -> str:
        return f"""
I need you to solve this programming or algorithm problem using a structured approach.

Structure your solution as follows:
1. PROBLEM ANALYSIS: Understand the problem requirements and constraints
2. ALGORITHM DESIGN: Describe your approach and any data structures needed
3. COMPLEXITY ANALYSIS: Analyze time and space complexity (Big O notation)
4. IMPLEMENTATION: Provide clean, well-commented code solution
5. TESTING & EDGE CASES: Discuss how to test and handle edge cases

PROGRAMMING PROBLEM: {question}
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        # Extract structured components from the response
        analysis = ""
        design = ""
        complexity = ""
        implementation = ""
        testing = ""
        
        # Extract each section using regex
        analysis_match = re.search(r'(?:PROBLEM ANALYSIS|Problem Analysis|1\.)[^\n]*\n(.*?)(?=(?:ALGORITHM DESIGN|Algorithm Design|2\.))', response, re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
            
        design_match = re.search(r'(?:ALGORITHM DESIGN|Algorithm Design|2\.)[^\n]*\n(.*?)(?=(?:COMPLEXITY ANALYSIS|Complexity Analysis|3\.))', response, re.DOTALL)
        if design_match:
            design = design_match.group(1).strip()
            
        complexity_match = re.search(r'(?:COMPLEXITY ANALYSIS|Complexity Analysis|3\.)[^\n]*\n(.*?)(?=(?:IMPLEMENTATION|Implementation|4\.))', response, re.DOTALL)
        if complexity_match:
            complexity = complexity_match.group(1).strip()
            
        implementation_match = re.search(r'(?:IMPLEMENTATION|Implementation|4\.)[^\n]*\n(.*?)(?=(?:TESTING|Testing|EDGE CASES|Edge Cases|5\.))', response, re.DOTALL)
        if implementation_match:
            implementation = implementation_match.group(1).strip()
            
        testing_match = re.search(r'(?:TESTING|Testing|EDGE CASES|Edge Cases|5\.)[^\n]*\n(.*?)(?=$|\n\n)', response, re.DOTALL)
        if testing_match:
            testing = testing_match.group(1).strip()
            
        return {
            "analysis": analysis,
            "design": design,
            "complexity": complexity,
            "implementation": implementation,
            "testing": testing,
            "full_response": response
        }
    
    def evaluate_reasoning(self, parsed_response: Dict[str, str]) -> float:
        """Evaluate the quality of programming and algorithm reasoning."""
        score = 0.0
        
        # Check for presence of all required components
        if parsed_response.get("analysis"):
            score += 0.15
        if parsed_response.get("design"):
            score += 0.20
        if parsed_response.get("complexity"):
            score += 0.15
        if parsed_response.get("implementation"):
            score += 0.30
        if parsed_response.get("testing"):
            score += 0.15
            
        # Check for code blocks in implementation
        implementation = parsed_response.get("implementation", "")
        if re.search(r'```[a-z]*\n[\s\S]*?\n```', implementation):
            score += 0.15
            
        # Check for complexity analysis using Big O notation
        complexity = parsed_response.get("complexity", "")
        if re.search(r'O\([^\)]+\)', complexity):
            score += 0.10
            
        # Check for edge cases in testing section
        testing = parsed_response.get("testing", "")
        if re.search(r'edge case|corner case|special case|boundary|limit|empty|null|exception|error', testing, re.IGNORECASE):
            score += 0.10
            
        return min(1.0, score)


# Dictionary of all available templates
AVAILABLE_TEMPLATES = {
    "general": GeneralAnalyticalTemplate(),
    "math": MathematicalTemplate(),
    "scientific": ScientificTemplate(),
    "ethical": EthicalTemplate(),
    "programming": ProgrammingTemplate()
}


def detect_template(question: str) -> ReasoningTemplate:
    """
    Automatically detect the most appropriate reasoning template for a given question.
    
    This function evaluates the question against all available templates and selects
    the one with the highest confidence score.
    
    Args:
        question: The input question to analyze
        
    Returns:
        The most appropriate ReasoningTemplate instance for the question
    """
    best_template = None
    best_score = -1.0
    
    # Test each template and find the one with the highest match score
    for template_name, template in AVAILABLE_TEMPLATES.items():
        match_score = template.matches(question)
        if match_score > best_score:
            best_score = match_score
            best_template = template
    
    # Default to general analytical template if no good match is found
    if best_template is None:
        return AVAILABLE_TEMPLATES["general"]
    
    return best_template


def get_template_by_name(template_name: str) -> ReasoningTemplate:
    """
    Retrieve a template by its name.
    
    Args:
        template_name: The name of the template to retrieve
        
    Returns:
        The requested ReasoningTemplate instance
        
    Raises:
        ValueError: If the requested template name doesn't exist
    """
    template_name = template_name.lower()
    
    # Check if the template name exactly matches an available template
    if template_name in AVAILABLE_TEMPLATES:
        return AVAILABLE_TEMPLATES[template_name]
    
    # Try to find a partial match
    for name, template in AVAILABLE_TEMPLATES.items():
        if template_name in name.lower() or template_name in template.name.lower():
            return template
    
    # If no match is found, raise an error
    available_names = ", ".join(AVAILABLE_TEMPLATES.keys())
    raise ValueError(f"Template '{template_name}' not found. Available templates: {available_names}")


def list_available_templates() -> List[Dict[str, str]]:
    """
    List all available reasoning templates.
    
    Returns:
        List of dictionaries containing template names and descriptions
    """
    return [
        {"name": template.name, "id": name, "description": template.description}
        for name, template in AVAILABLE_TEMPLATES.items()
    ]


def get_template_details(template_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific template.
    
    Args:
        template_name: The name of the template to get details for
        
    Returns:
        Dictionary containing template details
    """
    template = get_template_by_name(template_name)
    
    return {
        "name": template.name,
        "description": template.description,
        "example_prompt": template.format_prompt("Example question"),
        "parsing_capabilities": "Can extract structured components from model responses",
        "evaluation_capabilities": "Can assess reasoning quality specific to this domain"
    }

