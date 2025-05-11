# question_generator.py
import ollama
import re
import torch
from typing import List, Dict, Any

class QuestionGenerator:
    def __init__(self, hypergraph, embeddings):
        self.hypergraph = hypergraph
        self.embeddings = embeddings
        self.node_list = sorted(hypergraph.nodes)
        
    def generate(self, num_questions=5, difficulty="medium") -> List[Dict[str, Any]]:
        """
        Generate context-aware multiple choice questions
        Args:
            num_questions: Number of questions to generate
            difficulty: Question difficulty level (easy/medium/hard)
        Returns:
            List of question dictionaries with options and explanations
        """
        context = self._build_context()
        prompt = self._build_prompt(context, num_questions, difficulty)
        response = self._query_llm(prompt)
        return self._parse_questions(response)

    def _build_context(self) -> str:
        """Construct context string from hypergraph and embeddings"""
        context = []
        for edge_id in self.hypergraph.edges:
            members = list(self.hypergraph.edges[edge_id])
            indices = [self.node_list.index(node) for node in members]
            avg_embedding = torch.mean(self.embeddings[indices], dim=0).tolist()
            
            context.append(
                f"Topic {edge_id}:\n"
                f"- Concepts: {', '.join(members)}\n"
                f"- Relationship Strength: {avg_embedding[:3]}...\n"
            )
        return "\n".join(context)

    def _build_prompt(self, context: str, num_questions: int, difficulty: str) -> str:
        """Create LLM prompt with strict formatting rules"""
        guidelines = {
            'easy': "Focus on single-concept recall and basic definitions",
            'medium': "Compare related concepts and their applications",
            'hard': "Analyze complex interactions between multiple concepts"
        }[difficulty]
        
        return f"""
        Generate {num_questions} {difficulty}-difficulty multiple choice questions using this context:
        
        {context}
        
        Requirements:
        1. Follow these difficulty guidelines: {guidelines}
        2. Use EXACT concept names from the context
        3. Include 4 options (A-D) per question
        4. Mark correct answer with letter only
        5. Add brief explanation referencing context
        6. Never include difficulty labels in questions
        
        Format EXACTLY like:
        
        Question: [Full question text]
        A. [Option 1]
        B. [Option 2]
        C. [Option 3]
        D. [Option 4]
        Correct Answer: [Letter]
        Explanation: [1-2 sentence rationale]
        
        Example question:
        Question: Which algorithm handles sequential data most effectively?
        A. Random Forest
        B. K-Means
        C. RNN
        D. SVM
        Correct Answer: C
        Explanation: RNNs process sequential data through recurrent connections.

        Important Rules:
        1. Do NOT include difficulty levels like '(Easy)' in question text
        2. Never put difficulty labels after question numbers
        3. Keep questions focused on content, not difficulty
        
        Generate EXACTLY {num_questions} questions:
        """

    def _query_llm(self, prompt: str) -> str:
        """Query Ollama API with error handling"""
        try:
            response = ollama.chat(
                model='zephyr',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7, 'num_ctx': 4096}
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            return ""

    def _parse_questions(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse and validate LLM response"""
        questions = []
        current_q = {
            'question': '',
            'options': {},
            'correct': None,
            'explanation': ''
        }
        
        lines = raw_text.split('\n')
        line_idx = 0
        
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            
            # Detect question start
            if re.match(r'^Question\s*\d*:?', line, re.IGNORECASE):
                if current_q['question']:
                    questions.append(current_q)
                    current_q = {
                        'question': '',
                        'options': {},
                        'correct': None,
                        'explanation': ''
                    }
                
                # Extract question text
                question_text = re.sub(r'^Question\s*\d*:?\s*', '', line, flags=re.IGNORECASE)
                current_q['question'] = question_text.strip()
                line_idx += 1
                continue
                
            # Parse options
            option_match = re.match(r'^([A-D])[\.\)]\s*(.+)', line)
            if option_match:
                option_key = option_match.group(1).upper()
                option_text = option_match.group(2).strip()
                current_q['options'][option_key] = option_text
                line_idx += 1
                continue
                
            # Parse correct answer
            if re.match(r'^Correct Answer:', line, re.IGNORECASE):
                answer = line.split(':')[-1].strip().upper()
                if len(answer) >= 1:
                    current_q['correct'] = answer[0]
                line_idx += 1
                continue
                
            # Parse explanation (handle multi-line)
            if re.match(r'^Explanation:', line, re.IGNORECASE):
                explanation = line.split(':', 1)[-1].strip()
                # Collect continuation lines
                while (line_idx + 1 < len(lines) and 
                       not re.match(r'^(Question|Correct Answer|Explanation)', lines[line_idx+1], re.IGNORECASE)):
                    line_idx += 1
                    explanation += " " + lines[line_idx].strip()
                current_q['explanation'] = explanation
                line_idx += 1
                continue
                
            line_idx += 1  # Skip unrecognized lines
            
        if current_q['question']:
            questions.append(current_q)
            
        return [q for q in questions if self._validate_question(q)]

    def _validate_question(self, question: Dict) -> bool:
        """Validate question structure and content"""
        # Check required fields
        if not all([question['question'], question['options'], question['correct'], question['explanation']]):
            return False
            
        # Check at least 2 options
        if len(question['options']) < 2:
            return False
            
        # Check correct answer exists in options
        if question['correct'] not in question['options']:
            return False
            
        # Check question uses hypergraph terms
        valid_terms = set(self.node_list)
        question_text = question['question'].lower()
        if not any(term.lower() in question_text for term in valid_terms):
            return False
            
        return True
