import json
import os
import re
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI  
from dotenv import load_dotenv

load_dotenv()

class CodeDetector:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_API")
        )
        self.patterns = {
            'generic_vars': re.compile(r'\b(int|char|float|double|void|string|var|let|const)\s+(\w+)'),
            'c_error_handling': re.compile(r'if\s*\(.*==\s*NULL\)|perror\('),
            'comments': re.compile(r'/\*.*?\*/|//.*?$', re.DOTALL | re.MULTILINE),
            'c_pointers': re.compile(r'\*\w+\s*=\s*\w+;'),
            'c_memory': re.compile(r'malloc|free|calloc|realloc'),
            'c_main': re.compile(r'int\s+main\s*\(.*\)\s*\{.*return\s+0;.*\}', re.DOTALL),
            'simple_loops': re.compile(r'for\s*\([^;]*;[^;]*;[^)]*\)'),
            'behavioral_anomalies': re.compile(r'\b(function|process|handle|execute|calculate)\s*\('),
            'null_checks': re.compile(r'if\s*\(.*\s*==\s*NULL\)'),
            'else_checks': re.compile(r'\belse\b'),
            'loop_structures': re.compile(r'\bfor\b.*\b{'),
            'repetitive_patterns': re.compile(r'(for|while|if)\s*\(.*\)\s*\{.*\}'),
            'standard_headers': re.compile(r'#include\s*<(stdio|stdlib|string|math)\.h>'),
            'common_c_idioms': re.compile(r'strcpy|strcmp|printf|scanf|fopen|fclose'),
            'ai_specific_patterns': re.compile(r'//\s*(Generated|Created)\s*by|/\*.*AI.*generated.*\*/', re.IGNORECASE)
        }
    
    async def detect(self, code: str, language: str = "c", threshold: float = 70.0) -> Dict[str, Any]:
        """
        Enhanced detection method with multiple analysis techniques
        """
        clean_code = self._preprocess_code(code)
        
        pattern_analysis = self._analyze_c_code_patterns(clean_code)
        llm_analysis = await self._analyze_with_llm(code, language)
        
        style_analysis = self._analyze_coding_style(code)
        structure_analysis = self._analyze_code_structure(clean_code)
        
        combined_confidence = self._combine_confidence(
            pattern_analysis.get("confidence", 0),
            llm_analysis.get("confidence", 0),
            style_analysis.get("confidence", 0),
            structure_analysis.get("confidence", 0)
        )
        
        is_ai = combined_confidence >= threshold
        suspicious_sections = self._detect_behavioral_anomalies(clean_code)
        
        return {
            "is_ai_generated": is_ai,
            "confidence": round(combined_confidence, 2),
            "language": language,
            "indicators": (
                llm_analysis.get("key_indicators", []) +
                pattern_analysis.get("indicators", []) +
                style_analysis.get("indicators", []) +
                structure_analysis.get("indicators", [])
            ),
            "suspicious_sections": suspicious_sections,
            "model_used": "deepseek/deepseek-r1-distill-llama-70b:free",
            "warnings": self._generate_warnings(code, language),
            "details": {
                "pattern_analysis": pattern_analysis,
                "llm_analysis": llm_analysis,
                "style_analysis": style_analysis,
                "structure_analysis": structure_analysis
            }
        }
    
    def _preprocess_code(self, code: str) -> str:
        """Clean and normalize code for analysis"""
        code = self.patterns['comments'].sub('', code)
        code = '\n'.join(line.strip() for line in code.splitlines() if line.strip())
        return code
    
    async def _analyze_with_llm(self, code: str, language: str) -> Dict[str, Any]:
        """Enhanced LLM analysis with more specific prompting"""
        prompt = self._build_llm_prompt(code, language)
        
        try:
            completion = await self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "CodeDetector",
                },
                model="deepseek/deepseek-r1-distill-llama-70b:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            if not completion.choices or not completion.choices[0].message.content:
                raise ValueError("Empty response from LLM")
                
            response_content = completion.choices[0].message.content
            # print("Raw LLM response:", response_content)
            try:
                return self._parse_llm_response(response_content)
            except ValueError as e:
                raise ValueError(f"Failed to parse LLM response: {e}")
                
        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
            return {
                "error": str(e),
                "confidence": 0,
                "key_indicators": ["LLM analysis failed"],
                "suspicious_sections": []
            }

    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse and validate the LLM response, handling cases with explanatory text"""
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in LLM response")
        
        try:
            json_str = response_content[json_start:json_end]
            response = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        
        required_fields = ["confidence", "key_indicators"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field in LLM response: {field}")
        
        confidence = float(response.get("confidence", 0))
        confidence = max(0, min(100, confidence))
        
        return {
            "confidence": confidence,
            "key_indicators": response.get("key_indicators", []),
            "suspicious_sections": response.get("suspicious_sections", []),
            "analysis_summary": response.get("analysis_summary", ""),
            "pattern_details": response.get("pattern_details", {})
        }
            
    def _build_llm_prompt(self, code: str, language: str) -> str:
        """Construct a more detailed prompt for LLM analysis"""
        return f"""
        As an expert in AI-generated code detection, analyze this {language} code for signs of AI generation.
        Focus on the aspects listed below and provide your analysis in the exact JSON format specified.
        DO NOT include any explanatory text outside the JSON structure.

        Analysis aspects:
        1. CODE STRUCTURE (30% weight):
           - Overly consistent indentation and formatting
           - Predictable function structures
           - Missing or generic error handling
           - Overuse of common idioms

        2. VARIABLE & FUNCTION NAMES (20% weight):
           - Generic names (temp, data, value, func)
           - Lack of domain-specific terminology
           - Inconsistent naming schemes
           - Overuse of simple data types

        3. COMMENTS & DOCUMENTATION (15% weight):
           - Missing or overly verbose comments
           - Comments that state the obvious
           - Lack of "why" explanations
           - AI-generated signature patterns

        4. C-SPECIFIC PATTERNS (25% weight):
           - Basic pointer usage without proper checks
           - Standard main() boilerplate
           - Simple memory management patterns
           - Overuse of common C library functions
           - Lack of advanced C features

        5. COMPLEXITY & ORIGINALITY (10% weight):
           - Lack of creative solutions
           - Overly simplistic implementations
           - Missing edge cases
           - Formulaic problem-solving approaches

        Required JSON format:
        {{
            "confidence": 0-100,
            "key_indicators": ["list", "of", "specific", "indicators"],
            "suspicious_sections": ["specific", "code", "snippets"],
            "analysis_summary": "detailed text summary",
            "pattern_details": {{
                "structure_issues": ["list"],
                "naming_issues": ["list"],
                "comment_issues": ["list"],
                "c_specific_issues": ["list"]
            }}
        }}

        Code to analyze:
        ```{language}
        {code}
        ```
        """
    
    def _analyze_c_code_patterns(self, code: str) -> Dict[str, Any]:
        """Enhanced C-specific pattern analysis"""
        indicators = []
        confidence = 0
        
        var_analysis = self._analyze_variable_names(code)
        indicators.extend(var_analysis["indicators"])
        confidence += var_analysis["confidence"]
        
        mem_analysis = self._analyze_memory_management(code)
        indicators.extend(mem_analysis["indicators"])
        confidence += mem_analysis["confidence"]
        
        flow_analysis = self._analyze_control_flow(code)
        indicators.extend(flow_analysis["indicators"])
        confidence += flow_analysis["confidence"]
        
        std_analysis = self._analyze_standard_patterns(code)
        indicators.extend(std_analysis["indicators"])
        confidence += std_analysis["confidence"]
        
        ai_patterns = self._detect_ai_patterns(code)
        indicators.extend(ai_patterns["indicators"])
        confidence += ai_patterns["confidence"]
        
        return {
            "indicators": indicators,
            "confidence": min(confidence, 100),
            "pattern_details": {
                "variable_analysis": var_analysis,
                "memory_analysis": mem_analysis,
                "control_flow": flow_analysis,
                "standard_patterns": std_analysis,
                "ai_specific": ai_patterns
            }
        }
    
    def _analyze_variable_names(self, code: str) -> Dict[str, Any]:
        """Detailed analysis of variable naming patterns"""
        indicators = []
        confidence = 0
        generic_names = {'data', 'temp', 'result', 'value', 'var', 'ptr', 'num', 'obj', 
                        'i', 'j', 'k', 'x', 'y', 'z', 'n', 'count', 'size', 'len'}
        
        matches = self.patterns['generic_vars'].finditer(code)
        generic_count = 0
        total_vars = 0
        
        for match in matches:
            total_vars += 1
            var_name = match.group(2)
            if var_name in generic_names:
                generic_count += 1
        
        if total_vars > 0:
            generic_ratio = generic_count / total_vars
            if generic_ratio > 0.5:
                indicators.append(f"High ratio of generic variable names ({generic_ratio:.0%})")
                confidence += 15
            if generic_ratio > 0.7:
                confidence += 10
        
        if (re.search(r'\b[a-z][a-z0-9_]*\b', code) and 
            re.search(r'\b[A-Z][A-Za-z0-9_]*\b', code) and
            re.search(r'\b[a-z]+_[a-z]+\b', code)):
            indicators.append("Inconsistent naming conventions (mixed snake_case, camelCase)")
            confidence += 10
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "generic_vars_ratio": generic_count / total_vars if total_vars > 0 else 0,
            "total_vars": total_vars
        }
    
    def _analyze_memory_management(self, code: str) -> Dict[str, Any]:
        """Analyze memory management patterns in C code"""
        indicators = []
        confidence = 0
        
        pointer_uses = len(self.patterns['c_pointers'].findall(code))
        mem_ops = len(self.patterns['c_memory'].findall(code))
        
        if pointer_uses > 2 and mem_ops < pointer_uses / 2:
            indicators.append(f"Pointer-heavy code ({pointer_uses} uses) with insufficient memory management ({mem_ops} ops)")
            confidence += 20
        
        malloc_matches = re.finditer(r'(\w+\s*=\s*malloc\s*\(.*\))', code)
        for match in malloc_matches:
            var_name = match.group(1).split('=')[0].strip()
            null_check = re.compile(f'if\\s*\\(\\s*{var_name}\\s*==\\s*NULL\\s*\\)')
            if not null_check.search(code):
                indicators.append(f"malloc without NULL check for {var_name}")
                confidence += 5
        
        if mem_ops > 0 and not re.search(r'free\s*\(', code):
            indicators.append("Memory allocation without corresponding free()")
            confidence += 10
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "pointer_uses": pointer_uses,
            "memory_operations": mem_ops
        }
    
    def _analyze_control_flow(self, code: str) -> Dict[str, Any]:
        """Analyze control flow patterns"""
        indicators = []
        confidence = 0
        
        if_count = len(re.findall(r'\bif\s*\(', code))
        loop_count = len(self.patterns['simple_loops'].findall(code))
        switch_count = len(re.findall(r'\bswitch\s*\(', code))
        func_count = len(re.findall(r'\b\w+\s+\w+\s*\(.*\)\s*\{', code))
        
        if if_count > 5 and loop_count > 3 and switch_count == 0 and func_count < 3:
            indicators.append("Overuse of basic control structures with few functions")
            confidence += 15
        
        max_nesting = self._calculate_max_nesting(code)
        if max_nesting < 2:
            indicators.append("Shallow control flow (max nesting < 2)")
            confidence += 10
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "control_flow_stats": {
                "if_statements": if_count,
                "loops": loop_count,
                "switches": switch_count,
                "functions": func_count,
                "max_nesting": max_nesting
            }
        }
    
    def _calculate_max_nesting(self, code: str) -> int:
        """Calculate maximum nesting depth in control structures"""
        max_depth = 0
        current_depth = 0
        
        for line in code.splitlines():
            if re.search(r'\b(if|while|for|switch)\s*\(', line):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif '}' in line:
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _analyze_standard_patterns(self, code: str) -> Dict[str, Any]:
        """Analyze use of standard C patterns and idioms"""
        indicators = []
        confidence = 0
        
        if self.patterns['c_main'].search(code):
            indicators.append("Standard main() boilerplate with return 0")
            confidence += 10
        
        common_idioms = self.patterns['common_c_idioms'].findall(code)
        if len(common_idioms) > 5:
            indicators.append(f"Overuse of common C idioms ({len(common_idioms)} instances)")
            confidence += 10
        
        headers = self.patterns['standard_headers'].findall(code)
        if len(headers) >= 3 and len(headers) == len(re.findall(r'#include\s*<.*?>', code)):
            indicators.append("Only standard headers used")
            confidence += 5
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "common_idioms_count": len(common_idioms),
            "standard_headers": headers
        }
    
    def _detect_ai_patterns(self, code: str) -> Dict[str, Any]:
        """Detect patterns specifically common in AI-generated code"""
        indicators = []
        confidence = 0
        
        if self.patterns['ai_specific_patterns'].search(code):
            indicators.append("AI signature detected in comments")
            confidence += 30
        
        lines = code.splitlines()
        if len(lines) > 10:
            indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            if len(set(indent_levels)) <= 2:
                indicators.append("Overly consistent indentation")
                confidence += 10
        
        error_handlers = re.findall(r'if\s*\(.*\)\s*\{.*(printf|perror)\s*\(.*\).*\}', code)
        if len(error_handlers) > 2 and len(set(error_handlers)) < 2:
            indicators.append("Repetitive error handling patterns")
            confidence += 15
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "ai_signature_found": 'ai_specific_patterns' in indicators
        }
    
    def _analyze_coding_style(self, code: str) -> Dict[str, Any]:
        """Analyze coding style characteristics"""
        indicators = []
        confidence = 0
        
        inconsistent_spacing = False
        operators = re.finditer(r'[+\-*/%=!<>]=?|&&|\|\|', code)
        spaces_before = []
        spaces_after = []
        
        for op in operators:
            before = code[op.start()-1] if op.start() > 0 else ''
            after = code[op.end()] if op.end() < len(code) else ''
            spaces_before.append(before.isspace())
            spaces_after.append(after.isspace())
        
        if len(spaces_before) > 3:
            consistent_before = all(spaces_before) or not any(spaces_before)
            consistent_after = all(spaces_after) or not any(spaces_after)
            
            if not (consistent_before and consistent_after):
                indicators.append("Inconsistent spacing around operators")
                confidence += 10
                inconsistent_spacing = True
        
        brace_styles = set()
        function_braces = re.finditer(r'\b\w+\s+\w+\s*\(.*\)\s*(\{|$)', code, re.MULTILINE)
        for match in function_braces:
            line = code[:match.end()].splitlines()[-1]
            if '{' in line and not line.strip().endswith('{'):
                brace_styles.add("inline")
            else:
                brace_styles.add("newline")
        
        if len(brace_styles) > 1:
            indicators.append("Inconsistent brace style")
            confidence += 5
        
        if len(code.splitlines()) > 10:
            line_lengths = [len(line) for line in code.splitlines()]
            avg_length = sum(line_lengths) / len(line_lengths)
            dev = sum(abs(l - avg_length) for l in line_lengths) / len(line_lengths)
            
            if dev < 10:
                indicators.append("Overly consistent line lengths")
                confidence += 5
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "style_issues": {
                "inconsistent_spacing": inconsistent_spacing,
                "brace_styles": list(brace_styles),
                "line_length_variation": dev if 'dev' in locals() else None
            }
        }
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze overall code structure"""
        indicators = []
        confidence = 0
        
        functions = re.finditer(r'\b\w+\s+\w+\s*\(.*\)\s*\{', code)
        func_info = []
        for match in functions:
            func_body = self._extract_function_body(code, match.end()-1)
            func_info.append({
                "length": len(func_body.splitlines()),
                "complexity": self._calculate_function_complexity(func_body)
            })
        
        if func_info:
            avg_length = sum(f['length'] for f in func_info) / len(func_info)
            avg_complexity = sum(f['complexity'] for f in func_info) / len(func_info)
            
            if avg_length < 10:
                indicators.append("Short average function length")
                confidence += 5
            if avg_complexity < 2:
                indicators.append("Low average function complexity")
                confidence += 10
        
        if len(func_info) < 3 and len(code.splitlines()) > 30:
            indicators.append("Low modularity (few functions for code size)")
            confidence += 15
        
        return {
            "indicators": indicators,
            "confidence": confidence,
            "structure_metrics": {
                "function_count": len(func_info),
                "avg_function_length": avg_length if func_info else None,
                "avg_function_complexity": avg_complexity if func_info else None
            }
        }
    
    def _extract_function_body(self, code: str, start_pos: int) -> str:
        """Extract a function body from code"""
        brace_count = 1
        pos = start_pos + 1
        end_pos = start_pos
        
        while pos < len(code) and brace_count > 0:
            if code[pos] == '{':
                brace_count += 1
            elif code[pos] == '}':
                brace_count -= 1
            pos += 1
        
        return code[start_pos:pos]
    
    def _calculate_function_complexity(self, func_body: str) -> int:
        """Calculate simple complexity metric for a function"""
        complexity = 0
        complexity += len(re.findall(r'\bif\s*\(', func_body))
        complexity += len(re.findall(r'\bfor\s*\(', func_body))
        complexity += len(re.findall(r'\bwhile\s*\(', func_body))
        complexity += len(re.findall(r'\bswitch\s*\(', func_body))
        complexity += len(re.findall(r'\bcase\b', func_body))
        
        return complexity
    
    def _combine_confidence(self, *confidences: float) -> float:
        """Combine multiple confidence scores with weighted average"""
        weights = [0.4, 0.5, 0.2, 0.2]  
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        return min(100, weighted_sum * 1.1)  
    
    def _generate_warnings(self, code: str, language: str) -> List[str]:
        """Generate warnings about the code"""
        warnings = []
        
        if len(code.splitlines()) < 15:
            warnings.append("Short code samples (<15 lines) are harder to analyze accurately")
        
        if language == "c":
            if not self.patterns['c_memory'].search(code):
                warnings.append("No memory management functions found")
            if not re.search(r'#include\s*<', code):
                warnings.append("No standard library includes found")
        
        return warnings
    
    def _detect_behavioral_anomalies(self, code: str) -> List[str]:
        """
        Detect potential AI-specific behavioral anomalies with more patterns
        """
        anomalies = []
        
        type_counts = {}
        type_matches = re.finditer(r'\b(int|char|float|double|void)\s+', code)
        for match in type_matches:
            type_name = match.group(1)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        if sum(type_counts.values()) > 5 and max(type_counts.values(), default=0) / sum(type_counts.values()) > 0.6:
            anomalies.append("Overuse of basic data types")
        
        if not self.patterns['null_checks'].search(code) and not self.patterns['else_checks'].search(code):
            anomalies.append("No error handling or conditional logic detected")
        
        func_calls = re.findall(r'\b(\w+)\s*\(', code)
        func_counts = {}
        for call in func_calls:
            if call not in ['if', 'while', 'for', 'switch']:
                func_counts[call] = func_counts.get(call, 0) + 1
        
        if len(func_counts) > 0 and max(func_counts.values()) / len(code.splitlines()) > 0.3:
            anomalies.append("Repetitive function call patterns")
        
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        if len(lines) > 10:
            first_chars = [line[0] for line in lines if line]
            if len(set(first_chars)) < 3:
                anomalies.append("Overly consistent line starting patterns")
        
        return anomalies