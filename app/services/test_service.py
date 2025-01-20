# app/services/test_service.py
from typing import Dict, List, Optional
import logging
from openai import OpenAI
import os
import uuid
from sqlalchemy.orm import Session
from ..schemas.test_schemas import (
    TestConfig, 
    TestSubmission,
    TestResult,
    Question,
    QuestionType,
    Test,
    SWOTAnalysis
)
from ..models.tests_models import DBTest, DBQuestion, DBTestResult
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TestService:
    def __init__(self, db: Session):
        self.client = client
        self.db = db

    def _convert_to_db_model(self, test: Test) -> DBTest:
        """Convert domain model to database model"""
        db_questions = []
        for q in test.questions:
            db_question = DBQuestion(
                id=q.id,
                question_text=q.question,
                question_type=q.type,
                topic=q.topic,
                points=q.points,
                options=q.options,
                correct_answer=q.correct_answer,
                rubric=q.rubric
            )
            db_questions.append(db_question)

        return DBTest(
            id=test.id,
            subject=test.config.subject,
            grade=test.config.grade,
            complexity=test.config.complexity,
            question_type=test.config.questionType,
            time_limit=test.config.timeLimit,
            content=test.content,
            questions=db_questions
        )

    def _convert_from_db_model(self, db_test: DBTest) -> Test:
        """Convert database model to domain model"""
        questions = []
        for q in db_test.questions:
            question = Question(
                id=q.id,
                question=q.question_text,
                type=q.question_type,
                topic=q.topic,
                points=q.points,
                options=q.options,
                correct_answer=q.correct_answer,
                rubric=q.rubric
            )
            questions.append(question)

        config = TestConfig(
            subject=db_test.subject,
            grade=db_test.grade,
            complexity=db_test.complexity,
            questionType=db_test.question_type,
            timeLimit=db_test.time_limit
        )

        return Test(
            id=db_test.id,
            config=config,
            questions=questions,
            content=db_test.content
        )


    def generate_mcq_format(self) -> str:
        return """Format each MCQ as follows:
1. [Question]
   Topic: [Specific topic within subject]
   a) [Option A]
   b) [Option B]
   c) [Option C]
   d) [Option D]

Include an ANSWER KEY at the end:
1. [Correct Answer]
[and so on...]"""

    def generate_crq_format(self) -> str:
        return """Format each Constructed Response Question as follows:
1. [Question]
   Topic: [Specific topic within subject]
   Points: [point value]
   Expected Response Length: [short/medium length]
   
   Rubric:
   - [Key point 1] (X points)
   - [Key point 2] (X points)
   - [Key point 3] (X points)"""

    def generate_erq_format(self) -> str:
        return """Format each Extended Response Question as follows:
1. [Question]
   Topic: [Specific topic within subject]
   Points: [point value]
   Expected Response Length: [detailed paragraph/essay]
   
   Detailed Rubric:
   Content (X points):
   - [Critical point 1]
   - [Critical point 2]
   - [Critical point 3]
   
   Organization (X points):
   - [Organization criteria]
   
   Analysis (X points):
   - [Analysis criteria]"""

    def generate_prompt(self, config: TestConfig) -> str:
        """Generate a structured prompt for the LLM based on test configuration."""
        
        question_counts = {
            "mcq": 10,
            "crq": 5,
            "erq": 3
        }
        num_questions = question_counts.get(config.questionType.lower(), 10)
        
        format_template = {
            "mcq": self.generate_mcq_format(),
            "crq": self.generate_crq_format(),
            "erq": self.generate_erq_format()
        }.get(config.questionType.lower(), self.generate_mcq_format())

        prompt = f"""Generate a {config.complexity} level {config.subject} test for grade {config.grade}.

Test Requirements:
- Include {num_questions} {config.questionType.upper()} questions
- Should be completable within {config.timeLimit} minutes
- Must be appropriate for grade {config.grade} level
- Questions should progress from easier to more challenging
- Each question should be clearly numbered
- Include specific topic tags for each question

{format_template}

Additional Guidelines:
- For math questions, include step-by-step solutions in the answer key
- For science questions, ensure explanations reference relevant concepts
- For English questions, include text references where applicable
- Ensure questions test different cognitive levels (knowledge, application, analysis)

Please structure the test clearly with:
1. Clear instructions at the beginning
2. Questions in the specified format
3. Answer key/rubric at the end"""

        return prompt

    def _parse_test_content(self, content: str, config: TestConfig) -> List[Question]:
        """Parse the generated content into structured questions."""
        questions = []
        current_question = None
        current_section = "questions"  # Can be "questions" or "answer_key"
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section changes
            if "ANSWER KEY" in line.upper():
                current_section = "answer_key"
                continue
            elif "RUBRIC" in line.upper():
                current_section = "rubric"
                continue

            # Handle different sections
            if current_section == "questions":
                # New question starts with a number
                if line[0].isdigit() and line[1] == '.':
                    if current_question:
                        questions.append(current_question)
                    
                    current_question = Question(
                        id=str(len(questions) + 1),
                        question=line[2:].strip(),
                        type=config.questionType,
                        topic="",
                        points=1,
                        options=[],
                        rubric=[]
                    )
                
                elif current_question:
                    if line.startswith('Topic:'):
                        current_question.topic = line[6:].strip()
                    elif line.startswith(('a)', 'b)', 'c)', 'd)')):
                        current_question.options.append(line.strip())
                    elif line.startswith('Points:'):
                        try:
                            current_question.points = int(line[7:].strip().split()[0])
                        except:
                            current_question.points = 1
                    elif line.startswith('-'):
                        current_question.rubric.append(line[1:].strip())
            
            elif current_section == "answer_key":
                if current_question and line[0].isdigit():
                    answer_num = int(line[0])
                    if 0 < answer_num <= len(questions):
                        questions[answer_num-1].correct_answer = line[line.find(')')+1:].strip()

        # Add the last question
        if current_question:
            questions.append(current_question)

        return questions

    async def generate_test_with_llm(self, config: TestConfig) -> str:
        """Generate test using OpenAI's GPT model and store in database."""
        try:
            prompt = self.generate_prompt(config)
            logger.info(f"Generated prompt: {prompt}")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an experienced education professional who creates high-quality academic tests."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content
            questions = self._parse_test_content(content, config)
            
            # Create and store test
            test_id = f"test_{uuid.uuid4().hex[:8]}"
            test = Test(
                id=test_id,
                config=config,
                questions=questions,
                content=content
            )
            
            # Convert to DB model and store
            db_test = self._convert_to_db_model(test)
            self.db.add(db_test)
            self.db.commit()
            
            return content

        except Exception as e:
            logger.error(f"Error in generate_test_with_llm: {str(e)}")
            self.db.rollback()
            raise

    async def evaluate_written_answer(self, answer: str, question: Question) -> float:
        """Evaluate a written answer using GPT."""
        try:
            rubric_text = "\n".join(question.rubric)
            prompt = f"""Evaluate this answer based on the following rubric:
Rubric criteria:
{rubric_text}

Maximum points: {question.points}

Student answer:
{answer}

Provide only a numeric score out of {question.points} based on how well the answer meets the rubric criteria."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an experienced teacher evaluating student answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            try:
                score = float(response.choices[0].message.content.strip())
                return min(max(0, score), question.points)
            except ValueError:
                logger.error("Could not parse score from GPT response")
                return 0

        except Exception as e:
            logger.error(f"Error evaluating written answer: {str(e)}")
            return 0



    def _generate_feedback(self, analysis: SWOTAnalysis) -> str:
        """Generate personalized feedback based on SWOT analysis."""
        feedback_parts = ["Based on your performance:"]
        
        if analysis.strengths:
            feedback_parts.append("\nStrong Areas:")
            feedback_parts.extend(f"- {strength}" for strength in analysis.strengths)
        
        if analysis.weaknesses:
            feedback_parts.append("\nAreas for Improvement:")
            feedback_parts.extend(f"- {weakness}" for weakness in analysis.weaknesses)
        
        if analysis.opportunities:
            feedback_parts.append("\nRecommended Next Steps:")
            feedback_parts.extend(f"- {opportunity}" for opportunity in analysis.opportunities)
        
        return "\n".join(feedback_parts)
    async def generate_llm_analysis(self, test: Test, submission: TestSubmission, topic_performance: Dict[str, Dict[str, float]], score: float) -> dict:
        """Generate detailed LLM analysis of test performance."""
        try:
            # Prepare detailed context for LLM
            question_analysis = []
            for question in test.questions:
                answer = submission.answers.get(str(question.id))
                if answer:
                    if question.type == "mcq":
                        is_correct = answer == question.correct_answer
                        question_analysis.append({
                            "topic": question.topic,
                            "question": question.question,
                            "student_answer": answer,
                            "correct_answer": question.correct_answer,
                            "is_correct": is_correct
                        })
                    else:
                        # For CRQ/ERQ, include rubric points
                        question_analysis.append({
                            "topic": question.topic,
                            "question": question.question,
                            "student_answer": answer,
                            "rubric": question.rubric
                        })

            # Create comprehensive analysis prompt
            analysis_prompt = f"""As an educational expert, analyze this test performance:

    Test Details:
    - Subject: {test.config.subject}
    - Grade Level: {test.config.grade}
    - Question Type: {test.config.questionType}
    - Overall Score: {score:.1f}%

    Topic Performance:
    {self._format_topic_performance(topic_performance)}

    Question-by-Question Analysis:
    {self._format_question_analysis(question_analysis)}

    Based on this comprehensive data, provide:

    1. STRENGTHS (List specific concepts mastered and positive patterns):
    2. WEAKNESSES (Identify specific gaps in understanding and areas needing immediate attention):
    3. OPPORTUNITIES (Suggest specific learning strategies and resources for improvement):
    4. THREATS (Highlight potential challenges if weaknesses aren't addressed):
    5. DETAILED RECOMMENDATIONS:
      - Study strategies
      - Practice areas
      - Resource suggestions
      - Next steps for improvement

    Focus on actionable insights and specific patterns in the student's understanding."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational analyst specializing in personalized learning assessment and improvement strategies."
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7
            )

            # Parse the detailed analysis
            analysis_text = response.choices[0].message.content
            return self._parse_llm_analysis(analysis_text)

        except Exception as e:
            logger.error(f"Error generating LLM analysis: {str(e)}")
            return self._generate_fallback_analysis(topic_performance)

    def _format_topic_performance(self, topic_performance: Dict[str, Dict[str, float]]) -> str:
        """Format topic performance for LLM prompt."""
        formatted = []
        for topic, perf in topic_performance.items():
            percentage = (perf["earned"] / perf["total"] * 100) if perf["total"] > 0 else 0
            formatted.append(f"- {topic}: {percentage:.1f}% ({perf['earned']}/{perf['total']} points)")
        return "\n".join(formatted)

    def _format_question_analysis(self, question_analysis: List[dict]) -> str:
        """Format question analysis for LLM prompt."""
        formatted = []
        for idx, qa in enumerate(question_analysis, 1):
            if "is_correct" in qa:  # MCQ
                formatted.append(f"""Q{idx}. Topic: {qa['topic']}
    Question: {qa['question']}
    Student Answer: {qa['student_answer']}
    Correct Answer: {qa['correct_answer']}
    Status: {"Correct" if qa['is_correct'] else "Incorrect"}""")
            else:  # CRQ/ERQ
                formatted.append(f"""Q{idx}. Topic: {qa['topic']}
    Question: {qa['question']}
    Student Answer: {qa['student_answer']}
    Rubric Points:
    {chr(10).join(f"- {point}" for point in qa['rubric'])}""")
        return "\n\n".join(formatted)

    def _parse_llm_analysis(self, analysis_text: str) -> dict:
        """Parse LLM response into structured analysis."""
        sections = analysis_text.split("\n\n")
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
            "recommendations": []
        }
        
        current_section = None
        for section in sections:
            if section.startswith("1. STRENGTHS"):
                current_section = "strengths"
            elif section.startswith("2. WEAKNESSES"):
                current_section = "weaknesses"
            elif section.startswith("3. OPPORTUNITIES"):
                current_section = "opportunities"
            elif section.startswith("4. THREATS"):
                current_section = "threats"
            elif section.startswith("5. DETAILED RECOMMENDATIONS"):
                current_section = "recommendations"
            elif current_section:
                # Extract bullet points
                points = [p.strip("- ").strip() for p in section.split("\n") if p.strip().startswith("-")]
                if points:
                    analysis[current_section].extend(points)

        return analysis

    def _generate_fallback_analysis(self, topic_performance: Dict[str, Dict[str, float]]) -> dict:
        """Generate basic analysis when LLM analysis fails."""
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
            "recommendations": []
        }
        
        for topic, perf in topic_performance.items():
            percentage = (perf["earned"] / perf["total"] * 100) if perf["total"] > 0 else 0
            if percentage >= 70:
                analysis["strengths"].append(f"Strong understanding of {topic} ({percentage:.1f}%)")
            elif percentage <= 40:
                analysis["weaknesses"].append(f"Needs improvement in {topic} ({percentage:.1f}%)")
                analysis["opportunities"].append(f"Focus on strengthening {topic} concepts")
                analysis["threats"].append(f"Weak foundation in {topic} may affect future learning")
        
        return analysis

    async def evaluate_test(self, test_id: str, submission: TestSubmission) -> TestResult:
        """Evaluate a test submission and store results in database."""
        db_test = self.db.query(DBTest).filter(DBTest.id == test_id).first()
        if not db_test:
            raise ValueError("Test not found")

        test = self._convert_from_db_model(db_test)
        
        # Calculate scores and topic performance
        total_points = 0
        earned_points = 0
        topic_performance: Dict[str, Dict[str, float]] = {}

        # Evaluate each question
        for question in test.questions:
            answer = submission.answers.get(str(question.id))
            if not answer:
                continue

            points = question.points
            total_points += points

            # Initialize topic performance tracking
            topic = question.topic or "general"
            if topic not in topic_performance:
                topic_performance[topic] = {"earned": 0, "total": points}
            else:
                topic_performance[topic]["total"] += points

            # Evaluate answer
            if question.type == "mcq":
                if answer == question.correct_answer:
                    earned_points += points
                    topic_performance[topic]["earned"] += points
            else:
                points_earned = await self.evaluate_written_answer(answer, question)
                earned_points += points_earned
                topic_performance[topic]["earned"] += points_earned

        # Calculate overall score
        score = (earned_points / total_points * 100) if total_points > 0 else 0

        # Generate test result
        llm_analysis = await self.generate_llm_analysis(test, submission, topic_performance, score)
        
        # Create SWOT analysis
        analysis = SWOTAnalysis(
            strengths=llm_analysis["strengths"],
            weaknesses=llm_analysis["weaknesses"],
            opportunities=llm_analysis["opportunities"],
            threats=llm_analysis["threats"],
            topic_performance=topic_performance,
            overall_score=score
        )

        # Generate feedback
        feedback = self._generate_detailed_feedback(llm_analysis)

        # Store result in database
        result_id = f"result_{uuid.uuid4().hex[:8]}"
        db_result = DBTestResult(
            id=result_id,
            test_id=test_id,
            score=score,
            topic_performance=topic_performance,
            strengths=analysis.strengths,
            weaknesses=analysis.weaknesses,
            opportunities=analysis.opportunities,
            threats=analysis.threats,
            feedback=feedback
        )
        self.db.add(db_result)
        self.db.commit()

        return TestResult(
            test_id=test_id,
            score=score,
            analysis=analysis,
            feedback=feedback
        )

    def _generate_detailed_feedback(self, llm_analysis: dict) -> str:
        """Generate comprehensive feedback from LLM analysis."""
        sections = [
            ("Performance Analysis", llm_analysis["strengths"] + llm_analysis["weaknesses"]),
            ("Key Recommendations", llm_analysis["recommendations"]),
            ("Improvement Opportunities", llm_analysis["opportunities"]),
            ("Areas Requiring Attention", llm_analysis["threats"])
        ]
        
        feedback_parts = []
        for title, items in sections:
            if items:
                feedback_parts.append(f"\n{title}:")
                feedback_parts.extend(f"- {item}" for item in items)
        
        return "\n".join(feedback_parts)
    
    
    def get_test(self, test_id: str) -> Optional[Test]:
        """Retrieve a test from the database."""
        db_test = self.db.query(DBTest).filter(DBTest.id == test_id).first()
        if db_test:
            return self._convert_from_db_model(db_test)
        return None

    def get_test_result(self, test_id: str) -> Optional[TestResult]:
        """Retrieve test result from the database."""
        db_result = self.db.query(DBTestResult).filter(DBTestResult.test_id == test_id).first()
        if not db_result:
            return None

        analysis = SWOTAnalysis(
            strengths=db_result.strengths,
            weaknesses=db_result.weaknesses,
            opportunities=db_result.opportunities,
            threats=db_result.threats,
            topic_performance=db_result.topic_performance,
            overall_score=db_result.score
        )

        return TestResult(
            test_id=test_id,
            score=db_result.score,
            analysis=analysis,
            feedback=db_result.feedback
        )