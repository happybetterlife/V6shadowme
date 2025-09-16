"""
오류 에이전트 - 언어 오류 분석 및 교정 관리
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class ErrorAgent:
    def __init__(self):
        # 사용자별 오류 히스토리
        self.user_error_history = {}
        
        # 한국어 화자 특성 오류 패턴
        self.korean_speaker_patterns = self.initialize_korean_error_patterns()
        
        # 문법 규칙 데이터베이스
        self.grammar_rules = self.initialize_grammar_rules()
        
        # 어휘 오류 패턴
        self.vocabulary_errors = self.initialize_vocabulary_errors()
        
    async def initialize(self):
        """오류 에이전트 초기화"""
        logger.info("Error Agent initialized")
    
    def initialize_korean_error_patterns(self) -> Dict:
        """한국어 화자 특성 오류 패턴 초기화"""
        return {
            'article_errors': {
                'pattern': r'\b(a|an|the)\b',
                'common_mistakes': [
                    {'error': 'I go to school', 'correction': 'I go to the school'},
                    {'error': 'She is teacher', 'correction': 'She is a teacher'},
                    {'error': 'I have car', 'correction': 'I have a car'}
                ],
                'explanation': 'Korean doesn\'t have articles, so Korean speakers often omit them in English'
            },
            'subject_omission': {
                'pattern': r'^(am|is|are|was|were|have|has|do|does|did)',
                'common_mistakes': [
                    {'error': 'Am student', 'correction': 'I am a student'},
                    {'error': 'Is very good', 'correction': 'It is very good'},
                    {'error': 'Have many books', 'correction': 'I have many books'}
                ],
                'explanation': 'Korean allows subject omission, but English requires explicit subjects'
            },
            'word_order': {
                'pattern': r'\b(\w+ly)\s+(verb)\b',
                'common_mistakes': [
                    {'error': 'I quickly go', 'correction': 'I go quickly'},
                    {'error': 'She carefully reads', 'correction': 'She reads carefully'},
                    {'error': 'We together study', 'correction': 'We study together'}
                ],
                'explanation': 'Korean word order (SOV) differs from English (SVO)'
            },
            'plural_forms': {
                'pattern': r'\b\d+\s+(\w+)(?!s)\b',
                'common_mistakes': [
                    {'error': 'three book', 'correction': 'three books'},
                    {'error': 'many student', 'correction': 'many students'},
                    {'error': 'two car', 'correction': 'two cars'}
                ],
                'explanation': 'Korean doesn\'t have plural forms, leading to omission of -s/-es'
            },
            'preposition_errors': {
                'pattern': r'\b(in|on|at|to|for|with|by)\b',
                'common_mistakes': [
                    {'error': 'arrive to home', 'correction': 'arrive at home'},
                    {'error': 'listen music', 'correction': 'listen to music'},
                    {'error': 'depend of', 'correction': 'depend on'}
                ],
                'explanation': 'Korean prepositions don\'t directly correspond to English ones'
            },
            'verb_tense': {
                'pattern': r'\b(go|goes|went|going)\b',
                'common_mistakes': [
                    {'error': 'I go yesterday', 'correction': 'I went yesterday'},
                    {'error': 'He will goes', 'correction': 'He will go'},
                    {'error': 'I am go', 'correction': 'I am going'}
                ],
                'explanation': 'Korean tense system differs from English, causing confusion'
            }
        }
    
    def initialize_grammar_rules(self) -> Dict:
        """기본 문법 규칙 초기화"""
        return {
            'subject_verb_agreement': {
                'rules': [
                    'Singular subjects take singular verbs (He runs)',
                    'Plural subjects take plural verbs (They run)',
                    'Third person singular uses -s/-es (She likes)'
                ],
                'patterns': [
                    r'\b(I|you|we|they)\s+(am|is)\b',
                    r'\b(he|she|it)\s+(are)\b',
                    r'\b(he|she|it)\s+(\w+)(?<!s)(?<!ed)(?<!ing)\b'
                ]
            },
            'article_usage': {
                'rules': [
                    'Use "a" before consonant sounds',
                    'Use "an" before vowel sounds', 
                    'Use "the" for specific items'
                ],
                'patterns': [
                    r'\ban\s+[bcdfghjklmnpqrstvwxyz]',  # "an" before consonant
                    r'\ba\s+[aeiou]',  # "a" before vowel
                ]
            },
            'prepositions': {
                'rules': [
                    'Use "at" for specific times and places',
                    'Use "in" for months, years, enclosed spaces',
                    'Use "on" for days, dates, surfaces'
                ],
                'common_errors': [
                    {'wrong': 'in Monday', 'correct': 'on Monday'},
                    {'wrong': 'at 2020', 'correct': 'in 2020'},
                    {'wrong': 'on the morning', 'correct': 'in the morning'}
                ]
            }
        }
    
    def initialize_vocabulary_errors(self) -> Dict:
        """어휘 오류 패턴 초기화"""
        return {
            'false_friends': {
                # 한국어-영어 false friends
                'library': {'wrong_usage': 'book store', 'context': 'public place with books'},
                'mansion': {'wrong_usage': 'apartment', 'context': 'large expensive house'},
                'smart': {'wrong_usage': 'fashionable', 'context': 'intelligent'},
            },
            'word_formation': {
                'adjective_noun': [
                    {'wrong': 'beauty place', 'correct': 'beautiful place'},
                    {'wrong': 'danger situation', 'correct': 'dangerous situation'},
                    {'wrong': 'success person', 'correct': 'successful person'}
                ]
            },
            'spelling_patterns': {
                'double_consonants': [
                    {'wrong': 'runing', 'correct': 'running'},
                    {'wrong': 'begining', 'correct': 'beginning'},
                    {'wrong': 'geting', 'correct': 'getting'}
                ],
                'silent_letters': [
                    {'wrong': 'nife', 'correct': 'knife'},
                    {'wrong': 'clim', 'correct': 'climb'},
                    {'wrong': 'numb', 'correct': 'numb'}  # actually correct, but commonly misspelled as 'num'
                ]
            }
        }
    
    async def analyze(self, text: str, context: Dict = None) -> List[Dict]:
        """텍스트 오류 분석"""
        
        errors = []
        
        # 1. 문법 오류 감지
        grammar_errors = await self.detect_grammar_errors(text)
        errors.extend(grammar_errors)
        
        # 2. 어휘 오류 감지
        vocabulary_errors = await self.detect_vocabulary_errors(text)
        errors.extend(vocabulary_errors)
        
        # 3. 한국어 화자 특성 오류 감지
        korean_pattern_errors = await self.detect_korean_speaker_errors(text)
        errors.extend(korean_pattern_errors)
        
        # 4. 맞춤법 오류 감지
        spelling_errors = await self.detect_spelling_errors(text)
        errors.extend(spelling_errors)
        
        # 5. 구두점 오류 감지
        punctuation_errors = await self.detect_punctuation_errors(text)
        errors.extend(punctuation_errors)
        
        # 중복 제거 및 우선순위 정렬
        unique_errors = self.deduplicate_errors(errors)
        prioritized_errors = self.prioritize_errors(unique_errors)
        
        return prioritized_errors
    
    async def detect_grammar_errors(self, text: str) -> List[Dict]:
        """문법 오류 감지"""
        errors = []
        words = text.split()
        
        # Subject-Verb Agreement 체크
        sva_errors = self.check_subject_verb_agreement(text)
        errors.extend(sva_errors)
        
        # 관사 사용 오류
        article_errors = self.check_article_usage(text)
        errors.extend(article_errors)
        
        # 전치사 오류
        preposition_errors = self.check_preposition_usage(text)
        errors.extend(preposition_errors)
        
        return errors
    
    def check_subject_verb_agreement(self, text: str) -> List[Dict]:
        """주어-동사 일치 확인"""
        errors = []
        
        # 간단한 패턴 매칭 (실제로는 더 정교한 파싱 필요)
        patterns = [
            (r'\b(I)\s+(is|are)\b', 'I am'),
            (r'\b(you|we|they)\s+(is)\b', lambda m: f"{m.group(1)} are"),
            (r'\b(he|she|it)\s+(are)\b', lambda m: f"{m.group(1)} is"),
            (r'\b(he|she|it)\s+(\w+)(?<!s)(?<!ed)(?<!ing)\s', lambda m: f"{m.group(1)} {m.group(2)}s")
        ]
        
        for pattern, correction in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if callable(correction):
                    correct_form = correction(match)
                else:
                    correct_form = correction
                
                errors.append({
                    'type': 'subject_verb_agreement',
                    'error': match.group(0),
                    'correction': correct_form,
                    'position': match.span(),
                    'explanation': 'Subject and verb must agree in number',
                    'severity': 'high'
                })
        
        return errors
    
    def check_article_usage(self, text: str) -> List[Dict]:
        """관사 사용 확인"""
        errors = []
        
        # "a" before vowel sound
        matches = re.finditer(r'\ba\s+([aeiouAEIOU]\w*)', text)
        for match in matches:
            errors.append({
                'type': 'article_error',
                'error': f"a {match.group(1)}",
                'correction': f"an {match.group(1)}",
                'position': match.span(),
                'explanation': 'Use "an" before words that start with vowel sounds',
                'severity': 'medium'
            })
        
        # "an" before consonant sound  
        matches = re.finditer(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w*)', text)
        for match in matches:
            errors.append({
                'type': 'article_error',
                'error': f"an {match.group(1)}",
                'correction': f"a {match.group(1)}",
                'position': match.span(),
                'explanation': 'Use "a" before words that start with consonant sounds',
                'severity': 'medium'
            })
        
        return errors
    
    def check_preposition_usage(self, text: str) -> List[Dict]:
        """전치사 사용 확인"""
        errors = []
        
        preposition_corrections = [
            (r'\barrive\s+to\s+(home|school|office)', r'arrive at \1'),
            (r'\blisten\s+music', 'listen to music'),
            (r'\bdepend\s+of', 'depend on'),
            (r'\bthink\s+about\s+of', 'think about'),
            (r'\bin\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)', r'on \1')
        ]
        
        for pattern, correction in preposition_corrections:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append({
                    'type': 'preposition_error',
                    'error': match.group(0),
                    'correction': correction if isinstance(correction, str) else re.sub(pattern, correction, match.group(0)),
                    'position': match.span(),
                    'explanation': 'Incorrect preposition usage',
                    'severity': 'medium'
                })
        
        return errors
    
    async def detect_vocabulary_errors(self, text: str) -> List[Dict]:
        """어휘 오류 감지"""
        errors = []
        
        # False friends 체크
        for word, info in self.vocabulary_errors['false_friends'].items():
            if info['wrong_usage'] in text.lower():
                errors.append({
                    'type': 'vocabulary_error',
                    'subtype': 'false_friend',
                    'error': info['wrong_usage'],
                    'correction': word,
                    'explanation': f'In this context, use "{word}" instead of "{info["wrong_usage"]}"',
                    'severity': 'medium'
                })
        
        # 어형 변화 오류
        word_formation_errors = self.vocabulary_errors['word_formation']['adjective_noun']
        for error_pattern in word_formation_errors:
            if error_pattern['wrong'] in text.lower():
                errors.append({
                    'type': 'vocabulary_error',
                    'subtype': 'word_formation',
                    'error': error_pattern['wrong'],
                    'correction': error_pattern['correct'],
                    'explanation': 'Incorrect word formation',
                    'severity': 'medium'
                })
        
        return errors
    
    async def detect_korean_speaker_errors(self, text: str) -> List[Dict]:
        """한국어 화자 특성 오류 감지"""
        errors = []
        
        # 관사 누락
        if not re.search(r'\b(a|an|the)\b', text) and len(text.split()) > 3:
            # 명사가 있는지 확인
            if re.search(r'\b(student|teacher|book|car|house)\b', text):
                errors.append({
                    'type': 'korean_speaker_error',
                    'subtype': 'article_omission',
                    'error': 'missing articles',
                    'correction': 'Add appropriate articles (a, an, the)',
                    'explanation': 'English requires articles before nouns',
                    'severity': 'high'
                })
        
        # 주어 누락 확인
        if re.match(r'^(am|is|are|was|were|have|has)', text.strip()):
            errors.append({
                'type': 'korean_speaker_error',
                'subtype': 'subject_omission',
                'error': 'missing subject',
                'correction': 'Add a subject (I, you, he, she, it, etc.)',
                'explanation': 'English sentences require explicit subjects',
                'severity': 'high'
            })
        
        # 복수형 누락
        plural_pattern = r'\b(two|three|four|five|six|seven|eight|nine|ten|many|several)\s+([a-zA-Z]+)(?!s)(?!es)\b'
        matches = re.finditer(plural_pattern, text)
        for match in matches:
            number_word = match.group(1)
            noun = match.group(2)
            if noun not in ['fish', 'sheep', 'deer', 'information', 'water']:  # 불규칙 복수형 예외
                errors.append({
                    'type': 'korean_speaker_error',
                    'subtype': 'plural_omission',
                    'error': f"{number_word} {noun}",
                    'correction': f"{number_word} {noun}s",
                    'explanation': 'Use plural form after numbers greater than one',
                    'severity': 'medium'
                })
        
        return errors
    
    async def detect_spelling_errors(self, text: str) -> List[Dict]:
        """맞춤법 오류 감지"""
        errors = []
        
        # 일반적인 맞춤법 오류
        common_misspellings = {
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'begining': 'beginning',
            'runing': 'running',
            'geting': 'getting',
            'writeing': 'writing',
            'comeing': 'coming',
            'makeing': 'making',
            'hopeing': 'hoping'
        }
        
        words = text.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in common_misspellings:
                errors.append({
                    'type': 'spelling_error',
                    'error': word,
                    'correction': common_misspellings[clean_word],
                    'explanation': 'Common spelling mistake',
                    'severity': 'low'
                })
        
        return errors
    
    async def detect_punctuation_errors(self, text: str) -> List[Dict]:
        """구두점 오류 감지"""
        errors = []
        
        # 문장 끝 구두점 누락
        if text and not text.strip().endswith(('.', '!', '?')):
            errors.append({
                'type': 'punctuation_error',
                'error': 'missing end punctuation',
                'correction': 'Add period, exclamation mark, or question mark',
                'explanation': 'Sentences should end with proper punctuation',
                'severity': 'low'
            })
        
        # 대문자 시작 누락
        if text and text[0].islower():
            errors.append({
                'type': 'capitalization_error',
                'error': 'lowercase sentence start',
                'correction': 'Capitalize the first word',
                'explanation': 'Sentences should start with a capital letter',
                'severity': 'low'
            })
        
        return errors
    
    def deduplicate_errors(self, errors: List[Dict]) -> List[Dict]:
        """중복 오류 제거"""
        seen_errors = set()
        unique_errors = []
        
        for error in errors:
            error_key = (error['type'], error['error'])
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                unique_errors.append(error)
        
        return unique_errors
    
    def prioritize_errors(self, errors: List[Dict]) -> List[Dict]:
        """오류 우선순위 정렬"""
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        
        return sorted(errors, key=lambda x: (
            severity_order.get(x.get('severity', 'medium'), 1),
            x.get('type', ''),
            x.get('error', '')
        ))
    
    async def get_user_error_patterns(self, user_id: str) -> List[Dict]:
        """사용자별 오류 패턴 분석"""
        
        if user_id not in self.user_error_history:
            return []
        
        user_errors = self.user_error_history[user_id]
        
        # 오류 유형별 빈도 분석
        error_frequency = Counter()
        error_examples = defaultdict(list)
        
        # 최근 30일간의 오류만 분석
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for error_record in user_errors:
            if error_record['timestamp'] > cutoff_date:
                error_type = error_record['type']
                error_frequency[error_type] += 1
                
                if len(error_examples[error_type]) < 3:  # 최대 3개 예시
                    error_examples[error_type].append({
                        'error': error_record['error'],
                        'correction': error_record['correction'],
                        'timestamp': error_record['timestamp']
                    })
        
        # 패턴 분석 결과 생성
        patterns = []
        for error_type, frequency in error_frequency.most_common(10):
            pattern = {
                'error_type': error_type,
                'frequency': frequency,
                'percentage': round((frequency / len(user_errors)) * 100, 1),
                'examples': error_examples[error_type],
                'improvement_trend': self.calculate_error_trend(user_id, error_type),
                'recommendations': self.get_error_specific_recommendations(error_type)
            }
            patterns.append(pattern)
        
        return patterns
    
    def calculate_error_trend(self, user_id: str, error_type: str) -> str:
        """오류 개선 추세 계산"""
        if user_id not in self.user_error_history:
            return 'no_data'
        
        user_errors = self.user_error_history[user_id]
        
        # 최근 30일을 2주씩 나누어 비교
        now = datetime.now()
        recent_period = now - timedelta(days=14)
        older_period = now - timedelta(days=30)
        
        recent_errors = sum(1 for e in user_errors 
                          if e['type'] == error_type and e['timestamp'] > recent_period)
        older_errors = sum(1 for e in user_errors 
                         if e['type'] == error_type and older_period < e['timestamp'] <= recent_period)
        
        if older_errors == 0:
            return 'new_error_type'
        
        if recent_errors < older_errors:
            return 'improving'
        elif recent_errors > older_errors:
            return 'worsening'
        else:
            return 'stable'
    
    def get_error_specific_recommendations(self, error_type: str) -> List[str]:
        """오류 유형별 개선 추천"""
        recommendations = {
            'subject_verb_agreement': [
                'Practice identifying subjects and matching them with correct verbs',
                'Review third-person singular verb forms',
                'Use subject-verb agreement exercises'
            ],
            'article_error': [
                'Study when to use a, an, and the',
                'Practice with article exercises',
                'Pay attention to vowel and consonant sounds'
            ],
            'preposition_error': [
                'Learn common preposition combinations',
                'Practice with phrasal verbs',
                'Study prepositions of time and place'
            ],
            'korean_speaker_error': [
                'Focus on structural differences between Korean and English',
                'Practice with articles and plural forms',
                'Work on English sentence patterns'
            ],
            'vocabulary_error': [
                'Build vocabulary through reading',
                'Learn word collocations',
                'Practice word formation rules'
            ],
            'spelling_error': [
                'Use spell-check tools',
                'Practice common spelling patterns',
                'Read more to improve spelling recognition'
            ]
        }
        
        return recommendations.get(error_type, [
            'Continue practicing this area',
            'Ask for specific feedback',
            'Use additional learning resources'
        ])
    
    async def store_error_feedback(self, user_id: str, error_data: Dict):
        """오류 피드백 저장"""
        
        if user_id not in self.user_error_history:
            self.user_error_history[user_id] = []
        
        error_record = {
            'timestamp': datetime.now(),
            'type': error_data['type'],
            'error': error_data['error'],
            'correction': error_data['correction'],
            'context': error_data.get('context', ''),
            'user_acknowledged': False,
            'repeated_error': self.is_repeated_error(user_id, error_data)
        }
        
        self.user_error_history[user_id].append(error_record)
        
        # 메모리 관리 - 오래된 기록 정리 (최대 200개 유지)
        if len(self.user_error_history[user_id]) > 200:
            self.user_error_history[user_id] = self.user_error_history[user_id][-200:]
    
    def is_repeated_error(self, user_id: str, error_data: Dict) -> bool:
        """반복 오류 확인"""
        if user_id not in self.user_error_history:
            return False
        
        # 최근 10개 오류에서 같은 유형의 오류가 있는지 확인
        recent_errors = self.user_error_history[user_id][-10:]
        
        for error in recent_errors:
            if (error['type'] == error_data['type'] and 
                error['error'] == error_data['error']):
                return True
        
        return False
    
    async def get_recommendations(self, user_id: str) -> List[Dict]:
        """사용자별 오류 개선 추천"""
        
        error_patterns = await self.get_user_error_patterns(user_id)
        recommendations = []
        
        for pattern in error_patterns[:3]:  # 상위 3개 오류 패턴
            error_type = pattern['error_type']
            frequency = pattern['frequency']
            trend = pattern['improvement_trend']
            
            if trend == 'worsening' or frequency > 5:
                priority = 'high'
            elif trend == 'stable' and frequency > 2:
                priority = 'medium'
            else:
                priority = 'low'
            
            recommendations.append({
                'type': 'error_improvement',
                'error_type': error_type,
                'frequency': frequency,
                'trend': trend,
                'priority': priority,
                'specific_recommendations': self.get_error_specific_recommendations(error_type),
                'practice_exercises': self.get_practice_exercises(error_type)
            })
        
        return recommendations
    
    def get_practice_exercises(self, error_type: str) -> List[Dict]:
        """오류 유형별 연습 문제 추천"""
        exercises = {
            'subject_verb_agreement': [
                {
                    'type': 'fill_blank',
                    'question': 'She ___ (go) to school every day.',
                    'answer': 'goes',
                    'explanation': 'Third person singular takes -s/-es'
                },
                {
                    'type': 'correction',
                    'question': 'They is very happy.',
                    'answer': 'They are very happy.',
                    'explanation': 'Plural subject takes plural verb'
                }
            ],
            'article_error': [
                {
                    'type': 'fill_blank',
                    'question': 'I saw ___ elephant at the zoo.',
                    'answer': 'an',
                    'explanation': 'Use "an" before vowel sounds'
                },
                {
                    'type': 'fill_blank',
                    'question': 'She is ___ teacher.',
                    'answer': 'a',
                    'explanation': 'Use "a" before consonant sounds'
                }
            ],
            'preposition_error': [
                {
                    'type': 'fill_blank',
                    'question': 'I arrive ___ school at 8 AM.',
                    'answer': 'at',
                    'explanation': 'Use "at" for specific places'
                }
            ]
        }
        
        return exercises.get(error_type, [
            {
                'type': 'general',
                'question': f'Practice exercises for {error_type}',
                'answer': 'Continue practicing',
                'explanation': 'Focus on this error type'
            }
        ])
    
    async def generate_correction_feedback(self, errors: List[Dict]) -> str:
        """교정 피드백 생성"""
        
        if not errors:
            return "Great job! No errors detected in your message."
        
        high_priority = [e for e in errors if e.get('severity') == 'high']
        medium_priority = [e for e in errors if e.get('severity') == 'medium']
        
        feedback_parts = []
        
        if high_priority:
            feedback_parts.append("Let me help you with a few important corrections:")
            for error in high_priority[:2]:  # 최대 2개
                feedback_parts.append(f"• Instead of '{error['error']}', try '{error['correction']}' - {error['explanation']}")
        
        if medium_priority and len(feedback_parts) < 3:
            remaining_slots = 3 - len(feedback_parts)
            for error in medium_priority[:remaining_slots]:
                feedback_parts.append(f"• Consider: '{error['correction']}' instead of '{error['error']}'")
        
        if len(feedback_parts) == 0:
            feedback_parts.append("I noticed some small areas for improvement, but you're communicating clearly!")
        
        return " ".join(feedback_parts)