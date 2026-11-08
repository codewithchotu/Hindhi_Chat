"""
dataset.py — Standardized evaluation dataset for Phase 6 RAG benchmarking.

Contains:
  1. 12 Factual Evaluation Queries (6 Mandatory + 6 Additional)
     - Full ground truth: target pages, target chunk IDs, key factual terms, gold reference answers.
  2. 4 Out-of-Domain / Unanswerable Queries
     - Validates refusal behavior, absence of hallucination, and zero false citations.
  3. 10 Adversarial Grounding Test Cases
     - Tests the GroundingVerifier across fully grounded claims, invented IDs, year mismatches,
       and unsupported entities.
"""
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# 12 Factual Queries (6 Mandatory + 6 Additional)
# ---------------------------------------------------------------------------
EVAL_QUERIES: List[Dict[str, Any]] = [
    # ── 6 Mandatory Assignment Queries ──────────────────────────────────────
    {
        "id": 1,
        "query": "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?",
        "lang": "Hindi",
        "category": "mandatory",
        "primary_page": 8,
        "ground_truth_pages": [8, 17, 20, 22],
        "primary_chunk_ids": ["doc_p8_c3"],
        "ground_truth_chunk_ids": ["doc_p8_c3", "doc_p8_c2", "doc_p17_c2", "doc_p20_c2", "doc_p22_c3", "doc_p22_c4"],
        "key_terms": ["रोहिणी", "1980", "SLV-III"],
        "reference_answer": "18 जुलाई 1980 को SLV-III रॉकेट द्वारा रोहिणी उपग्रह (RS-1) को सफलतापूर्वक पृथ्वी की कक्षा में स्थापित किया गया।",
        "expected_refusal": False,
    },
    {
        "id": 2,
        "query": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?",
        "lang": "Hindi",
        "category": "mandatory",
        "primary_page": 16,
        "ground_truth_pages": [16, 17, 20, 22],
        "primary_chunk_ids": ["doc_p16_c1", "doc_p16_c2"],
        "ground_truth_chunk_ids": ["doc_p16_c1", "doc_p16_c2", "doc_p17_c2", "doc_p20_c3", "doc_p22_c4"],
        "key_terms": ["1997", "भारत रत्न"],
        "reference_answer": "डॉ. कलाम को वर्ष 1997 में भारत का सर्वोच्च नागरिक सम्मान 'भारत रत्न' प्रदान किया गया।",
        "expected_refusal": False,
    },
    {
        "id": 3,
        "query": "पोखरण-II में कलाम की क्या भूमिका थी?",
        "lang": "Hindi",
        "category": "mandatory",
        "primary_page": 10,
        "ground_truth_pages": [10, 17, 20],
        "primary_chunk_ids": ["doc_p10_c1"],
        "ground_truth_chunk_ids": ["doc_p10_c1", "doc_p10_c2", "doc_p17_c2", "doc_p20_c2"],
        "key_terms": ["1998", "पोखरण-II", "DRDO", "वैज्ञानिक सलाहकार"],
        "reference_answer": "मई 1998 के पोखरण-II परमाणु परीक्षणों में कलाम ने रक्षा मंत्री के वैज्ञानिक सलाहकार और DRDO प्रमुख के रूप में परमाणु ऊर्जा विभाग (DAE) के साथ मिलकर प्रमुख रणनीतिक एवं समन्वयक भूमिका निभाई।",
        "expected_refusal": False,
    },
    {
        "id": 4,
        "query": "Which institution did Kalam attend to study aeronautical engineering?",
        "lang": "English",
        "category": "mandatory",
        "primary_page": 6,
        "ground_truth_pages": [6, 17, 20, 22],
        "primary_chunk_ids": ["doc_p6_c2"],
        "ground_truth_chunk_ids": ["doc_p6_c1", "doc_p6_c2", "doc_p17_c1", "doc_p20_c3", "doc_p22_c1", "doc_p22_c2"],
        "key_terms": ["Madras Institute of Technology", "MIT", "Chennai"],
        "reference_answer": "Dr. Kalam attended the Madras Institute of Technology (MIT) in Chennai to study aeronautical engineering.",
        "expected_refusal": False,
    },
    {
        "id": 5,
        "query": "Who co-wrote the autobiography, and in what year was it published?",
        "lang": "English",
        "category": "mandatory",
        "primary_page": 3,
        "ground_truth_pages": [1, 3, 17, 22],
        "primary_chunk_ids": ["doc_p3_c3"],
        "ground_truth_chunk_ids": ["doc_p3_c3", "doc_p1_c1", "doc_p17_c3", "doc_p22_c1", "doc_p22_c3"],
        "key_terms": ["Arun Tiwari", "1999", "Wings of Fire"],
        "reference_answer": "Dr. Kalam's autobiography 'Wings of Fire' was co-written with Arun Tiwari and published in 1999.",
        "expected_refusal": False,
    },
    {
        "id": 6,
        "query": "How and where did Kalam die in 2015?",
        "lang": "English",
        "category": "mandatory",
        "primary_page": 15,
        "ground_truth_pages": [15, 17, 22],
        "primary_chunk_ids": ["doc_p15_c3", "doc_p22_c2"],
        "ground_truth_chunk_ids": ["doc_p15_c3", "doc_p22_c2", "doc_p15_c1", "doc_p15_c2", "doc_p17_c3"],
        "key_terms": ["27 July 2015", "Shillong"],
        "reference_answer": "Dr. Kalam passed away on 27 July 2015 in Shillong after collapsing while delivering a lecture at the Indian Institute of Management (IIM) Shillong.",
        "expected_refusal": False,
    },
    # ── 6 Additional Assignment Queries ─────────────────────────────────────
    {
        "id": 7,
        "query": "पृथ्वी और अग्नि मिसाइल किस कार्यक्रम के तहत विकसित की गईं?",
        "lang": "Hindi",
        "category": "additional",
        "primary_page": 9,
        "ground_truth_pages": [9, 17, 20, 22],
        "primary_chunk_ids": ["doc_p9_c1", "doc_p22_c3"],
        "ground_truth_chunk_ids": ["doc_p9_c1", "doc_p22_c3", "doc_p9_c2", "doc_p9_c3", "doc_p17_c2", "doc_p20_c2"],
        "key_terms": ["IGMDP", "निर्देशित मिसाइल", "1980"],
        "reference_answer": "पृथ्वी और अग्नि मिसाइलें 1980 के दशक में एकीकृत निर्देशित मिसाइल विकास कार्यक्रम (IGMDP) के तहत विकसित की गईं।",
        "expected_refusal": False,
    },
    {
        "id": 8,
        "query": "What were the names of Kalam's parents and where did he grow up?",
        "lang": "English",
        "category": "additional",
        "primary_page": 4,
        "ground_truth_pages": [4, 12, 17, 22],
        "primary_chunk_ids": ["doc_p4_c1"],
        "ground_truth_chunk_ids": ["doc_p4_c1", "doc_p4_c2", "doc_p12_c1", "doc_p17_c1", "doc_p22_c1"],
        "key_terms": ["Jainulabdeen", "Ashiamma", "Rameswaram"],
        "reference_answer": "Dr. Kalam's father was Jainulabdeen and his mother was Ashiamma. He grew up in the pilgrimage town of Rameswaram, Tamil Nadu.",
        "expected_refusal": False,
    },
    {
        "id": 9,
        "query": "विक्रम साराभाई और सतीश धवन ने कलाम के जीवन में क्या भूमिका निभाई?",
        "lang": "Hindi",
        "category": "additional",
        "primary_page": 7,
        "ground_truth_pages": [7, 8, 12, 17],
        "primary_chunk_ids": ["doc_p12_c3", "doc_p12_c2", "doc_p7_c3", "doc_p8_c2"],
        "ground_truth_chunk_ids": ["doc_p12_c3", "doc_p12_c2", "doc_p7_c3", "doc_p8_c2", "doc_p12_c1", "doc_p8_c3", "doc_p17_c1", "doc_p17_c2"],
        "key_terms": ["साराभाई", "धवन", "1979", "मार्गदर्शन"],
        "reference_answer": "विक्रम साराभाई ने कलाम की प्रतिभा को पहचानकर उन्हें INCOSPAR रॉकेट टीम में शामिल किया और मार्गदर्शन दिया, जबकि सतीश धवन ने 1979 के SLV-III असफलता के दौरान विफलता की जिम्मेदारी खुद लेकर और सफलता का श्रेय कलाम को देकर नेतृत्व का महान पाठ सिखाया।",
        "expected_refusal": False,
    },
    {
        "id": 10,
        "query": "During which years did Kalam serve as the President of India?",
        "lang": "English",
        "category": "additional",
        "primary_page": 14,
        "ground_truth_pages": [3, 14, 17, 22],
        "primary_chunk_ids": ["doc_p14_c1"],
        "ground_truth_chunk_ids": ["doc_p14_c1", "doc_p14_c2", "doc_p3_c1", "doc_p17_c3", "doc_p22_c3"],
        "key_terms": ["2002", "2007", "11th President"],
        "reference_answer": "Dr. Kalam served as the 11th President of India from 2002 to 2007, widely known as the 'People's President'.",
        "expected_refusal": False,
    },
    {
        "id": 11,
        "query": "कलाम को 1981 और 1990 में कौन-से नागरिक पुरस्कार मिले?",
        "lang": "Hindi",
        "category": "additional",
        "primary_page": 16,
        "ground_truth_pages": [16, 17, 22],
        "primary_chunk_ids": ["doc_p16_c2"],
        "ground_truth_chunk_ids": ["doc_p16_c1", "doc_p16_c2", "doc_p17_c2", "doc_p22_c3"],
        "key_terms": ["पद्म भूषण", "पद्म विभूषण", "1981", "1990"],
        "reference_answer": "कलाम को वर्ष 1981 में पद्म भूषण तथा वर्ष 1990 में पद्म विभूषण से सम्मानित किया गया।",
        "expected_refusal": False,
    },
    {
        "id": 12,
        "query": "What was Kalam's core message to youth about dreams?",
        "lang": "English",
        "category": "additional",
        "primary_page": 18,
        "ground_truth_pages": [3, 11, 18, 22],
        "primary_chunk_ids": ["doc_p3_c5", "doc_p18_c1"],
        "ground_truth_chunk_ids": ["doc_p3_c5", "doc_p18_c1", "doc_p18_c2", "doc_p3_c4", "doc_p11_c1", "doc_p22_c4"],
        "key_terms": ["dream", "thoughts", "action"],
        "reference_answer": "Kalam taught youth to dream, as dreams transform into thoughts, and thoughts result in action.",
        "expected_refusal": False,
    },
]

# ---------------------------------------------------------------------------
# 4 Out-of-Domain / Unanswerable Queries
# ---------------------------------------------------------------------------
UNANSWERABLE_QUERIES: List[Dict[str, Any]] = [
    {
        "id": 13,
        "query": "What was Dr. Kalam's favorite traditional Indian recipe to cook on Sunday afternoons?",
        "lang": "English",
        "category": "unanswerable",
        "primary_page": None,
        "ground_truth_pages": [],
        "primary_chunk_ids": [],
        "ground_truth_chunk_ids": [],
        "key_terms": [],
        "reference_answer": "The provided context does not contain sufficient information to answer this question.",
        "expected_refusal": True,
    },
    {
        "id": 14,
        "query": "कलाम ने किस वर्ष मंगल ग्रह (Mars) पर भारतीय अंतरिक्ष यात्रियों को भेजने के मानव मिशन का नेतृत्व किया था?",
        "lang": "Hindi",
        "category": "unanswerable",
        "primary_page": None,
        "ground_truth_pages": [],
        "primary_chunk_ids": [],
        "ground_truth_chunk_ids": [],
        "key_terms": [],
        "reference_answer": "प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।",
        "expected_refusal": True,
    },
    {
        "id": 15,
        "query": "What are the names of Dr. Kalam's children and which universities did they graduate from?",
        "lang": "English",
        "category": "unanswerable",
        "primary_page": None,
        "ground_truth_pages": [],
        "primary_chunk_ids": [],
        "ground_truth_chunk_ids": [],
        "key_terms": [],
        "reference_answer": "The provided context does not contain sufficient information to answer this question.",
        "expected_refusal": True,
    },
    {
        "id": 16,
        "query": "कलाम ने वर्ष 2040 के लिए भारत के संविधान में किन अनुच्छेदों को संशोधित करने की आधिकारिक सिफारिश की थी?",
        "lang": "Hindi",
        "category": "unanswerable",
        "primary_page": None,
        "ground_truth_pages": [],
        "primary_chunk_ids": [],
        "ground_truth_chunk_ids": [],
        "key_terms": [],
        "reference_answer": "प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।",
        "expected_refusal": True,
    },
]

# ---------------------------------------------------------------------------
# 10 Adversarial Grounding Test Cases
# ---------------------------------------------------------------------------
ADVERSARIAL_GROUNDING_CASES: List[Dict[str, Any]] = [
    {
        "id": "adv_01_fully_grounded_hi",
        "desc": "Valid Hindi factual statement with valid citation",
        "answer": "18 जुलाई 1980 को SLV-III रॉकेट द्वारा रोहिणी उपग्रह को सफलतापूर्वक पृथ्वी की कक्षा में स्थापित किया गया [doc_p8_c3]।",
        "context_chunks": [
            {"chunk_id": "doc_p8_c3", "text": "18 जुलाई 1980 को SLV-III रॉकेट द्वारा रोहिणी उपग्रह को सफलतापूर्वक पृथ्वी की कक्षा में स्थापित किया गया।", "metadata": {"page": 8, "section": "6 · इसरो के वर्ष"}}
        ],
        "expected_is_grounded": True,
        "expected_issues": [],
    },
    {
        "id": "adv_02_fully_grounded_en",
        "desc": "Valid English factual statement with valid citation",
        "answer": "Dr. Kalam attended the Madras Institute of Technology (MIT) in Chennai to study aeronautical engineering [doc_p6_c2].",
        "context_chunks": [
            {"chunk_id": "doc_p6_c2", "text": "He joined the Madras Institute of Technology (MIT), Chennai, to study aeronautical engineering.", "metadata": {"page": 6, "section": "4 · College Years"}}
        ],
        "expected_is_grounded": True,
        "expected_issues": [],
    },
    {
        "id": "adv_03_invented_chunk_id",
        "desc": "Answer cites a fake non-existent chunk ID [doc_fake_chunk_99]",
        "answer": "कलाम ने 1999 में आत्मकथा लिखी [doc_fake_chunk_99]।",
        "context_chunks": [
            {"chunk_id": "doc_p3_c3", "text": "1999 में आत्मकथा 'विंग्स ऑफ फायर' प्रकाशित हुई।", "metadata": {"page": 3, "section": "1 · ए.पी.जे. अब्दुल कलाम"}}
        ],
        "expected_is_grounded": False,
        "expected_issues": ["invented_citations"],
    },
    {
        "id": "adv_04_wrong_year_in_sentence",
        "desc": "Citation points to doc_p16_c1, but sentence claims Bharat Ratna in 1980 instead of 1997",
        "answer": "डॉ. कलाम को वर्ष 1980 में भारत रत्न पुरस्कार प्रदान किया गया था [doc_p16_c1]।",
        "context_chunks": [
            {"chunk_id": "doc_p16_c1", "text": "वर्ष 1997 में डॉ. कलाम को भारत का सर्वोच्च नागरिक सम्मान 'भारत रत्न' प्रदान किया गया।", "metadata": {"page": 16, "section": "14 · पुरस्कार"}}
        ],
        "expected_is_grounded": False,
        "expected_issues": ["citation_mismatches", "unsupported_facts"],
    },
    {
        "id": "adv_05_unsupported_acronym_award",
        "desc": "Answer claims Kalam won a Nobel Prize (NOBEL acronym / ungrounded)",
        "answer": "Dr. Kalam was awarded the NOBEL prize in physics during 2005 [doc_p6_c1].",
        "context_chunks": [
            {"chunk_id": "doc_p6_c1", "text": "Dr. Kalam studied physics at St. Joseph's College in Tiruchirappalli.", "metadata": {"page": 6, "section": "4 · College Years"}}
        ],
        "expected_is_grounded": False,
        "expected_issues": ["citation_mismatches", "unsupported_facts"],
    },
    {
        "id": "adv_06_unsupported_future_year",
        "desc": "Answer introduces year 2045 not present in context",
        "answer": "कलाम ने वर्ष 2045 के लिए नया अंतरिक्ष विज़न तैयार किया था [doc_p8_c3]।",
        "context_chunks": [
            {"chunk_id": "doc_p8_c3", "text": "18 जुलाई 1980 को SLV-III रॉकेट द्वारा रोहिणी उपग्रह को प्रक्षेपित किया गया।", "metadata": {"page": 8, "section": "6 · इसरो"}}
        ],
        "expected_is_grounded": False,
        "expected_issues": ["citation_mismatches", "unsupported_facts"],
    },
    {
        "id": "adv_07_correct_refusal_hi",
        "desc": "Explicit refusal in Hindi due to lack of evidence",
        "answer": "प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।",
        "context_chunks": [
            {"chunk_id": "doc_p1_c1", "text": "अग्नि की उड़ान अध्ययन सहायिका", "metadata": {"page": 1, "section": "शीर्षक"}}
        ],
        "expected_is_grounded": True,
        "expected_issues": [],
    },
    {
        "id": "adv_08_correct_refusal_en",
        "desc": "Explicit refusal in English due to lack of evidence",
        "answer": "The provided context does not contain sufficient information to answer this question.",
        "context_chunks": [
            {"chunk_id": "doc_p1_c1", "text": "Wings of Fire Study Guide", "metadata": {"page": 1, "section": "Title"}}
        ],
        "expected_is_grounded": True,
        "expected_issues": [],
    },
    {
        "id": "adv_09_empty_answer",
        "desc": "Empty response string should not be considered grounded",
        "answer": "   ",
        "context_chunks": [
            {"chunk_id": "doc_p1_c1", "text": "Sample text", "metadata": {"page": 1, "section": "Title"}}
        ],
        "expected_is_grounded": False,
        "expected_issues": ["empty_answer"],
    },
    {
        "id": "adv_10_wrong_chunk_attribution",
        "desc": "Sentence discussing 1998 Pokhran test cites doc_p4_c1 (childhood page)",
        "answer": "मई 1998 में पोखरण-II परमाणु परीक्षण आयोजित किए गए [doc_p4_c1]।",
        "context_chunks": [
            {"chunk_id": "doc_p4_c1", "text": "कलाम का बचपन रामेश्वरम में बीता। उनके पिता जैनुलाबदीन थे।", "metadata": {"page": 4, "section": "2 · बचपन"}},
            {"chunk_id": "doc_p10_c1", "text": "मई 1998 में पोखरण-II परमाणु परीक्षण आयोजित किए गए।", "metadata": {"page": 10, "section": "8 · पोखरण-II"}}
        ],
        "expected_is_grounded": False,
        "expected_issues": ["citation_mismatches"],
    },
]
