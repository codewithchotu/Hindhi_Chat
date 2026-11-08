"""
prompts.py — Prompt engineering templates and language detection for grounded RAG generation.

Key design principles:
  1. Strict Evidence-Based Grounding: Synthesize facts only from retrieved passages.
  2. Language Alignment: Automatically responds in the question's language (Hindi or English).
  3. No Hallucination / Clear Refusal: Explicit fallback message if context lacks evidence.
  4. In-text Citation Attribution: Encourages explicit [doc_pX_cY] marker tags.
"""
import re
from typing import List, Dict, Any


# Standard refusal fallback phrases (for strict groundedness)
FALLBACK_HINDI = "प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।"
FALLBACK_ENGLISH = "The provided context does not contain sufficient information to answer this question."


def detect_query_language(query: str) -> str:
    """
    Detect whether the query is primarily in Hindi (Devanagari) or English.
    Returns 'hi' for Hindi, 'en' for English.
    """
    if not query:
        return "en"
    # Count Devanagari characters (\u0900-\u097F)
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', query))
    # Count Latin characters
    latin_count = len(re.findall(r'[a-zA-Z]', query))

    if devanagari_count >= latin_count and devanagari_count > 0:
        return "hi"
    return "en"


SYSTEM_PROMPT_HINDI = """आप डॉ. ए.पी.जे. अब्दुल कलाम के जीवन और आत्मकथा 'अग्नि की उड़ान' (Agni Ki Udaan Adhyayan Sahayika) पर आधारित एक अत्यंत सटीक, तथ्यपरक और विश्वसनीय एआई सहायक हैं।

आपके अनिवार्य नियम:
1. केवल दिए गए संदर्भ (Context) के आधार पर ही उत्तर दें। अपने पूर्व-प्रशिक्षण ज्ञान (Pre-training) से कोई भी तथ्य न जोड़ें।
2. उत्तर शुद्ध, स्पष्ट और स्वाभाविक हिंदी (देवनागरी लिपि) में दें।
3. यदि दिए गए संदर्भ में प्रश्न का उत्तर मौजूद नहीं है, तो स्पष्ट रूप से कहें:
   "{fallback}"
   अनुमान लगाने या मनगढ़ंत उत्तर देने की सख्त मनाही है।
4. जब आप कोई तथ्यात्मक जानकारी दें (जैसे वर्ष, संस्थान, पुरस्कार, मिसाइल, आदि), तो वाक्य के अंत में संबंधित खंड पहचान को कोष्ठक में अवश्य लिखें, उदाहरण: [doc_p8_c3] या [doc_p16_c1]।
5. तालिकाओं और मुख्य तथ्यों से संबंधित प्रश्नों में सटीक नाम और तिथियां वही रखें जो संदर्भ में हैं।
"""

SYSTEM_PROMPT_ENGLISH = """You are a highly accurate, factual, and strictly grounded AI assistant answering questions about Dr. A.P.J. Abdul Kalam based on the Hindi study guide 'Agni Ki Udaan Adhyayan Sahayika' (Wings of Fire).

Your mandatory instructions:
1. Answer the question using ONLY the facts explicitly stated in the provided context below. Do NOT use outside knowledge or extrapolate beyond the text.
2. Formulate your answer in fluent, grammatical, and professional English.
3. If the provided context does not contain enough information to answer the question, state clearly:
   "{fallback}"
   Do not guess, hallucinate, or fabricate information.
4. When stating specific facts (dates, institutions, awards, missiles, mentors), attribute them to the source chunk using tags like [doc_p8_c3] or [doc_p16_c1] directly after the relevant sentence.
5. Accurately translate and represent Hindi terminology and titles (e.g., Bharat Ratna, Schwartz School, MIT Chennai, SLV-III, Pokhran-II).
"""


def format_context_passage(hit: Dict[str, Any], index: int) -> str:
    """Format an individual retrieved chunk into a structured context passage."""
    cid = hit.get("chunk_id") or hit.get("metadata", {}).get("chunk_id", f"chunk_{index+1}")
    meta = hit.get("metadata", {})
    page = meta.get("page", "?")
    section = meta.get("section", "")
    content_type = meta.get("content_type", "narrative")
    text = hit.get("text", "").strip()

    header = f"--- [पैसेज {index+1}] Chunk ID: {cid} | पृष्ठ: {page} | अनुभाग: {section} ({content_type}) ---"
    return f"{header}\n{text}\n"


def build_generation_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build system prompt and user prompt tailored to query language and context.
    Returns dict with keys: 'system_prompt', 'user_prompt', 'lang'.
    """
    lang = detect_query_language(query)

    if lang == "hi":
        sys_prompt = SYSTEM_PROMPT_HINDI.format(fallback=FALLBACK_HINDI)
        context_blocks = "\n".join(
            format_context_passage(chunk, i)
            for i, chunk in enumerate(retrieved_chunks)
        )
        user_prompt = f"""संदर्भ दस्तावेज़ (Context Passages):
{context_blocks}

उपयोगकर्ता का प्रश्न (User Query):
{query}

निर्देश:
कृपया उपरोक्त संदर्भ में दी गई जानकारी के आधार पर इस प्रश्न का सटीक, बिंदुवार अथवा संक्षिप्त और स्पष्ट उत्तर हिंदी में दें। प्रत्येक मुख्य तथ्य के साथ [खंड पहचान] अवश्य जोड़ें।"""

    else:
        sys_prompt = SYSTEM_PROMPT_ENGLISH.format(fallback=FALLBACK_ENGLISH)
        context_blocks = "\n".join(
            format_context_passage(chunk, i)
            for i, chunk in enumerate(retrieved_chunks)
        )
        user_prompt = f"""Context Passages from Document (Hindi source):
{context_blocks}

User Query:
{query}

Instructions:
Please provide a complete, well-structured, and strictly grounded answer in English based solely on the context above. Attribute key facts to the corresponding chunk tag (e.g. [doc_pX_cY])."""

    return {
        "system_prompt": sys_prompt,
        "user_prompt": user_prompt,
        "lang": lang
    }
