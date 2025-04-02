from datetime import datetime

import numpy as np
import json
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, validator
import openai
from fastapi import FastAPI, HTTPException
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import faiss
from sentence_transformers import SentenceTransformer


# ============== Configuration ==============
class Config:
    OPENAI_API_KEY = "apikey"  # Replace with your key
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local embedding model
    COMPLETION_MODEL = "gpt-4-turbo-preview"
    LOG_LEVEL = logging.INFO
    FAISS_INDEX_PATH = "ros_guidelines_faiss.index"
    GUIDELINES_PATH = "ros_guidelines.json"


# ============== Logging Setup ==============
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== Pydantic Models ==============
class ConstitutionalROS(BaseModel):
    constructional_ros__c: Optional[Literal["Assessed", "Not Assessed"]] = None
    not_assessed_reason: Optional[Literal["Cognitive Impairment", "Patient Refused"]] = None
    reviewed_with_negative: Optional[Literal["true", "false"]] = None
    fever: Optional[Literal["true", "false"]] = None
    chills: Optional[Literal["true", "false"]] = None
    fatigue: Optional[Literal["true", "false"]] = None
    change_in_sleep: Optional[Literal["true", "false"]] = None
    change_in_appetite: Optional[Literal["true", "false"]] = None
    unintentional_weight_gain: Optional[Literal["true", "false"]] = None
    unintentional_weight_loss: Optional[Literal["true", "false"]] = None
    night_sweats: Optional[Literal["true", "false"]] = None
    weakness: Optional[Literal["true", "false"]] = None

    @validator('constructional_ros__c', always=True)
    def validate_ros_status(cls, v, values):
        symptoms = ['fever', 'chills', 'fatigue', 'change_in_sleep',
                    'change_in_appetite', 'unintentional_weight_gain',
                    'unintentional_weight_loss', 'night_sweats', 'weakness']
        if any(values.get(f) is not None for f in symptoms):
            return "Assessed"
        return v

    @validator('not_assessed_reason')
    def validate_reason(cls, v, values):
        if values.get('constructional_ros__c') == 'Not Assessed' and v is None:
            raise ValueError("Reason required when not assessed")
        return v

    @validator('reviewed_with_negative')
    def validate_reviewed(cls, v, values):
        symptoms = ['fever', 'chills', 'fatigue', 'change_in_sleep',
                    'change_in_appetite', 'unintentional_weight_gain',
                    'unintentional_weight_loss', 'night_sweats', 'weakness']
        if v is not None and any(values.get(f) is not None for f in symptoms):
            raise ValueError("Cannot have reviewed_with_negative with symptoms")
        return v


class NeurologicalROS(BaseModel):
    neurological_ros__c: Optional[Literal["Assessed", "Not Assessed"]] = None
    reviewed_and_negative: Optional[Literal["true", "false"]] = None
    cognitive_impairment: Optional[Literal["true", "false"]] = None
    cognitive_impairment_type: Optional[str] = None
    numbness: Optional[Literal["true", "false"]] = None
    tingling: Optional[Literal["true", "false"]] = None
    prickling: Optional[Literal["true", "false"]] = None
    burning_sensation: Optional[Literal["true", "false"]] = None
    itching_sensation: Optional[Literal["true", "false"]] = None
    pains_and_needles: Optional[Literal["true", "false"]] = None
    pain_d_t_innocuous_stimuli: Optional[Literal["true", "false"]] = None
    increased_sensitivity_to_pain: Optional[Literal["true", "false"]] = None
    dizziness: Optional[Literal["true", "false"]] = None
    light_headedness: Optional[Literal["true", "false"]] = None
    vertigo: Optional[Literal["true", "false"]] = None
    fainting: Optional[Literal["true", "false"]] = None
    loss_of_balance: Optional[Literal["true", "false"]] = None
    memory_problem: Optional[Literal["true", "false"]] = None
    difficulty_speaking: Optional[Literal["true", "false"]] = None
    motor_weakness: Optional[Literal["true", "false"]] = None
    seizures: Optional[Literal["true", "false"]] = None

    @validator('neurological_ros__c', always=True)
    def validate_ros_status(cls, v, values):
        symptoms = ['cognitive_impairment', 'numbness', 'tingling', 'prickling',
                    'burning_sensation', 'itching_sensation', 'pains_and_needles',
                    'dizziness', 'light_headedness', 'vertigo', 'fainting',
                    'loss_of_balance', 'memory_problem', 'difficulty_speaking',
                    'motor_weakness', 'seizures']
        if any(values.get(f) is not None for f in symptoms):
            return "Assessed"
        return v

    @validator('reviewed_and_negative')
    def validate_reviewed(cls, v, values):
        symptoms = ['cognitive_impairment', 'numbness', 'tingling', 'prickling',
                    'burning_sensation', 'itching_sensation', 'pains_and_needles',
                    'dizziness', 'light_headedness', 'vertigo', 'fainting',
                    'loss_of_balance', 'memory_problem', 'difficulty_speaking',
                    'motor_weakness', 'seizures']
        if v is not None and any(values.get(f) is not None for f in symptoms):
            raise ValueError("Cannot have reviewed_and_negative with symptoms")
        return v

    @validator('cognitive_impairment_type')
    def validate_cognitive_type(cls, v, values):
        if values.get('cognitive_impairment') == 'true' and v is None:
            raise ValueError("cognitive_impairment_type required when impairment present")
        return v


class SOAPNote(BaseModel):
    constitutional: Optional[ConstitutionalROS] = None
    neurological: Optional[NeurologicalROS] = None


# ============== FAISS Vector Database ==============
class FAISSGuidelineDB:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.guidelines = []
        self._initialize_db()

    def _initialize_db(self):
        """Load or create FAISS index with guidelines"""
        try:
            # Load existing index
            self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
            with open(Config.GUIDELINES_PATH, 'r') as f:
                self.guidelines = json.load(f)
            logger.info("Loaded existing FAISS index and guidelines")
        except:
            # Create new index with sample guidelines
            self._create_sample_guidelines()
            self._build_index()
            logger.info("Created new FAISS index with sample guidelines")

    def _create_sample_guidelines(self):
        """Create sample ROS guidelines"""
        self.guidelines = [
            {
                "id": "1",
                "text": "If Constitutional ROS is 'Not Assessed', include only not_assessed_reason ('Cognitive Impairment' or 'Patient Refused') and no other fields",
                "category": "constitutional",
                "rules": {
                    "condition": "constructional_ros__c == 'Not Assessed'",
                    "required_fields": ["not_assessed_reason"],
                    "excluded_fields": ["fever", "chills", "fatigue", "reviewed_with_negative"]
                }
            },
            {
                "id": "2",
                "text": "If any constitutional symptom exists ( fever, chills, fatigue, change_in_sleep, change_in_appetite, unintentional_weight_gain, unintentional_weight_loss,night_sweats, weakness), set constructional_ros__c to 'Assessed' and exclude reviewed_with_negative",
                "category": "constitutional",
                "rules": {
                    "condition": "any_symptom_present",
                    "required_fields": ["constructional_ros__c = 'Assessed'"],
                    "excluded_fields": ["reviewed_with_negative"]
                }
            },
            {
                "id": "3",
                "text": "If Neurological ROS is 'Not Assessed', include only not_assessed_reason ('Cognitive Impairment' or 'Patient Refused' or 'other') and no other fields",
                "category": "neurological",
                "rules": {
                    "condition": "neurological_ros__c == 'Not Assessed'",
                    "required_fields": ["not_assessed_reason"],
                    "excluded_fields": ["cognitive_impairment", "numbness", "reviewed_and_negative"]
                }
            },
            {
                "id": "4",
                "text": "If any neurological symptom exists, set neurological_ros__c to 'Assessed' and exclude reviewed_and_negative",
                "category": "neurological",
                "rules": {
                    "condition": "any_symptom_present",
                    "required_fields": ["neurological_ros__c = 'Assessed'"],
                    "excluded_fields": ["reviewed_and_negative"]
                }
            },
            {
                "id": "5",
                "text": "cognitive_impairment_type is required when cognitive_impairment is 'true'",
                "category": "neurological",
                "rules": {
                    "condition": "cognitive_impairment == 'true'",
                    "required_fields": ["cognitive_impairment_type"]
                }
            }
        ]
        with open(Config.GUIDELINES_PATH, 'w') as f:
            json.dump(self.guidelines, f)

    def _build_index(self):
        """Build FAISS index from guidelines"""
        texts = [g["text"] for g in self.guidelines]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, Config.FAISS_INDEX_PATH)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant guidelines"""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                guideline = self.guidelines[idx]
                guideline["score"] = float(score)
                results.append(guideline)

        return results


# ============== RAG Service ==============
class ROSRAGService:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.guideline_db = FAISSGuidelineDB()
        logger.info("RAG Service initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_note(self, prompt: str) -> dict:
        """Generate note with retry logic"""
        response = self.openai_client.chat.completions.create(
            model=Config.COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)

    def _build_prompt(self, text: str, guidelines: List[Dict]) -> str:
        """Construct RAG-augmented prompt"""
        guideline_text = "\n".join([f"- {g['text']}" for g in guidelines])

        return f"""
        Generate a SOAP note JSON based on these guidelines:
        {guideline_text}

        For this clinical text:
        {text}

        STRICT RULES:
        1. Omit entire sections if they have no data
        2. Example outputs:
            - With data: {{"neurological": {{"cognitive_impairment": "true"}}}}
            - Empty: {{}} or {{"constitutional": null}}
        3. Constitutional ROS:
           - 'Not Assessed': Only include not_assessed_reason
           - Symptoms present: Auto-set to 'Assessed', exclude reviewed_with_negative
           - Only include explicitly mentioned fields

        4. Neurological ROS:
           - 'Not Assessed': Only include not_assessed_reason
           - Symptoms present: Auto-set to 'Assessed', exclude reviewed_and_negative
           - cognitive_impairment_type required if cognitive_impairment=true
           - Only include explicitly mentioned fields

        Respond with JSON only, following this structure:
        {{
        "constitutional": {{
            "constructional_ros__c": "Assessed"|"Not Assessed",
            "not_assessed_reason": null|"Cognitive Impairment"|"Patient Refused",
            "fever": null|"true"|"false",
            "chills": null|"true"|"false",
            "fatigue": null|"true"| "false",
            "change_in_sleep": null|"true"|"false"
            "change_in_appetite": null|"true"|"false"
            "unintentional_weight_gain": null|"true"|"false"
            "unintentional_weight_loss": null|"true"|"false"
            "night_sweats": null|"true"|"false"
            "weakness": null|"true"|"false"
        }},
        "neurological": {{
            "neurological_ros__c": null|"Assessed"|"Not Assessed",
            "reviewed_and_negative": null|"true"|"false",
            "cognitive_impairment: null|"true"|"false"
            "cognitive_impairment_type": null,
            "numbness": null|"true"|"false",
            "tingling": null|"true"|"false",
            "prickling": null|"true"|"false",
            "burning_sensation": null|"true"|"false",
            "itching_sensation": null|"true"|"false"
            "pains_and_needles": null|"true"|"false",
            "pain_d_t_innocuous_stimuli": null|"true"|"false",
            "increased_sensitivity_to_pain": null|"true"|"false",
            "dizziness": null|"true"|"false",
            "light_headedness": null|"true"|"false",
            "vertigo": null|"true"|"false",
            "fainting": null|"true"|"false",
            "loss_of_balance": null|"true"|"false",
            "memory_problem": null|"true"|"false",
            "difficulty_speaking": null|"true"|"false",
            "motor_weakness": null|"true"|"false",
            "seizures": null|"true"|"false"
        }}
        }}
        """

    def _clean_response(self, response: dict) -> dict:
        """Enforce business rules in the generated response"""
        cleaned = {}

        # Process constitutional ROS
        if "constitutional" in response:
            const = response["constitutional"]

            if const.get("constructional_ros__c") == "Not Assessed":
                cleaned["constitutional"] = {
                    "constructional_ros__c": "Not Assessed",
                    "not_assessed_reason": const.get("not_assessed_reason")
                }
            else:
                symptoms = ['fever', 'chills', 'fatigue', 'change_in_sleep',
                            'change_in_appetite', 'unintentional_weight_gain',
                            'unintentional_weight_loss', 'night_sweats', 'weakness']
                if any(f in const for f in symptoms):
                    const.pop("reviewed_with_negative", None)
                    const["constructional_ros__c"] = "Assessed"
                cleaned["constitutional"] = const

        # Process neurological ROS
        if "neurological" in response:
            neuro = response["neurological"]

            if neuro.get("neurological_ros__c") == "Not Assessed":
                cleaned["neurological"] = {
                    "neurological_ros__c": "Not Assessed",
                    "not_assessed_reason": neuro.get("not_assessed_reason")
                }
            else:
                symptoms = ['cognitive_impairment', 'numbness', 'tingling',
                            'dizziness', 'memory_problem', 'seizures']
                if any(f in neuro for f in symptoms):
                    neuro.pop("reviewed_and_negative", None)
                    neuro["neurological_ros__c"] = "Assessed"
                cleaned["neurological"] = neuro

        return cleaned

    def generate_soap_note(self, clinical_text: str) -> dict:
        """End-to-end SOAP note generation"""
        try:
            # Step 1: Retrieve relevant guidelines
            guidelines = self.guideline_db.search(clinical_text)
            logger.info(f"Retrieved {len(guidelines)} guidelines")

            # Step 2: Build RAG-augmented prompt
            prompt = self._build_prompt(clinical_text, guidelines)
            logger.debug(f"Prompt: {prompt[:500]}...")
            print(f"Prompt: {prompt}")
            # Step 3: Generate note
            raw_response = self._generate_note(prompt)
            logger.debug(f"Raw response: {raw_response}")
            print(f"Raw response: {raw_response}")
            # Step 4: Clean and validate
            cleaned_response = self._clean_response(raw_response)
            print(f"cleaned_response: {cleaned_response}")
            validated_note = SOAPNote(**cleaned_response).model_dump(exclude_unset=True, exclude_none=True)

            # Audit log
            self._log_generation(clinical_text, guidelines, prompt, raw_response, validated_note)
            print(f"validate_nite : {validated_note}")
            return validated_note
        except Exception as e:
            logger.error(f"SOAP note generation failed: {str(e)}")
            raise

    def _log_generation(self, text: str, guidelines: List[Dict], prompt: str, raw: dict, final: dict):
        """Log generation details for audit trail"""
        log_entry = {
            "input_text": text,
            "retrieved_guidelines": guidelines,
            "prompt": prompt,
            "raw_response": raw,
            "final_output": final,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Generation log: {json.dumps(log_entry, indent=2)}")
        print(f"Generation log: {json.dumps(log_entry, indent=2)}")


# ============== FastAPI Application ==============
app = FastAPI(title="ROS Documentation Service with FAISS")

service = ROSRAGService()
class Message(BaseModel):
    clinical_text: str

@app.post("/generate")
async def generate_note(message: Message):
    try:
        response = service.generate_soap_note(message.clinical_text)
        print(f"final response: {response} ")
        return response
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# ============== Run the Application ==============
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
