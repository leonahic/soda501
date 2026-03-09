
#Question 3: Human-in-the-Loop Evaluation
#Both spot-audits and precision/recall evaluations are necessary because they catch different types of errors.

#Precision/Recall measures overall performance on known data but can miss failure modes related to data drift or edge cases that are not represented in the small gold standard set.

#Spot-Audits allow you to see how the model behaves on raw data, but a small random sample might miss rare but critical errors, like a 1-in-100 failure to detect a specific violent event, that a systematic evaluation would catch

import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

# Question 4
model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

json_template = {
    "doc_id": "doc_XXX",
    "event_type": "other",
    "event_date_iso": None,
    "date_is_approximate": False,
    "country": None,
    "admin1_or_state": None,
    "city_or_local": None,
    "geo_precision": "unknown",
    "actors": [],
    "outcome_summary": None,
    "extraction_confidence": 0.5,
    "uncertainty_flags": [],
    "evidence": [
        {"field": "event_type", "quote": ""},
        {"field": "date", "quote": ""},
        {"field": "location", "quote": ""},
        {"field": "actors", "quote": ""},
        {"field": "outcome", "quote": ""}
    ]
}

system_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Output MUST be valid JSON only, exactly matching the provided keys.\n"
    "Allowed event_type: protest, election, policy_change, violence, disaster, other.\n"
    "If unknown: use null for optional fields.\n"
    "Include short evidence quotes from the text to support your extraction.\n"
)

docs = [
    {"doc_id": "doc_001", "text": "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested."},
    {"doc_id": "doc_002", "text": "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote."},
    {"doc_id": "doc_003", "text": "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital."},
    {"doc_id": "doc_004", "text": "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents."},
    {"doc_id": "doc_005", "text": "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear."},
    {"doc_id": "doc_006", "text": "The governor announced a new curfew order effective immediately. Critics called it an overreach."},
    {"doc_id": "doc_007", "text": "Early April saw renewed demonstrations in the northern province after fuel prices rose again."},
    {"doc_id": "doc_008", "text": "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said."},
    {"doc_id": "doc_009", "text": "Opposition leaders met with international observers in Brussels to discuss election monitoring."},
    {"doc_id": "doc_010", "text": "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm."},
    {"doc_id": "doc_011", "text": "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility."},
    {"doc_id": "doc_012", "text": "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings."},
    {"doc_id": "doc_013", "text": "Following a landslide, the ministry declared a state of emergency in two districts."},
    {"doc_id": "doc_014", "text": "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision."},
    {"doc_id": "doc_015", "text": "The article mentions reforms and elections in passing but gives no clear time or place."},
]

docs_df = pd.DataFrame(docs)

extractions = []
parse_fail_count = 0

for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]
    prompt = (
        f"{system_instructions}\n"
        f"JSON template:\n{json.dumps(json_template)}\n\n"
        f"Text: {text}\n\n"
        "Return JSON only."
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id 
        )        
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    try:
        match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        json_str = match.group(0) if match else raw_output
        parsed_data = json.loads(json_str) 
        data = json_template.copy()
        data.update(parsed_data)
        data["parse_failed"] = False
        
        if "event_type" not in data or not data["event_type"]:
            data["event_type"] = "other"
            
    except json.JSONDecodeError:
        parse_fail_count += 1
        data = json_template.copy() 
        data["parse_failed"] = True
        data["uncertainty_flags"] = ["parse_failure"]
        data["event_type"] = "other"

    data["doc_id"] = doc_id
    data["raw_text"] = text
    data["model_raw_output"] = raw_output 
    data["evidence_json"] = json.dumps(data.get("evidence", []), ensure_ascii=False)
    data["uncertainty_flags_json"] = json.dumps(data.get("uncertainty_flags", []), ensure_ascii=False)
    data.pop("evidence", None)
    data.pop("uncertainty_flags", None)
    
    extractions.append(data)

extractions_df = pd.DataFrame(extractions)
extractions_df.to_csv("Week 7/outputs/extractions_raw.csv", index=False)

#For this extraction task, I used the Qwen/Qwen2.5-1.5B-Instruct model. Out of the 15 documents processed, there were 0 parse failures. To handle potential invalid JSON outputs, I implemented a try...except block in Python to catch json.JSONDecodeError exceptions. If the model generated broken or incomplete JSON, the script was designed to catch the error, fall back to a default empty dictionary template, flag the row with parse_failed: True, and set the event_type to other. This fallback mechanism ensures that a single bad generation does not crash the entire extraction pipeline.

#Task: Extract ONE event record from the text.
#Output MUST be valid JSON only, exactly matching the provided keys.
#Allowed event_type: protest, election, policy_change, violence, disaster, other.
#If unknown: use null for optional fields.
#Include short evidence quotes from the text to support your extraction.
#JSON template:
#{"doc_id": "doc_XXX", "event_type": "other", "event_date_iso": null, "date_is_approximate": false, "country": null, "admin1_or_state": null, "city_or_local": null, "geo_precision": "unknown", "actors": [], "outcome_summary": null, "extraction_confidence": 0.5, "uncertainty_flags": [], "evidence": [{"field": "event_type", "quote": ""}, {"field": "date", "quote": ""}, {"field": "location", "quote": ""}, {"field": "actors", "quote": ""}, {"field": "outcome", "quote": ""}]}
#Text: {text}
#Return JSON only.


# Question 5

extractions_df["flag_low_conf"] = pd.to_numeric(extractions_df["extraction_confidence"], errors='coerce') < 0.7
extractions_df["flag_missing_date"] = extractions_df["event_date_iso"].isna()
extractions_df["flag_missing_country"] = extractions_df["country"].isna()
extractions_df["flag_geo_unknown"] = extractions_df["geo_precision"].isin(["unknown", "country_only"])
extractions_df["flag_parse_fail"] = extractions_df["parse_failed"]

flag_cols = ["flag_low_conf", "flag_missing_date", "flag_missing_country", "flag_geo_unknown", "flag_parse_fail"]
extractions_df["needs_human_review"] = extractions_df[flag_cols].any(axis=1)

audit_cols = [
    "doc_id", 
    "raw_text", 
    "event_type", 
    "event_date_iso", 
    "country", 
    "actors", 
    "evidence_json",
    "model_raw_output",
    "needs_human_review"
]

audit_sample = extractions_df[audit_cols].sample(n=5, random_state=42)

audit_sample["human_is_correct"] = ""      
audit_sample["human_correct_event"] = ""   
audit_sample["human_correct_date"] = ""    
audit_sample["failure_mode"] = ""         
audit_sample["reviewer_notes"] = ""

audit_sample.to_csv("Week 7/outputs/human_audit_sheet.csv", index=False)

#The human-in-the-loop audit revealed that while the model correctly identified the primary event type in all five sampled cases, it frequently hallucinated specific timestamps and geographic locations that were not present in the source text. Consequently, the audit yielded a 40% accuracy rate for complete record extraction. I identified hallucination as the primary failure cause.

