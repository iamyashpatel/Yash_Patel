import json
import torch
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import openai


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_BANK = "bank_policy"
COLLECTION_CLIENT = "client_policy"
TOP_K = 1  
OPENAI_API_KEY = "Sk........."  # Replace with your actual API key


connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


bank_collection = Collection(COLLECTION_BANK)
client_collection = Collection(COLLECTION_CLIENT)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-m3", device=device)

bank_collection.load()
bank_clauses = bank_collection.query(expr="id >= 0", output_fields=["text"], limit=1000)
bank_texts = [item["text"] for item in bank_clauses]

client_collection.load()

results_json = []

def generate_analysis_with_gpt(prompt):
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

for bank_text in bank_texts:
    question_prompt = f"""
    **Bank Policy Clause:**  
    {bank_text}  

    **Task:**  
    Generate a verification question that must be asked to the client to check compliance.
    """
    verification_question = generate_analysis_with_gpt(question_prompt)

    
    query_embedding = model.encode(verification_question, convert_to_tensor=False).tolist()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = client_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["text"]
    )
    
    if results and results[0]:
        client_answer = results[0][0].entity.get("text")
    else:
        client_answer = "No relevant clause found in the client policy."

    prompt = f"""
    **Verification Question:**  
    {verification_question}  

    **Bank's Expected Clause:**  
    {bank_text}  

    **Client's Provided Answer:**  
    {client_answer}  

    ### 🔍 **Analysis**:  
    1️⃣ **Compliance Check:**  
    - Does the client's response align with the bank's policy clause? (Yes/No)  
    - If not, explain why and highlight any inconsistencies.  

    2️⃣ **Differences Between Bank & Client:**  
    - List key points **present in the bank policy but missing in the client policy**.  
    - List key points **present in the client policy but missing in the bank policy**.  

    3️⃣ **Summary of Differences:**  
    - Provide a **concise summary** of key discrepancies.  
    - Suggest possible **resolutions** to ensure compliance.  

    ### ✅ **Structured Output:**  
    """
    
    analysis = generate_analysis_with_gpt(prompt)
    
    if analysis:
        result_entry = {
            "verification_question": verification_question,
            "bank_policy_clause": bank_text,
            "client_policy_clause": client_answer,
            "analysis": analysis
        }
        results_json.append(result_entry)

with open("policy_compliance_check1.json", "w", encoding="utf-8") as json_file:
    json.dump(results_json, json_file, indent=4, ensure_ascii=False)

print("✅ Policy compliance check completed and saved successfully.")
